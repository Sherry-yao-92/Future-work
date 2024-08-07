#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_queue.h>
#include <atomic>
#include <mutex>

#define _USE_MATH_DEFINES
#include <math.h>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

struct ContourMetrics {
    double area_original = 0;
    double area_hull = 0;
    double area_ratio = 0;
    double circularity_original = 0;
    double circularity_hull = 0;
    double circularity_ratio = 0;
};

ContourMetrics calculate_contour_metrics(const vector<vector<Point>>& contours) {
    if (contours.empty()) {
        return ContourMetrics();
    }

    auto cnt = *max_element(contours.begin(), contours.end(),
                            [](const auto& c1, const auto& c2) { return contourArea(c1) < contourArea(c2); });

    double area_original = contourArea(cnt);
    double perimeter_original = arcLength(cnt, true);

    if (area_original <= 1e-6 || perimeter_original <= 1e-6) {
        return ContourMetrics();
    }

    double circularity_original = 4 * M_PI * area_original / (perimeter_original * perimeter_original);

    vector<Point> hull;
    convexHull(cnt, hull);

    double area_hull = contourArea(hull);
    double perimeter_hull = arcLength(hull, true);

    if (area_hull <= 1e-6 || perimeter_hull <= 1e-6) {
        return ContourMetrics();
    }

    double circularity_hull = 4 * M_PI * area_hull / (perimeter_hull * perimeter_hull);

    ContourMetrics results;
    results.area_original = area_original;
    results.area_hull = area_hull;
    results.area_ratio = area_hull / area_original;
    results.circularity_original = circularity_original;
    results.circularity_hull = circularity_hull;
    results.circularity_ratio = circularity_hull / circularity_original;

    return results;
}

bool process_single_image(const string& image_path, const Mat& blurred_bg, vector<vector<Point>>& contours, ContourMetrics& metrics, double& duration) {
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    auto start_time = chrono::high_resolution_clock::now();

    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 0);
    Mat bg_sub;
    subtract(blurred_bg, blurred, bg_sub);
    Mat binary;
    threshold(bg_sub, binary, 10, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat dilate1, erode1, dilate2;
    dilate(binary, dilate1, kernel, Point(-1, -1), 2);
    erode(dilate1, erode1, kernel, Point(-1, -1), 3);
    dilate(erode1, dilate2, kernel, Point(-1, -1), 1);

    // 檢查處理時間是否超過200微秒
    auto current_time = chrono::high_resolution_clock::now();
    duration = chrono::duration<double, micro>(current_time - start_time).count();
    if (duration > 200) {
        return false;
    }

    vector<Vec4i> hierarchy;
    findContours(dilate2, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    current_time = chrono::high_resolution_clock::now();
    duration = chrono::duration<double, micro>(current_time - start_time).count();

    if (duration > 200) {
        return false;
    }

    if (!contours.empty()) {
        metrics = calculate_contour_metrics(contours);
    }

    return true;
}

void run_experiment(string directory, vector<tuple<string, double, double, double>>& results, vector<pair<string, double>>& skipped_images, pair<string, double>& max_time_image) {
    string background_path = directory + "/background.tiff";
    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    if (background.empty()) {
        cerr << "Error: Could not read background image: " << background_path << endl;
        return;
    }
    Mat blurred_bg;
    GaussianBlur(background, blurred_bg, Size(5, 5), 0);

    atomic<double> total_time(0);
    atomic<int> number(0);
    atomic<double> max_process_time(0);
    mutex mtx;

    tbb::task_arena arena(thread::hardware_concurrency());
    tbb::task_group group;
    tbb::concurrent_queue<fs::path> image_queue;

    atomic<bool> processing_complete(false);

    arena.execute([&]() {
        group.run([&]() {
            while (!processing_complete || !image_queue.empty()) {
                fs::path path;
                if (image_queue.try_pop(path)) {
                    vector<vector<Point>> contours;
                    ContourMetrics metrics;
                    double process_time;
                    bool processed = process_single_image(path.string(), blurred_bg, contours, metrics, process_time);

                    if (processed) {  // 只處理有效的圖片
                        lock_guard<mutex> lock(mtx);
                        total_time = total_time + process_time;
                        if (process_time > max_process_time) {
                            max_process_time = process_time;
                            max_time_image = { path.string(), process_time };
                        }
                        number++;
                        results.push_back(make_tuple(path.string(), metrics.circularity_ratio, metrics.area_ratio, process_time));
                    } else {
                        lock_guard<mutex> lock(mtx);
                        skipped_images.push_back({ path.string(), process_time });
                    }
                } else {
                    this_thread::yield();
                }
            }
        });
    });

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            image_queue.push(entry.path());
        }
    }

    processing_complete = true;
    group.wait();
}

int main() {
    string directory = "Test_images/Cropped";
    vector<tuple<string, double, double, double>> results;
    vector<pair<string, double>> skipped_images;
    pair<string, double> max_time_image;

    run_experiment(directory, results, skipped_images, max_time_image);

    // 輸出結果
    cout << "Circularity ratio and area ratio for each processed image:" << endl;
    for (const auto& result : results) {
        fs::path path(get<0>(result));
        cout << "Image: " << path.filename().string() << endl;
        cout << "  Circularity ratio: " << get<1>(result) << ", Area ratio: " << get<2>(result) << endl;
        cout << "  Processing time: " << get<3>(result) << " microseconds" << endl;
        cout << "\n";
    }

    cout << "\nSkipped images:" << endl;
    for (const auto& image : skipped_images) {
        fs::path path(image.first);
        cout << "Image: " << path.filename().string() << " with processing time: " << image.second << " microseconds" << endl;
    }

    double total_time = 0;
    for (const auto& result : results) {
        total_time += get<3>(result);
    }
    double average_time = results.empty() ? 0 : total_time / results.size();

    cout << "\nAverage processing time: " << average_time << " microseconds" << endl;
    
    fs::path max_time_path(max_time_image.first);
    cout << "Max processing time: " << max_time_image.second << " microseconds for image: " 
         << max_time_path.filename().string() << endl;

    return 0;
}
