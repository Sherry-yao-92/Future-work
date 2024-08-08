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

void process_single_image(const string& image_path, const Mat& blurred_bg, vector<vector<Point>>& contours, ContourMetrics& metrics, double& duration, double& findcontour_duration) {
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    auto start_time = chrono::high_resolution_clock::now();

    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 0);
    Mat bg_sub;
    subtract(blurred_bg, blurred, bg_sub);
    Mat binary;
    threshold(bg_sub, binary, 10, 255, THRESH_BINARY);

    int white_pixel_count = countNonZero(binary);
    
    if (white_pixel_count < 250 || white_pixel_count > 650) {
        duration = 0;
        return;
    }

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat dilate1, erode1, dilate2;
    dilate(binary, dilate1, kernel, Point(-1, -1), 2);
    erode(dilate1, erode1, kernel, Point(-1, -1), 3);
    dilate(erode1, dilate2, kernel, Point(-1, -1), 1);

    vector<Vec4i> hierarchy;
    auto findcontour_start = chrono::high_resolution_clock::now();

    findContours(dilate2, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    auto findcontour_end = chrono::high_resolution_clock::now();
    findcontour_duration = chrono::duration<double, micro>(findcontour_end - findcontour_start).count();

    auto end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration<double, micro>(end_time - start_time).count();

    if (!contours.empty()) {
        metrics = calculate_contour_metrics(contours);
    }
}

void run_experiment(string directory, vector<tuple<string, double, double, double, double>>& results, vector<string>& skipped_images, pair<string, double>& max_time_image) {
    string background_path = directory + "/background.tiff";
    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    if (background.empty()) {
        cerr << "Error: Could not read background image: " << background_path << endl;
        return;
    }
    Mat blurred_bg;
    GaussianBlur(background, blurred_bg, Size(5, 5), 0);

    atomic<double> total_time(0);
    atomic<double> total_findcontour_time(0);
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
                    double findcontour_time;
                    process_single_image(path.string(), blurred_bg, contours, metrics, process_time, findcontour_time);

                    if (process_time > 0) {
                        lock_guard<mutex> lock(mtx);
                        total_time = total_time + process_time;
                        total_findcontour_time = total_findcontour_time + findcontour_time;
                        if (process_time > max_process_time) {
                            max_process_time = process_time;
                            max_time_image = { path.string(), process_time };
                        }
                        number++;
                        results.push_back(make_tuple(path.string(), metrics.circularity_ratio, metrics.area_ratio, process_time, findcontour_time));
                    } else {
                        lock_guard<mutex> lock(mtx);
                        skipped_images.push_back(path.string());
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

void print_progress(int current, int total) {
    int barWidth = 70;
    float progress = static_cast<float>(current) / total;
    cout << "[";
    int pos = static_cast<int>(barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << static_cast<int>(std::round(progress * 100.0)) << " %\r";
    cout.flush();
}

int main() {
    string directory = "Test_images/512x96crop";
    vector<tuple<string, double, double, double, double>> results;
    vector<string> skipped_images;
    pair<string, double> max_time_image;

    const int repetitions = 10000;
    double total_circularity_ratio = 0;
    double total_area_ratio = 0;
    double total_processing_time = 0;
    double total_findcontour_time = 0;

    for (int i = 0; i < repetitions; ++i) {
        results.clear();
        skipped_images.clear();
        max_time_image = {"", 0};

        run_experiment(directory, results, skipped_images, max_time_image);

        for (const auto& result : results) {
            total_circularity_ratio += get<1>(result);
            total_area_ratio += get<2>(result);
            total_processing_time += get<3>(result);
            total_findcontour_time += get<4>(result);
        }

        print_progress(i + 1, repetitions);
    }
    cout << endl;

    size_t total_processed_images = results.size() * repetitions;

    double average_circularity_ratio = total_processed_images > 0 ? total_circularity_ratio / total_processed_images : 0;
    double average_area_ratio = total_processed_images > 0 ? total_area_ratio / total_processed_images : 0;
    double average_processing_time = total_processed_images > 0 ? total_processing_time / total_processed_images : 0;
    double average_findcontour_time = total_processed_images > 0 ? total_findcontour_time / total_processed_images : 0;

    cout << "Average Circularity Ratio: " << average_circularity_ratio << endl;
    cout << "Average Area Ratio: " << average_area_ratio << endl;
    cout << "Average Processing Time: " << average_processing_time << " microseconds" << endl;
    cout << "Average FindContours Time: " << average_findcontour_time << " microseconds" << endl;

    return 0;
}
