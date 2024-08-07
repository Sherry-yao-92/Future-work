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
#include <map>
#include <algorithm>
#include <iomanip>

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

void process_single_image(const string& image_path, const Mat& blurred_bg, vector<vector<Point>>& contours, ContourMetrics& metrics, double& duration) {
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

    vector<Vec4i> hierarchy;
    findContours(dilate2, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

    auto end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration<double, micro>(end_time - start_time).count();

    if (!contours.empty()) {
        metrics = calculate_contour_metrics(contours);
    }
}

void run_experiment(const string& directory, const Mat& blurred_bg, double& max_processing_time, string& max_processing_time_image) {
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
                    process_single_image(path.string(), blurred_bg, contours, metrics, process_time);

                    if (process_time > 0) {  // 只處理有效的圖片
                        lock_guard<mutex> lock(mtx);
                        total_time = total_time + process_time;
                        if (process_time > max_process_time) {
                            max_process_time = process_time;
                            max_processing_time_image = path.filename().string();
                        }
                        number = number + 1;
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

    max_processing_time = max_process_time;
}

void print_progress_bar(int progress, int total) {
    const int bar_width = 70;
    float progress_ratio = static_cast<float>(progress) / total;
    int pos = static_cast<int>(bar_width * progress_ratio);

    cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress_ratio * 100.0) << "% \r";
    cout.flush();
}

int main() {
    string directory = "Test_images/Cropped";
    string background_path = directory + "/background.tiff";
    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    if (background.empty()) {
        cerr << "Error: Could not read background image: " << background_path << endl;
        return -1;
    }
    Mat blurred_bg;
    GaussianBlur(background, blurred_bg, Size(5, 5), 0);

    map<string, int> image_count;
    const int total_iterations = 1000;

    for (int i = 0; i < total_iterations; ++i) {
        double current_max_time = 0;
        string current_max_image;
        run_experiment(directory, blurred_bg, current_max_time, current_max_image);

        if (!current_max_image.empty()) {
            image_count[current_max_image] = image_count[current_max_image] + 1;
        }

        // 更新進度條
        print_progress_bar(i + 1, total_iterations);
    }

    cout << endl; // 進度條完成後換行

    // 將 map 轉換為 vector 以便排序
    vector<pair<string, int>> image_count_vec(image_count.begin(), image_count.end());

    // 按出現次數降序排序
    sort(image_count_vec.begin(), image_count_vec.end(),
         [](const pair<string, int>& a, const pair<string, int>& b) {
             return a.second > b.second;
         });

    // 輸出出現最長時間前五次數最多的圖片
    cout << "Top 5 images that appeared most frequently as the one with the longest processing time:" << endl;
    for (int i = 0; i < min(5, static_cast<int>(image_count_vec.size())); ++i) {
        cout << i+1 << ". " << image_count_vec[i].first << ": " 
             << image_count_vec[i].second << " occurrences" << endl;
    }

    return 0;
}
