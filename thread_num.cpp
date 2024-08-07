#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
namespace fs = std::filesystem;
using namespace cv;
using namespace std;
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdio> //含c語言printf
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <numeric>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tbb/concurrent_queue.h>
#include <atomic>
#include <fstream>

struct ContourMetrics {
    double area_original;
    double area_hull;
    double area_ratio;
    double circularity_original;
    double circularity_hull;
    double circularity_ratio;
};

ContourMetrics calculate_contour_metrics(const vector<vector<Point>>& contours)         //取contours位置的值作為輸入，且形式是多輪廓的向量
    {if (contours.empty()) {
        printf("No contours to calculate metrics.\n");
        return ContourMetrics();
    }
    auto cnt = *max_element(contours.begin(), contours.end(),
    [](const auto& c1, const auto& c2) { return contourArea(c1) < contourArea(c2); });//這邊是一個一個比，或許可優化(並行)
    
    double area_original = contourArea(cnt);
    double perimeter_original = arcLength(cnt, true);

    if (area_original <= 1e-6 || perimeter_original <= 1e-6) {
        printf("Invalid contour measurements: area=%f, perimeter=%f\n", area_original, perimeter_original);
        return ContourMetrics();
    }

    double circularity_original = 2 * sqrt(M_PI * area_original) / perimeter_original;

    //凸包處理
    vector<Point> hull; //創建hull向量儲存cvpoint
    convexHull(cnt, hull); //hull=convexHull(cnt)

    double area_hull = contourArea(hull);
    double perimeter_hull = arcLength(hull, true);

    if (area_hull <= 1e-6 || perimeter_hull <= 1e-6) {
        printf("Invalid hull measurements: area=%f, perimeter=%f\n",area_hull,perimeter_hull);
        return ContourMetrics();
    }

    double circularity_hull = 2 * sqrt(M_PI * area_hull) / perimeter_hull;

    ContourMetrics results;
    results.area_original = area_original;
    results.area_hull = area_hull;
    results.area_ratio = area_hull / area_original;
    results.circularity_original = circularity_original;
    results.circularity_hull = circularity_hull;
    results.circularity_ratio = circularity_hull / circularity_original;
    return results;
}

void process_image_origin(const string& image_path, const Mat& blurred_bg,double& numbersum, vector<vector<Point>>& contours, ContourMetrics& metrics, double& duration,double& number)  //constant為輸入值，非constant為輸出
    {
    Mat image = imread(image_path, IMREAD_GRAYSCALE);

    auto start_time = std::chrono::high_resolution_clock::now();
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 0);

    Mat bg_sub;
    subtract(blurred_bg, blurred, bg_sub);

    Mat binary;
    threshold(bg_sub, binary, 10, 255, THRESH_BINARY);

    Mat dilate1, erode1, erode2, dilate2;
    cv::dilate(binary, dilate1, kernel, Point(), 2);  //point:錨點（anchor）的位置。表示使用結構元素的中心作為錨點，未指定寂寞認為(-1,-1)，默認值與python相同
    cv::erode(dilate1, erode1, kernel, Point(), 3);
    cv::dilate(erode1, dilate2, kernel, Point(), 1);

    /*Mat edges;
    cv::Canny(dilate2, edges, 50, 150);*/

    //vector<vector<Point>> contours;  //指定內部向量包含一組point點，每一個內部向量代表一個輪廓，外部向量代表多個輪廓
    vector<Vec4i> hierarchy; //hierarchy: 存儲輪廓之間的層次關係信息
    cv::findContours(dilate2, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
    if (contours.empty()) {
    printf("No contours found in the image.\n");
    return;}
    metrics = calculate_contour_metrics(contours);
    auto end_time = std::chrono::high_resolution_clock::now();
    #pragma warning(push)
    #pragma warning(disable : 4244)
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    #pragma warning(pop)
    
    //printf("process time: %f us \n", duration);
    number=numbersum+1;
}

struct ImageResult {
    string path;
    vector<vector<Point>> contours;
    ContourMetrics metrics;
    double processtime;
};

void thread_main(const string &directory,const cv::Mat& blurred_bg, double& Average_processtime_minrec_thread,double& max_processing_time_minrec_thread,std::string &max_processing_time_image_minrec_thread) {    
    std::atomic<double> totaltime_c = 0;
    max_processing_time_minrec_thread=0;
    std::mutex mtx;
    tbb::task_arena arena(8); //自行設定線程數量
    tbb::task_group group;
    tbb::concurrent_queue<fs::path> image_queue;

    // 輸出使用的線程數量
    std::cout << "Using " << arena.max_concurrency() << " threads for processing." << std::endl;

    std::atomic<bool> processing_complete(false);
    double numbersum=0,number=0;
    // 啟動處理線程
    arena.execute([&]() {
        group.run([&]() {
            while (!processing_complete || !image_queue.empty()) {
                fs::path path;
                if (image_queue.try_pop(path)) {
                    vector<vector<Point>> contours;
                    ContourMetrics metrics;
                    double processtime;
                    process_image_origin(path.string(), blurred_bg, numbersum, contours, metrics, processtime, number);
                    /*if (!contours.empty()) {
                        printf("processing: %s\n",path.filename().string().c_str());
                        printf("processtime= %f\n",processtime);
                        printf("Original area: %f\n", metrics.area_original);
                        printf("Convex Hull area: %f\n", metrics.area_hull);
                        printf("Area ratio (hull/original): %f\n", metrics.area_ratio);
                        printf("Original circularity: %f\n", metrics.circularity_original);
                        printf("Convex Hull circularity: %f\n", metrics.circularity_hull);
                        printf("Circularity ratio (hull/original): %f\n", metrics.circularity_ratio);
                        printf("\n");}*/
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        totaltime_c = totaltime_c + processtime;
                        if (processtime > max_processing_time_minrec_thread) {
                            max_processing_time_minrec_thread = processtime;
                            max_processing_time_image_minrec_thread=path.filename().string().c_str();
                        }
                    numbersum=number;
                    }
                } 
                else {
                    std::this_thread::yield();
                }
            }
            Average_processtime_minrec_thread=totaltime_c/number;
        });
    });

    // 遍歷目錄並立即分發任務
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            image_queue.push(entry.path());
        }
    }
    processing_complete = true;
    group.wait();
}

int main () {
    string directory = "Test_images/Cropped";
    string background_path = directory + "/background.tiff";
    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    Mat blurred_bg;
    GaussianBlur(background, blurred_bg, Size(5, 5), 0);
    double avrtime_o, max_processtime_o;
    std::string max_processing_time_image_o;

    // 輸出系統可用的硬件線程數
    std::cout << "Available hardware threads: " << std::thread::hardware_concurrency() << std::endl;

    thread_main(directory, blurred_bg, avrtime_o,max_processtime_o,max_processing_time_image_o);
    printf("averagetime=%f       maximum processtime= %f      max process image=%s \n",avrtime_o, max_processtime_o, max_processing_time_image_o.c_str());

    return 0;
}
