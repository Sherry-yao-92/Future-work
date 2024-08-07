#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <numeric>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

struct ContourMetrics {
    double area_original;
    double area_hull;
    double area_ratio;
    double circularity_original;
    double circularity_hull;
    double circularity_ratio;
    vector<Point> contour;
    vector<Point> hull;
};

ContourMetrics calculate_contour_metrics(const vector<vector<Point>>& contours) {
    ContourMetrics results;
    if (contours.empty()) {
        return results;
    }

    auto cnt = *max_element(contours.begin(), contours.end(),
        [](const vector<Point>& c1, const vector<Point>& c2) {
            return contourArea(c1) < contourArea(c2);
        });

    results.area_original = contourArea(cnt);
    double perimeter_original = arcLength(cnt, true);
    results.circularity_original = (2 * sqrt(M_PI * results.area_original)) / perimeter_original;

    convexHull(cnt, results.hull);

    results.area_hull = contourArea(results.hull);
    double perimeter_hull = arcLength(results.hull, true);
    results.circularity_hull = (2 * sqrt(M_PI * results.area_hull)) / perimeter_hull;

    results.area_ratio = results.area_hull / results.area_original;
    results.circularity_ratio = results.circularity_hull / results.circularity_original;

    results.contour = cnt;

    return results;
}

bool is_contour_complete(const vector<Point>& contour, const Size& image_size) {
    for (const Point& point : contour) {
        if (point.x <= 0 || point.y <= 0 || point.x >= image_size.width - 1 || point.y >= image_size.height - 1) {
            return false;
        }
    }
    return true;
}

ContourMetrics process_image(const Mat& img, const Mat& background, bool use_canny) {
    Mat blur_img, blur_background;
    GaussianBlur(img, blur_img, Size(3, 3), 0);
    GaussianBlur(background, blur_background, Size(3, 3), 0);

    Mat substract;
    subtract(blur_background, blur_img, substract);

    Mat binary;
    threshold(substract, binary, 10, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

    Mat erode1, dilate1, dilate2, erode2;
    erode(binary, erode1, kernel);
    dilate(erode1, dilate1, kernel);
    dilate(dilate1, dilate2, kernel);
    erode(dilate2, erode2, kernel);

    Mat edge;
    if (use_canny) {
        Canny(erode2, edge, 50, 150);
    } else {
        edge = erode2;
    }

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    if (contours.empty() || contours.size() > 1 || !is_contour_complete(contours[0], img.size())) {
        return ContourMetrics();
    }

    return calculate_contour_metrics(contours);
}

void process_and_compare(const string& img_path, const Mat& background) {
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Error: Unable to read image: " << img_path << endl;
        return;
    }

    auto start_time_with_canny = high_resolution_clock::now();
    ContourMetrics results_with_canny = process_image(img, background, true);
    auto end_time_with_canny = high_resolution_clock::now();

    if (results_with_canny.contour.empty()) {
        cout << "Skipping " << fs::path(img_path).filename() << " due to no contours found, multiple contours, or incomplete contours." << endl;
        return;
    }

    auto start_time_without_canny = high_resolution_clock::now();
    ContourMetrics results_without_canny = process_image(img, background, false);
    auto end_time_without_canny = high_resolution_clock::now();

    if (results_without_canny.contour.empty()) {
        cout << "Skipping " << fs::path(img_path).filename() << " due to no contours found, multiple contours, or incomplete contours." << endl;
        return;
    }

    auto process_time_with_canny = duration_cast<microseconds>(end_time_with_canny - start_time_with_canny).count() / 1e6;
    auto process_time_without_canny = duration_cast<microseconds>(end_time_without_canny - start_time_without_canny).count() / 1e6;

    cout << "Processing " << fs::path(img_path).filename() << ":" << endl;
    cout << fixed << setprecision(6);
    cout << "With Canny processing time: " << process_time_with_canny << " seconds" << endl;
    cout << "Without Canny processing time: " << process_time_without_canny << " seconds" << endl;
    cout << "With Canny area: " << results_with_canny.area_original << " | Without Canny area: " << results_without_canny.area_original << endl;
    cout << "With Canny Convex Hull area: " << results_with_canny.area_hull << " | Without Canny Convex Hull area: " << results_without_canny.area_hull << endl;
    cout << "With Canny Area ratio: " << results_with_canny.area_ratio << " | Without Canny Area ratio: " << results_without_canny.area_ratio << endl;
    cout << "With Canny circularity: " << results_with_canny.circularity_original << " | Without Canny circularity: " << results_without_canny.circularity_original << endl;
    cout << "With Canny Convex Hull circularity: " << results_with_canny.circularity_hull << " | Without Canny Convex Hull circularity: " << results_without_canny.circularity_hull << endl;
    cout << "With Canny Circularity ratio: " << results_with_canny.circularity_ratio << " | Without Canny Circularity ratio: " << results_without_canny.circularity_ratio << endl;
    cout << endl;

    // 繪製輪廓
    Mat contour_image_with_canny = Mat::zeros(img.size(), CV_8U);
    Mat hull_contour_image_with_canny = Mat::zeros(img.size(), CV_8U);
    Mat contour_image_without_canny = Mat::zeros(img.size(), CV_8U);
    Mat hull_contour_image_without_canny = Mat::zeros(img.size(), CV_8U);

    drawContours(contour_image_with_canny, vector<vector<Point>>{results_with_canny.contour}, -1, Scalar(255), 1);
    drawContours(hull_contour_image_with_canny, vector<vector<Point>>{results_with_canny.hull}, -1, Scalar(255), 1);
    drawContours(contour_image_without_canny, vector<vector<Point>>{results_without_canny.contour}, -1, Scalar(255), 1);
    drawContours(hull_contour_image_without_canny, vector<vector<Point>>{results_without_canny.hull}, -1, Scalar(255), 1);

    imshow("Contour (With Canny) - " + fs::path(img_path).filename().string(), contour_image_with_canny);
    imshow("Convex Hull (With Canny) - " + fs::path(img_path).filename().string(), hull_contour_image_with_canny);
    imshow("Contour (Without Canny) - " + fs::path(img_path).filename().string(), contour_image_without_canny);
    imshow("Convex Hull (Without Canny) - " + fs::path(img_path).filename().string(), hull_contour_image_without_canny);
    waitKey(0);
    destroyAllWindows();
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    cout << "OpenCV version: " << CV_VERSION << endl;

    string cropped_folder = "Test_images/Cropped/";
    string background_path = cropped_folder + "background.tiff";

    Mat background = imread(background_path, IMREAD_GRAYSCALE);
    if (background.empty()) {
        cout << "Error: Unable to read background image: " << background_path << endl;
        return -1;
    }

    vector<double> times_with_canny, times_without_canny;

    // 第一次遍歷：計算處理時間
    for (const auto& entry : fs::directory_iterator(cropped_folder)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            string img_path = entry.path().string();
            Mat img = imread(img_path, IMREAD_GRAYSCALE);
            if (img.empty()) {
                cout << "Error: Unable to read image: " << img_path << endl;
                continue;
            }

            auto start_time_with_canny = high_resolution_clock::now();
            ContourMetrics results_with_canny = process_image(img, background, true);
            auto end_time_with_canny = high_resolution_clock::now();

            if (!results_with_canny.contour.empty()) {
                auto start_time_without_canny = high_resolution_clock::now();
                ContourMetrics results_without_canny = process_image(img, background, false);
                auto end_time_without_canny = high_resolution_clock::now();

                if (!results_without_canny.contour.empty()) {
                    times_with_canny.push_back(duration_cast<microseconds>(end_time_with_canny - start_time_with_canny).count() / 1e6);
                    times_without_canny.push_back(duration_cast<microseconds>(end_time_without_canny - start_time_without_canny).count() / 1e6);
                }
            }
        }
    }

    // 計算並顯示平均處理時間
    if (!times_with_canny.empty() && !times_without_canny.empty()) {
        double avg_with_canny = accumulate(times_with_canny.begin(), times_with_canny.end(), 0.0) / times_with_canny.size();
        double avg_without_canny = accumulate(times_without_canny.begin(), times_without_canny.end(), 0.0) / times_without_canny.size();

        cout << fixed << setprecision(6);
        cout << "Average processing time with Canny: " << avg_with_canny << " seconds" << endl;
        cout << "Average processing time without Canny: " << avg_without_canny << " seconds" << endl;
        cout << endl;
    } else {
        cout << "No valid images processed." << endl;
    }

    // 第二次遍歷：處理圖像並顯示結果
    for (const auto& entry : fs::directory_iterator(cropped_folder)) {
        if (entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            string img_path = entry.path().string();
            process_and_compare(img_path, background);
        }
    }

    return 0;
}
