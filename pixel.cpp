#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <limits>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

struct ImageInfo {
    string filename;
    int white_pixel_count;
};

void process_single_image(const fs::path& image_path, const Mat& blurred_bg, ImageInfo& max_info, ImageInfo& min_info) {
    Mat image = imread(image_path.string(), IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Unable to open or find image: " << image_path << endl;
        return;
    }

    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 0);
    Mat bg_sub;
    subtract(blurred_bg, blurred, bg_sub);
    Mat binary;
    threshold(bg_sub, binary, 10, 255, THRESH_BINARY);

    // Count white pixels
    int white_pixel_count = countNonZero(binary);

    // Update max and min info if necessary
    if (white_pixel_count > max_info.white_pixel_count) {
        max_info.filename = image_path.filename().string();
        max_info.white_pixel_count = white_pixel_count;
    }
    if (white_pixel_count < min_info.white_pixel_count) {
        min_info.filename = image_path.filename().string();
        min_info.white_pixel_count = white_pixel_count;
    }

    // Output results for current image
    cout << "Image: " << image_path.filename().string() << endl;
    cout << "White pixel count: " << white_pixel_count << endl;
    cout << endl;
}

int main() {
    fs::path directory = "Test_images/cropped single";
    fs::path background_path = directory / "background.tiff";
    
    if (!fs::exists(directory)) {
        cerr << "Directory does not exist: " << directory << endl;
        return -1;
    }

    Mat background = imread(background_path.string(), IMREAD_GRAYSCALE);
    if (background.empty()) {
        cerr << "Unable to open or find background image: " << background_path << endl;
        return -1;
    }

    Mat blurred_bg;
    GaussianBlur(background, blurred_bg, Size(5, 5), 0);

    ImageInfo max_info = {"", 0};
    ImageInfo min_info = {"", numeric_limits<int>::max()};

    bool processed_any_file = false;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".tiff" && entry.path().filename() != "background.tiff") {
            process_single_image(entry.path(), blurred_bg, max_info, min_info);
            processed_any_file = true;
        }
    }

    if (!processed_any_file) {
        cout << "No matching image files found." << endl;
    } else {
        cout << "Image with maximum white pixels: " << max_info.filename << endl;
        cout << "Maximum white pixel count: " << max_info.white_pixel_count << endl;
        cout << endl;
        cout << "Image with minimum white pixels: " << min_info.filename << endl;
        cout << "Minimum white pixel count: " << min_info.white_pixel_count << endl;
    }

    return 0;
}
