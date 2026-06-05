#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// 缩放显示函数
Mat resizeForDisplay(const Mat &img, int max_width = 800) {
    if (img.empty()) return img;
    int w = img.cols;
    int h = img.rows;
    if (w > max_width) {
        double scale = static_cast<double>(max_width) / w;
        int new_w = static_cast<int>(w * scale);
        int new_h = static_cast<int>(h * scale);
        Mat resized;
        resize(img, resized, Size(new_w, new_h));
        return resized;
    }
    return img.clone();
}


int main() {
    cout << "DISPLAY = " << (getenv("DISPLAY") ? getenv("DISPLAY") : "not set") << endl;
    // 1. 读取图片
    string image_path;
    cout << "请输入图片路径: ";
    getline(cin, image_path);

    Mat img = imread(image_path);
    if (img.empty()) {
        cout << "错误：无法读取图片，请检查图片路径！" << endl;
        return -1;
    }

    // 2. 输出图像信息
    cout << "图像尺寸（宽 x 高）: " << img.cols << " x " << img.rows << endl;
    cout << "图像通道数: " << img.channels() << endl;
    int depth = img.depth();
    string type_str;
    switch (depth) {
        case CV_8U:  type_str = "CV_8U"; break;
        case CV_8S:  type_str = "CV_8S"; break;
        case CV_16U: type_str = "CV_16U"; break;
        case CV_16S: type_str = "CV_16S"; break;
        case CV_32S: type_str = "CV_32S"; break;
        case CV_32F: type_str = "CV_32F"; break;
        case CV_64F: type_str = "CV_64F"; break;
        default:     type_str = "Unknown";
    }
    cout << "像素数据类型: " << type_str << endl;

    // 3. 显示原图
    Mat display_img = resizeForDisplay(img);
    imshow("Original Image", display_img);
    cout << "显示原图，按任意键继续..." << endl;
    waitKey(0);
    destroyAllWindows();

    // 4. 灰度图处理并显示
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    Mat display_gray = resizeForDisplay(gray_img);
    imshow("Grayscale Image", display_gray);
    cout << "显示灰度图，按任意键继续..." << endl;
    waitKey(0);
    destroyAllWindows();

    // 5. 保存灰度图
    string output_gray_path = "gray_image.jpg";
    imwrite(output_gray_path, gray_img);
    cout << "灰度图已保存为: " << output_gray_path << endl;

    // 6. 裁剪并显示
    int crop_size = 50;
    Rect crop_rect(0, 0, crop_size, crop_size);
    Mat cropped = img(crop_rect);
    imshow("Cropped Region", cropped);
    cout << "显示裁剪区域，按任意键继续..." << endl;
    waitKey(0);
    destroyAllWindows();
    string crop_path = "cropped_region.jpg";
    imwrite(crop_path, cropped);
    cout << "裁剪区域已保存为: " << crop_path << endl;

    // 7. 修改图像（左上角涂黑）并显示
    Mat img_copy = img.clone();
    img_copy(crop_rect) = Scalar(0, 0, 0);
    Mat display_modified = resizeForDisplay(img_copy);
    imshow("Modified Image (Black Corner)", display_modified);
    cout << "显示修改后的图像，按任意键继续..." << endl;
    waitKey(0);
    destroyAllWindows();
    string modified_path = "modified_image.jpg";
    imwrite(modified_path, img_copy);
    cout << "修改后的图像已保存为: " << modified_path << endl;

    cout << "所有任务完成！" << endl;
    return 0;
}
