#include <stdio.h>
#include <opencv2/opencv.hpp>

int main(int nargs, char **args) {
    assert(nargs == 2); // given the image path

    int src_w = 383, src_h = 383;
    int dst_w = 256, dst_h = 256;

    // 读取图片
    char *img_path = args[1];
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        printf("Failed to load image.\n");
        return -1;
    }

    // BGR 转 RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 使用 OpenCV resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(dst_w, dst_h));

    // 保存成 JPG
    cv::cvtColor(resized, resized, cv::COLOR_RGB2BGR); // 可选，如果想存成标准 JPG
    cv::imwrite("resized.jpg", resized);

    printf("Saved resized image to resized.jpg\n");
    return 0;
}
