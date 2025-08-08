#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    string img_path = "1-FemaleNoGlasses_334.jpg";
    Mat img = imread(img_path, IMREAD_COLOR);

    if (img.empty()) {
        return -1;
    }

    // Draw text on the image
    putText(img, "hello", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);

    // Save the result
    imwrite("output_with_text.jpg", img);

    return 0;
}
