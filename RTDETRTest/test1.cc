#include "YRTDETRDetector.h"
#include <opencv2/opencv.hpp>  // 添加这行
#include <opencv2/imgcodecs.hpp>
#include <torch/script.h>
#include <iostream>

int main(int argc, char **argv){

    if(argc != 4)
    {
        std::cerr << "\nUsage: ./test path_to_model path_to_classes path_to_img" << std::endl;
        return -1;
    }
    std::string model_path = argv[1];
    std::string class_path = argv[2];
    std::string img_path = argv[3];
    DETRDetector detector(model_path,class_path);

    cv::Mat image = cv::imread(img_path);
    if(image.empty()){
        std::cerr << "ERROR: Could not read the image!" << std::endl;
        return -1;
    }
    try {
        detector.GetImage(image);
        auto success = detector.Detect();
        std::cout << "Detect finish: " << success << std::endl;
        return 0;
    }catch (const c10::Error &e) {
        std::cerr << "ERROR: " << e.msg() << std::endl;
        return -1;
    }
    
}