// #include "RTDETR.h"
// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <cstring>  // std::memset

// int main(int argc, char** argv) {
//     if (argc != 3) {
//         std::cout << "Usage: test_rt_detr <model.engine> <image.jpg|image.png>\n";
//         return -1;
//     }

//     std::string model_path = argv[1];
//     RTDETRv4 detector(model_path, 640);
//     //detector.warmup(5);
//     std::string img_path = argv[2];
//     cv::Mat image = cv::imread(img_path);
//     auto results = detector.detect(image, 0.7);
    
//     for (const auto& det : results) {
//         cv::rectangle(image, det.bbox, cv::Scalar(255, 255, 0), 2);
//         std::string label = std::to_string(det.class_id) + ": " + 
//                            std::to_string(det.confidence);
//         cv::putText(image, label, det.bbox.tl(), 
//                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
//     }
    
//     cv::imwrite("result.jpg", image);
//     return 0;
// }
#include "rtdetr_opencv.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./app [engine_path] [image_path]" << std::endl;
        return -1;
    }

    std::string enginePath = argv[1];
    std::string imagePath = argv[2];

    // 初始化（会自动尝试加载 GPU 0）
    RTDetr detector(enginePath);

    // 读取图像
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << imagePath << std::endl;
        return -1;
    }

    // 推理
    auto results = detector.infer(img, 0.8f);

    // 绘制并保存
    RTDetr::draw(img, results);
    cv::imwrite("output.jpg", img);
    
    std::cout << "Detected " << results.size() << " objects. Saved to output.jpg" << std::endl;
    return 0;
}