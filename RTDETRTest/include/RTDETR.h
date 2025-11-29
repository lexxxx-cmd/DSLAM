#ifndef RTDETR_H
#define RTDETR_H

#include <string>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cassert>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <opencv2/opencv.hpp>

#define CHECK(call) \
    do { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class RTDETRv4 {
public:
    RTDETRv4(const std::string& engine_path, int input_size = 640);
    ~RTDETRv4();
    
    std::vector<Detection> detect(const cv::Mat& image, float conf_thresh = 0.4);
    void warmup(int iterations = 5);

private:
    struct TensorInfo {
        std::string name;
        nvinfer1::Dims dims;
        size_t size;
        void* device_ptr = nullptr;
        void* host_ptr = nullptr;
    };

    struct PreprocessParams {
        float scale;
        int pad_w;
        int pad_h;
        int orig_width;
        int orig_height;
    };

    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_;
    
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
    PreprocessParams preprocess_params_;
    int input_size_;
    
    void buildEngine(const std::string& path);
    void allocateBuffers();
    cv::Mat preprocess(const cv::Mat& image);
    void copyInputToDevice(const cv::Mat& preprocessed);
    void infer();
    std::vector<Detection> postprocess(float conf_thresh);
    
    static size_t getElementSize(nvinfer1::DataType dtype);
    static size_t getTensorSize(const nvinfer1::Dims& dims);
};


#endif // RTDETR_H

