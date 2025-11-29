#ifndef RTDETR_OPENCV_H
#define RTDETR_OPENCV_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

// 检测结果结构体
struct Detection {
    float bbox[4]; // x1, y1, x2, y2
    float conf;
    int classId;
};

class RTDetr {
public:
    /**
     * @brief 构造函数
     * @param modelPath .engine 或 .onnx 文件路径
     * @param gpuId GPU ID
     */
    RTDetr(const std::string& modelPath, int gpuId = 0);
    ~RTDetr();

    /**
     * @brief 推理函数
     * @param img 输入图像 (BGR, uint8)
     * @param confThresh 置信度阈值
     * @return 检测结果列表
     */
    std::vector<Detection> infer(const cv::Mat& img, float confThresh = 0.45f);

    // 辅助函数：绘制结果
    static void draw(cv::Mat& img, const std::vector<Detection>& objects);

private:
    void initEngine(const std::string& modelPath);
    // 使用 OpenCV 进行 CPU 预处理
    void preprocessCpu(const cv::Mat& img);

private:
    int mGpuId;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::IExecutionContext* mContext = nullptr;
    nvinfer1::IRuntime* mRuntime = nullptr;
    cudaStream_t mStream = nullptr;

    // 内存指针
    void* mDeviceBuffers[10] = {nullptr}; // 显存指针数组
    
    // CPU 侧的临时 buffer，用于预处理后的数据，之后拷贝到 GPU
    std::vector<float> mInputHostBuffer; 

    // 输出结果的 Host 内存
    float* mHostScores = nullptr;
    float* mHostBoxes = nullptr;
    int64_t* mHostLabels = nullptr;

    // 模型参数
    int mInputW = 640;
    int mInputH = 640;
    int mOutputCandidates = 300; 

    // Tensor索引
    int mIdxImage = -1;
    int mIdxOrigSize = -1;
    int mIdxScores = -1;
    int mIdxBoxes = -1;
    int mIdxLabels = -1;

    // Logger
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    } mLogger;
};

#endif // RTDETR_OPENCV_H