#include "RTdetr_opencv.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvOnnxParser.h>

// 简单的错误检查宏
#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort(); \
        } \
    } while (0)

void RTDetr::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[RTDetr] " << msg << std::endl;
    }
}

RTDetr::RTDetr(const std::string& modelPath, int gpuId) : mGpuId(gpuId) {
    cudaSetDevice(mGpuId);
    CHECK(cudaStreamCreate(&mStream));
    
    initEngine(modelPath);

    // 预分配 CPU 端输入 Buffer (1 * 3 * H * W)
    mInputHostBuffer.resize(3 * mInputH * mInputW);

    // 分配 Host 端输出 Buffer
    mHostScores = new float[mOutputCandidates];
    mHostBoxes  = new float[mOutputCandidates * 4];
    mHostLabels = new int64_t[mOutputCandidates];
}

RTDetr::~RTDetr() {
    cudaStreamDestroy(mStream);
    delete[] mHostScores;
    delete[] mHostBoxes;
    delete[] mHostLabels;
    
    for (void* ptr : mDeviceBuffers) {
        if (ptr) cudaFree(ptr);
    }
    delete mContext;
    delete mEngine;
    delete mRuntime;
}

void RTDetr::initEngine(const std::string& modelPath) {
    // 提示：确保已加载 TensorRT 插件库
    // 实际项目中可能需要：dlopen("libfused_attn_offset_prediction_plugin_v3.so", RTLD_NOW);
    
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: File " << modelPath << " not found!" << std::endl;
        abort();
    }
    
    // 读取文件内容
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    mRuntime = nvinfer1::createInferRuntime(mLogger);
    
    // 假设是 .engine 文件，如果是 onnx 需要在这里集成构建逻辑（参考上一个回答）
    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), size);
    if (!mEngine) {
        std::cerr << "Error: Failed to deserialize engine. If this is an ONNX file, please build it first." << std::endl;
        abort();
    }
    
    mContext = mEngine->createExecutionContext();

    // 解析 IO Binding
    int nbIO = mEngine->getNbIOTensors();
    for (int i = 0; i < nbIO; ++i) {
        const char* name = mEngine->getIOTensorName(i);
        nvinfer1::Dims dims = mEngine->getTensorShape(name);
        
        // 简单计算 Tensor 大小
        int64_t totalSize = 1;
        for(int j=0; j<dims.nbDims; ++j) totalSize *= dims.d[j];

        // 绑定索引
        std::string sName = name;
        if (sName.find("images") != std::string::npos) { 
            mIdxImage = i; 
            mInputH = dims.d[2]; 
            mInputW = dims.d[3];
        }
        else if (sName.find("orig_target_sizes") != std::string::npos) mIdxOrigSize = i;
        else if (sName.find("scores") != std::string::npos) mIdxScores = i;
        else if (sName.find("boxes") != std::string::npos) mIdxBoxes = i;
        else if (sName.find("labels") != std::string::npos) mIdxLabels = i;

        // 分配 GPU 显存
        size_t typeSize = (mEngine->getTensorDataType(name) == nvinfer1::DataType::kINT64) ? 8 : 4;
        CHECK(cudaMalloc(&mDeviceBuffers[i], totalSize * typeSize));
        mContext->setTensorAddress(name, mDeviceBuffers[i]);
    }
}

// -------------------------------------------------------------------------
// 核心修改：使用 OpenCV 实现 Letterbox 预处理
// -------------------------------------------------------------------------
void RTDetr::preprocessCpu(const cv::Mat& img) {
    // 1. 计算缩放比例 (Letterbox)
    int w = img.cols;
    int h = img.rows;
    float scale = std::min((float)mInputW / w, (float)mInputH / h);
    int newW = w * scale;
    int newH = h * scale;

    // 2. Resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH));

    // 3. 创建画布并填充灰色 (128)
    cv::Mat canvas(mInputH, mInputW, CV_8UC3, cv::Scalar(128, 128, 128));
    
    // 4. 将 Resize 后的图拷贝到画布中心
    int dx = (mInputW - newW) / 2;
    int dy = (mInputH - newH) / 2;
    resized.copyTo(canvas(cv::Rect(dx, dy, newW, newH)));

    // 5. 转换格式: BGR -> RGB, uint8 -> float32, 归一化 / 255.0
    cv::Mat floatMat;
    canvas.convertTo(floatMat, CV_32FC3, 1.0f / 255.0f);
    cv::cvtColor(floatMat, floatMat, cv::COLOR_BGR2RGB);

    // 6. HWC -> CHW 转换
    // OpenCV 默认是 HWC (interleaved)，TensorRT 需要 CHW (planar)
    std::vector<cv::Mat> channels(3);
    cv::split(floatMat, channels);

    // 将三个通道的数据依次拷贝到连续的 Host Buffer 中
    // mInputHostBuffer 布局: [RRRR...GGGG...BBBB...]
    int planeSize = mInputH * mInputW;
    memcpy(mInputHostBuffer.data(), channels[0].data, planeSize * sizeof(float));
    memcpy(mInputHostBuffer.data() + planeSize, channels[1].data, planeSize * sizeof(float));
    memcpy(mInputHostBuffer.data() + planeSize * 2, channels[2].data, planeSize * sizeof(float));

    // 7. 将准备好的数据拷贝到 GPU
    CHECK(cudaMemcpyAsync(mDeviceBuffers[mIdxImage], mInputHostBuffer.data(), 
                          mInputHostBuffer.size() * sizeof(float), 
                          cudaMemcpyHostToDevice, mStream));

    // 8. 处理 orig_target_sizes (如有需要)
    if (mIdxOrigSize >= 0) {
        int64_t sizeData[2] = { (int64_t)mInputW, (int64_t)mInputH }; 
        CHECK(cudaMemcpyAsync(mDeviceBuffers[mIdxOrigSize], sizeData, 
                              sizeof(int64_t) * 2, cudaMemcpyHostToDevice, mStream));
    }
}

std::vector<Detection> RTDetr::infer(const cv::Mat& img, float confThresh) {
    mvDynamicArea.clear();
    // 1. CPU 预处理 + GPU 上传
    preprocessCpu(img);

    // 2. 执行推理
    mContext->enqueueV3(mStream);

    // 3. 拷贝结果回 CPU
    CHECK(cudaMemcpyAsync(mHostScores, mDeviceBuffers[mIdxScores], mOutputCandidates * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    CHECK(cudaMemcpyAsync(mHostBoxes, mDeviceBuffers[mIdxBoxes], mOutputCandidates * 4 * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    CHECK(cudaMemcpyAsync(mHostLabels, mDeviceBuffers[mIdxLabels], mOutputCandidates * sizeof(int64_t), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);

    // 4. 后处理：坐标还原
    std::vector<Detection> results;
    float scale = std::min((float)mInputW / img.cols, (float)mInputH / img.rows);
    int newW = img.cols * scale;
    int newH = img.rows * scale;
    int dx = (mInputW - newW) / 2;
    int dy = (mInputH - newH) / 2;

    for (int i = 0; i < mOutputCandidates; ++i) {
        float score = mHostScores[i];
        if (score < confThresh) continue;

        Detection det;
        det.classId = (int)mHostLabels[i];
        if (det.classId != 0) continue;
        det.conf = score;

        // 假设模型输出的是基于 640x640 的绝对坐标
        float x = mHostBoxes[i * 4 + 0];
        float y = mHostBoxes[i * 4 + 1];
        float w  = mHostBoxes[i * 4 + 2];
        float h  = mHostBoxes[i * 4 + 3];

        // 减去 padding 偏移，再除以缩放系数
        det.bbox[0] = (x - dx) / scale; // x1
        det.bbox[1] = (y - dy) / scale; // y1
        det.bbox[2] = (w - dx) / scale; // x2
        det.bbox[3] = (h - dy) / scale; // y2

        results.push_back(det);
        mvDynamicArea.push_back(cv::Rect2i((int)det.bbox[0], (int)det.bbox[1], (int)(det.bbox[2] - det.bbox[0]), (int)(det.bbox[3] - det.bbox[1])));
    }
    return results;
}

bool RTDetr::draw(const cv::Mat& img, cv::Mat &mMask, const cv::Mat &mDepth, const std::vector<Detection>& objects) {
    int cnt = 0;
    bool fill = false;
    for (const auto& obj : objects) {
        // cv::rectangle(img, cv::Point(obj.bbox[0], obj.bbox[1]), cv::Point(obj.bbox[2], obj.bbox[3]), cv::Scalar(255, 255, 0), 2);
        // std::string label = std::to_string(obj.classId) + ": " + std::to_string(obj.conf).substr(0, 4);
        // cv::putText(img, label, cv::Point(obj.bbox[0], obj.bbox[1] - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        // 确保输入框在图像范围内
        cv::Rect2i DynamicArea(obj.bbox[0], obj.bbox[1],obj.bbox[2]-obj.bbox[0], obj.bbox[3]-obj.bbox[1]);
        cv::Rect2i box = DynamicArea & cv::Rect(0, 0, mDepth.cols, mDepth.rows);
        if (box.area() == 0) {
            // 如果框无效，则标记为删除
            fill = true;
        }

        // 1. 正确提取ROI并处理数据类型
        // mDepth 是 CV_16U (unsigned short), mMask 是 CV_8U (uchar)
        // .at<T> 中的 T 必须与Mat的类型严格匹配！
        cv::Mat imBoxTight = mDepth(box);

        std::vector<ushort> validPixelsInBox;
        validPixelsInBox.reserve(imBoxTight.rows * imBoxTight.cols);
        for (int y = 0; y < imBoxTight.rows; y++) {
            // 使用指针访问，比 .at() 更快
            const ushort* row_ptr = imBoxTight.ptr<ushort>(y);
            for (int x = 0; x < imBoxTight.cols; x++) {
                const ushort val = row_ptr[x];
                if (val > 0) { // 假设0是无效深度值
                    validPixelsInBox.push_back(val);
                }
            }
        }

        // 如果框内没有有效深度点，则无法进行分割
        if (validPixelsInBox.empty()) {
            
            fill =  true;
        }

        // 2. 稳健地计算中位数和阈值
        // 使用 std::nth_element 比完全排序更高效
        auto median_it = validPixelsInBox.begin() + validPixelsInBox.size() / 2;
        std::nth_element(validPixelsInBox.begin(), median_it, validPixelsInBox.end());
        const float median = static_cast<float>(*median_it);

        std::vector<float> differences;
        differences.reserve(validPixelsInBox.size());
        for (const ushort& val : validPixelsInBox) {
            differences.push_back(std::abs(static_cast<float>(val) - median));
        }

        // 同样，使用 nth_element 找到68%分位数的差值作为阈值
        auto threshold_it = differences.begin() + static_cast<size_t>(differences.size() * 0.68);
        std::nth_element(differences.begin(), threshold_it, differences.end());
        const float threshold = *threshold_it;

        // 3. 修正循环边界和mask赋值
        const float aspect = static_cast<float>(box.height) / box.width;
        fill = (aspect < 1.0/2.8) || (aspect > 2.8);

        // 扩展区域，并进行边界检查
        const int expansion = 5;
        const int x1 = std::max(0, box.x - expansion);
        const int y1 = std::max(0, box.y - expansion);
        const int x2 = std::min(mDepth.cols, box.x + box.width + expansion);
        const int y2 = std::min(mDepth.rows, box.y + box.height + expansion);

        
        for (int y = y1; y < y2; ++y) {
            // 获取深度图和掩码图的行指针
            const ushort* depth_row_ptr = mDepth.ptr<ushort>(y);
            uchar* mask_row_ptr = mMask.ptr<uchar>(y);

            // 关键修正：循环边界是 x2，不是 y2！
            for (int x = x1; x < x2; ++x) {
                if (fill) {
                    // 关键修正：直接用 x 索引，不再 *2
                    mask_row_ptr[x] = 1;
                    cnt++;
                } else {
                    const ushort depth_val = depth_row_ptr[x];
                    // 确保深度有效，并且与中位数的差在阈值内
                    //if (depth_val > 0 && (std::abs(static_cast<float>(depth_val) - median) < threshold)) {
                    if (depth_val > 0 && (static_cast<float>(depth_val) < median+threshold)) {
                        // 关键修正：直接用 x 索引
                        mask_row_ptr[x] = 1;
                        cnt++;
                    }
                }
            }
        }
    }
    // 4. 更新状态
    // 判断填充的像素数是否足够
    if (static_cast<float>(cnt) / img.total() < 0.25) {
        return true;
    } else {
        return false;
    }
    return false;
}