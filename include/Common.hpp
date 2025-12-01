#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/video/tracking.hpp> // 包含光流函数
#include <vector>
#include <Eigen/Dense>
#include <cmath>

/**
 * @brief 包含光流计算相关静态工具函数的类
 */
// class OpticalFlowUtils {
// public:
//     OpticalFlowUtils(const OpticalFlowUtils&) = delete;
//     void operator=(const OpticalFlowUtils&) = delete;

//     /**
//      * @brief 计算两张灰度图之间的稠密光流，并根据运动幅度生成一个掩码。
//      * 
//      * @param prevFrame 前一帧的灰度图 (CV_8UC1)。
//      * @param nextFrame 后一帧的灰度图 (CV_8UC1)。
//      * @param magnitudeThreshold 幅度阈值。光流向量的长度超过此值时，才被认为是显著运动。
//      */
//     static void calculateMotionMask(const cv::Mat& prevFrame, const cv::Mat& nextFrame, double magnitudeThreshold = 4.0, int num_bins = 256, int offset = 10) {
//         auto& mask = getMask();
//         mask = cv::Mat::zeros(prevFrame.size(), CV_8UC1);
//         mask.setTo(255);
//         // --- 1. 输入验证 ---
//         // 确保输入图像非空
//         if (prevFrame.empty() || nextFrame.empty()) {
//             std::cerr << "Error: Input frames cannot be empty." << std::endl;

//             return;
//         }
//         // 确保图像尺寸相同
//         if (prevFrame.size() != nextFrame.size()) {
//             std::cerr << "Error: Input frames must have the same size." << std::endl;

//             return;
//         }
//         // 确保图像为单通道8位灰度图
//         if (prevFrame.type() != CV_8UC1 || nextFrame.type() != CV_8UC1) {
//             std::cerr << "Error: Input frames must be of type CV_8UC1 (grayscale)." << std::endl;

//             return;
//         }
//         // 确保掩码与图像尺寸相同且类型正确
//         if (mask.size() != prevFrame.size() || mask.type() != CV_8UC1) {
//             std::cerr << "Error: Mask must have the same size as frames and be of type CV_8UC1." << std::endl;
//             // 可以选择在这里自动创建/重置mask
//             return;
//         }

//         // --- 2. 计算稠密光流 ---
//         // 创建一个Mat来存储光流结果 (x和y方向的位移)
//         // 光流结果是 CV_32FC2 类型，每个像素点包含两个float值
//         cv::Mat flow;

//         // 调用Farneback算法计算稠密光流
//         // 参数解释:
//         // prev, next: 输入图像
//         // flow: 输出的光流场
//         // 0.5: pyr_scale, 图像金字塔缩放比例
//         // 3: levels, 金字塔层数
//         // 15: winsize, 平均窗口大小，越大对噪声越不敏感，但会模糊运动边缘
//         // 3: iterations, 在每层金字塔上的迭代次数
//         // 5: poly_n, 像素邻域大小，用于多项式展开
//         // 1.2: poly_sigma, 高斯标准差，用于平滑导数
//         // 0: flags
//         cv::calcOpticalFlowFarneback(prevFrame, nextFrame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//         int motionPixels = 0;
//         int totalPixels = flow.rows * flow.cols;
//         // 1. 计算光流幅度并归一化到8位
//         std::vector<cv::Mat> flow_channels(2);
//         cv::split(flow, flow_channels); // 将2通道的flow分离为x和y两个单通道Mat

//         cv::Mat magnitude;
//         // 计算光流的幅度 sqrt(flow_x^2 + flow_y^2)
//         cv::magnitude(flow_channels[0], flow_channels[1], magnitude);

//         cv::Mat mag_norm;
//         // 将幅度归一化到 0-255 范围
//         cv::normalize(magnitude, mag_norm, 0, 255, cv::NORM_MINMAX);

//         cv::Mat mag_uint8;
//         // 转换为8位无符号整数类型 (CV_8U)
//         mag_norm.convertTo(mag_uint8, CV_8U);

//         // 2. 计算直方图
//         cv::Mat hist;
//         int histSize[] = { num_bins }; // 直方图的bin数量
//         float range[] = { 0, 256 };    // 像素值范围
//         const float* histRange[] = { range };
//         int channels[] = { 0 };

//         cv::calcHist(&mag_uint8, 1, channels, cv::Mat(), hist, 1, histSize, histRange, true, false);

//         // 3. 找到峰值位置
//         // 为避免极端情况，忽略直方图最开始和最后的部分
//         // Python中 hist[5:250] 对应C++中 rowRange(5, 250)
//         cv::Mat search_range = hist.rowRange(5, 250);
        
//         double minVal, maxVal;
//         cv::Point minLoc, maxLoc;
//         // 在指定范围内查找最大值及其位置
//         cv::minMaxLoc(search_range, &minVal, &maxVal, &minLoc, &maxLoc);

//         // peak_location 是最大值在 search_range 中的索引(maxLoc.y)
//         // 加上起始偏移量5，得到在完整直方图中的索引
//         // 注意：原始Python代码中为 `+ 15`，这可能是个笔误，逻辑上应为 `+ 5`。
//         // 这里我们遵循原始代码的逻辑，但已添加注释。如果需要，可以改为 `+ 5`。
//         int peak_location = maxLoc.y + 5; 

//         // 4. 计算自适应阈值
//         double adaptive_threshold = peak_location + offset;
//         // 阈值不能超过255
//         adaptive_threshold = std::min(adaptive_threshold, 255.0);
        
//         //std::cout << "直方图峰值位于: " << peak_location << ", 自适应阈值: " << adaptive_threshold << std::endl;

//         // 5. 应用阈值分割
//         cv::Mat ratio;
//         cv::threshold(mag_uint8, ratio, 50, 255, cv::THRESH_BINARY);
//         if(static_cast<float>(countNonZero(ratio)/ratio.total())>0.7){
//             mask.setTo(255); 
//         }else{
       
//             cv::Mat motion_mask;
//             cv::threshold(mag_uint8, motion_mask, adaptive_threshold, 255, cv::THRESH_BINARY);
//             cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//             cv::Mat kernel_large = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

//             cv::Mat temp_mask;
//             cv::Mat result_mask;

//             // 1. 开运算：去除外部噪点 (Erosion followed by Dilation)
//             // 使用较小的内核去除小的椒盐噪声
//             cv::morphologyEx(motion_mask, temp_mask, cv::MORPH_OPEN, kernel_small);

//             // 2. 闭运算：填充内部空洞 (Dilation followed by Erosion)
//             // 使用较大的内核来连接邻近的白色区域并填充较大的孔洞
//             cv::morphologyEx(temp_mask, result_mask, cv::MORPH_CLOSE, kernel_large);

//             mask = result_mask.clone();
                 
//         }
//     }
//     /*
//     static bool& isOFReliable() {
//         static bool isReliable = false;
//         return isReliable;
//     }*/
//     static cv::Mat& getMask() {
//         static cv::Mat mask = cv::Mat(480, 640, CV_8UC1, cv::Scalar(255));
//         return mask;
//     }
// private:
//     OpticalFlowUtils() = default;

// };

class OpticalFlowUtils {
public:
    OpticalFlowUtils(const OpticalFlowUtils&) = delete;
    void operator=(const OpticalFlowUtils&) = delete;

    static void calculateMotionMask(const cv::Mat& prevFrame, const cv::Mat& nextFrame, double magnitudeThreshold = 4.0, int num_bins = 256, int offset = 10) {
        auto& mask = getMask();
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
        // mask 重置逻辑移到最后或根据需要重置，为了性能，这里我们先不管 CPU mask 的初始化，直接覆盖

        // --- 1. 输入验证 ---
        if (prevFrame.empty() || nextFrame.empty()) {
            std::cerr << "Error: Input frames cannot be empty." << std::endl;
            return;
        }
        if (prevFrame.size() != nextFrame.size()) {
            std::cerr << "Error: Input frames must have the same size." << std::endl;
            return;
        }
        if (prevFrame.type() != CV_8UC1 || nextFrame.type() != CV_8UC1) {
            std::cerr << "Error: Input frames must be of type CV_8UC1 (grayscale)." << std::endl;
            return;
        }

        // --- GPU 资源初始化 (Static 保证只创建一次，避免重复开销) ---
        // 对应原参数: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        static auto farneback = cv::cuda::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3, 5, 1.2, 0);
        
        // 形态学滤波器
        static auto kernel_small = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        static auto kernel_large = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        static auto morph_open = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernel_small);
        static auto morph_close = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, kernel_large);

        // GPU 内存 (GpuMat). 声明为 static 以复用显存，避免频繁 malloc
        static cv::cuda::GpuMat d_prev, d_next, d_flow;
        static cv::cuda::GpuMat d_flow_xy[2];
        static cv::cuda::GpuMat d_magnitude, d_mag_norm, d_mag_uint8;
        static cv::cuda::GpuMat d_hist;
        static cv::cuda::GpuMat d_ratio, d_motion_mask, d_temp_mask, d_final_mask;

        // --- 2. H2D 数据上传 ---
        d_prev.upload(prevFrame);
        d_next.upload(nextFrame);

        // --- 3. 计算稠密光流 (GPU) ---
        farneback->calc(d_prev, d_next, d_flow);

        // --- 4. 幅度计算与归一化 (GPU) ---
        cv::cuda::split(d_flow, d_flow_xy);
        cv::cuda::magnitude(d_flow_xy[0], d_flow_xy[1], d_magnitude);
        
        // Normalize: alpha=0, beta=255, norm_type=NORM_MINMAX, dtype=CV_32F (计算时保持精度)
        cv::cuda::normalize(d_magnitude, d_mag_norm, 0, 255, cv::NORM_MINMAX, CV_32F);
        d_mag_norm.convertTo(d_mag_uint8, CV_8U);

        // --- 5. 计算直方图 (GPU -> CPU) ---
        // cv::cuda::calcHist 输出是一个行向量 (1 x 256)
        cv::cuda::calcHist(d_mag_uint8, d_hist);
        
        cv::Mat hist;
        d_hist.download(hist); // 下载回 CPU 进行分析 (只有256个数据，极快)

        // --- 6. 峰值查找 (CPU, 保持原有逻辑不变) ---
        // 注意：CUDA calcHist 结果通常是 CV_32S (int)
        if (hist.rows == 1) hist = hist.reshape(1, 256); // 确保是列向量以便 rowRange 正常工作

        cv::Mat search_range = hist.rowRange(5, 250);
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(search_range, &minVal, &maxVal, &minLoc, &maxLoc);

        int peak_location = maxLoc.y + 5; 
        double adaptive_threshold = std::min((double)(peak_location + offset), 255.0);

        // --- 7. 阈值判断与形态学 (GPU) ---
        cv::cuda::threshold(d_mag_uint8, d_ratio, 50, 255, cv::THRESH_BINARY);
        
        // countNonZero 在 GPU 上执行
        int nonZero = cv::cuda::countNonZero(d_ratio);
        int totalPixels = prevFrame.total();

        if (static_cast<float>(nonZero) / totalPixels > 0.7) {
            d_final_mask.create(d_prev.size(), CV_8UC1);
            d_final_mask.setTo(cv::Scalar(255));
        } else {
            // 应用自适应阈值
            cv::cuda::threshold(d_mag_uint8, d_motion_mask, adaptive_threshold, 255, cv::THRESH_BINARY);
            
            // 形态学操作
            morph_open->apply(d_motion_mask, d_temp_mask);
            morph_close->apply(d_temp_mask, d_final_mask);
        }

        // --- 8. D2H 结果下载 ---
        d_final_mask.download(mask);
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();
        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        std::cout << "timeOPtflow = " << timePreImu << std::endl;
    }

    static cv::Mat& getMask() {
        // 注意：虽然 calculateMotionMask 内部会覆盖 mask，但保持这个静态 buffer 也可以
        static cv::Mat mask = cv::Mat(480, 640, CV_8UC1, cv::Scalar(255));
        return mask;
    }

private:
    OpticalFlowUtils() = default;
};


/**
 * @class StaticProbabilityGrid
 * @brief 一个静态类，用于全局管理和更新一个静态概率网格。
 *
 * 此类被设计为“静态类”，意味着它的所有成员和函数都是静态的。
 * 您不需要也无法创建它的实例。所有操作都通过类名直接调用，
 * 例如：StaticProbabilityGrid::update(...);
 *
 * 这种设计确保了在整个应用程序中只有一个共享的概率网格实例，
 * 非常适合在ORB-SLAM3中作为全局唯一的特征区域静态性指导。
 */
class StaticProbabilityGrid {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    // 删除拷贝构造和赋值操作，确保该类不会被复制。
    StaticProbabilityGrid(const StaticProbabilityGrid&) = delete;
    void operator=(const StaticProbabilityGrid&) = delete;

    // --- 公共接口 ---

    // 定义网格的维度
    static constexpr int GRID_ROWS = 12;
    static constexpr int GRID_COLS = 16;

    // 根据您文档中提供的置信度p计算对数几率更新值
    // L = log(p / (1 - p))
    static constexpr float LOG_ODDS_HIT = 0.84729786f;  // log(0.7 / 0.3)
    static constexpr float LOG_ODDS_MISS = -0.40546511f; // log(0.4 / 0.6)

    // 为防止数值溢出或漂移，设定对数几率的上下限
    static constexpr float LOG_ODDS_MIN = -5.0f; // 对应概率约 0.0067
    static constexpr float LOG_ODDS_MAX = 5.0f;  // 对应概率约 0.9933

    /**
     * @brief 重置整个网格，将所有单元的概率设置为0.5。
     *
     * 在对数几率空间中，这等价于将所有值设置为0。
     * 由于是静态类，此操作将影响全局唯一的网格。
     */
    static void reset() {
        getLogOddsGrid_Internal().setZero();
    }

    /**
     * @brief 根据当前的观测结果更新全局的静态概率网格。
     * @param observation_matrix 一个与网格同维度的Eigen::Matrix。
     *                           矩阵中大于0的值表示在该单元格中成功提取了静态特征点（命中），
     *                           等于0或负值表示未提取到（未命中）。
     */
    static void update(const Eigen::Matrix<float, GRID_ROWS, GRID_COLS>& observation_matrix) {
        // 获取对内部静态网格的引用
        auto& grid = getLogOddsGrid_Internal();
        auto tmp = Eigen::Matrix<float, GRID_ROWS, GRID_COLS>::Ones(12, 16);
        // 将更新值加到当前的对数几率网格上
        grid += LOG_ODDS_HIT * observation_matrix;
        grid += LOG_ODDS_MISS * (tmp-observation_matrix);

        // 对更新后的值进行钳位，防止无限增长或减小
        grid = grid.cwiseMax(LOG_ODDS_MIN).cwiseMin(LOG_ODDS_MAX);
    }

    /**
     * @brief 获取当前表示为[0, 1]区间的概率矩阵。
     * @return 一个Eigen::Matrix，其元素值为每个单元格是静态的概率。
     *
     * 该方法将内部存储的对数几率矩阵转换为标准概率。
     * 转换公式为: p = 1 - 1 / (1 + exp(L))
     */
    static Eigen::Matrix<float, GRID_ROWS, GRID_COLS> getProbabilityGrid() {
        // 获取对内部静态网格的常量引用
        const auto& grid = getLogOddsGrid_Internal();

        // 使用Eigen的向量化操作高效地完成转换
        return 1.0f - (1.0f + grid.array().exp()).inverse();
    }

    static void saveVisualization(const std::string& filename) {
        // 1. 获取概率矩阵
        Eigen::Matrix<float, GRID_ROWS, GRID_COLS> prob_grid = getProbabilityGrid();

        // 2. 定义输出图像尺寸并创建图像
        const int img_height = 480;
        const int img_width = 640;
        cv::Mat visualization_image(img_height, img_width, CV_8UC1);

        // 3. 遍历概率网格，填充图像
        for (int r = 0; r < GRID_ROWS; ++r) {
            for (int c = 0; c < GRID_COLS; ++c) {
                // 根据概率值确定颜色 (0=黑, 255=白)
                uchar color = (prob_grid(r, c) < 0.5f) ? 0 : 255;

                // 计算此单元格在输出图像中对应的矩形区域
                // 这种计算方式可以精确处理640/14不是整数的情况，确保填满整个图像
                int y_start = r * img_height / GRID_ROWS;
                int y_end = (r + 1) * img_height / GRID_ROWS;
                int x_start = c * img_width / GRID_COLS;
                int x_end = (c + 1) * img_width / GRID_COLS;

                // 获取图像中的矩形区域(ROI)并用计算出的颜色填充
                cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);
                visualization_image(roi).setTo(color);
            }
        }

        // 4. 保存图像
        try {
            if (!cv::imwrite(filename, visualization_image)) {
                std::cerr << "Warning: Could not save visualization image to " << filename << std::endl;
            }
        } catch (const cv::Exception& ex) {
            std::cerr << "Error: OpenCV exception while saving image: " << ex.what() << std::endl;
        }
    }
private:
    /**
     * @brief 私有构造函数，防止外部创建此类的实例。
     */
    StaticProbabilityGrid() = default;

    /**
     * @brief 内部函数，用于获取唯一的、静态的网格实例。
     * @return 对内部log-odds网格的引用。
     *
     * 这里使用了“函数内静态变量”（Meyers' Singleton）的设计模式。
     * 这确保了log_odds_grid只被初始化一次，并且这种初始化在C++11及
     * 以后的版本中是线程安全的。这是实现头文件-only静态类的最佳实践。
     */
    static Eigen::Matrix<float, GRID_ROWS, GRID_COLS>& getLogOddsGrid_Internal() {
        // 静态变量在第一次调用此函数时被初始化为全零矩阵，之后调用都返回同一个实例的引用。
        static Eigen::Matrix<float, GRID_ROWS, GRID_COLS> log_odds_grid = 
            Eigen::Matrix<float, GRID_ROWS, GRID_COLS>::Zero();
        return log_odds_grid;
    }
};

#endif // COMMON_H