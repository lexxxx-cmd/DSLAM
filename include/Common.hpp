#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp> // 包含光流函数
#include <vector>

/**
 * @brief 包含光流计算相关静态工具函数的类
 */
class OpticalFlowUtils {
public:
    /**
     * @brief 计算两张灰度图之间的稠密光流，并根据运动幅度生成一个掩码。
     * 
     * @param prevFrame 前一帧的灰度图 (CV_8UC1)。
     * @param nextFrame 后一帧的灰度图 (CV_8UC1)。
     * @param mask      输出掩码 (CV_8UC1)。函数将在此掩码上将运动幅度大于阈值的点设置为255。
     *                  该掩码在传入前应被正确初始化（例如，全黑）。
     * @param magnitudeThreshold 幅度阈值。光流向量的长度超过此值时，才被认为是显著运动。
     */
    static void calculateMotionMask(const cv::Mat& prevFrame, const cv::Mat& nextFrame, cv::Mat& mask, double magnitudeThreshold = 5.0) {
        // --- 1. 输入验证 ---
        // 确保输入图像非空
        if (prevFrame.empty() || nextFrame.empty()) {
            std::cerr << "Error: Input frames cannot be empty." << std::endl;
            return;
        }
        // 确保图像尺寸相同
        if (prevFrame.size() != nextFrame.size()) {
            std::cerr << "Error: Input frames must have the same size." << std::endl;
            return;
        }
        // 确保图像为单通道8位灰度图
        if (prevFrame.type() != CV_8UC1 || nextFrame.type() != CV_8UC1) {
            std::cerr << "Error: Input frames must be of type CV_8UC1 (grayscale)." << std::endl;
            return;
        }
        // 确保掩码与图像尺寸相同且类型正确
        if (mask.size() != prevFrame.size() || mask.type() != CV_8UC1) {
            std::cerr << "Error: Mask must have the same size as frames and be of type CV_8UC1." << std::endl;
            // 可以选择在这里自动创建/重置mask
            // mask = cv::Mat::zeros(prevFrame.size(), CV_8UC1);
            return;
        }

        // --- 2. 计算稠密光流 ---
        // 创建一个Mat来存储光流结果 (x和y方向的位移)
        // 光流结果是 CV_32FC2 类型，每个像素点包含两个float值
        cv::Mat flow;

        // 调用Farneback算法计算稠密光流
        // 参数解释:
        // prev, next: 输入图像
        // flow: 输出的光流场
        // 0.5: pyr_scale, 图像金字塔缩放比例
        // 3: levels, 金字塔层数
        // 15: winsize, 平均窗口大小，越大对噪声越不敏感，但会模糊运动边缘
        // 3: iterations, 在每层金字塔上的迭代次数
        // 5: poly_n, 像素邻域大小，用于多项式展开
        // 1.2: poly_sigma, 高斯标准差，用于平滑导数
        // 0: flags
        cv::calcOpticalFlowFarneback(prevFrame, nextFrame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // --- 3. 遍历光流场并更新Mask ---
        for (int y = 0; y < flow.rows; ++y) {
            for (int x = 0; x < flow.cols; ++x) {
                // 获取(x, y)处的光流向量
                const cv::Point2f& flow_at_xy = flow.at<cv::Point2f>(y, x);
                
                // 计算光流向量的幅度 (magnitude)
                float dx = flow_at_xy.x;
                float dy = flow_at_xy.y;
                double magnitude = std::sqrt(dx * dx + dy * dy);

                // 如果幅度大于阈值，则在mask上标记该点
                if (magnitude > magnitudeThreshold) {
                    mask.at<uchar>(y, x) = 255;
                }
            }
        }
    }
};
