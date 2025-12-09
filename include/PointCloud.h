#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_search.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/common/projection_matrix.h>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

#include "KeyFrame.h"
#include "Converter.h"

// 候选帧结构体保持不变
struct CandidateFrame {
    ORB_SLAM3::KeyFrame* mpKF;
    cv::Mat mColorImg;
    cv::Mat mDepthImg;
    double mBlurScore;

    CandidateFrame(ORB_SLAM3::KeyFrame* pKF, const cv::Mat& color, const cv::Mat& depth, double score)
        : mpKF(pKF), mColorImg(color.clone()), mDepthImg(depth.clone()), mBlurScore(score) {}
};

class PointCloudMapping {
public:
    using PointT = pcl::PointXYZRGB;
    using PointCloud = pcl::PointCloud<PointT>;

    /**
     * @brief 构造函数
     * @param resolution 地图分辨率
     * @param blurThreshold 模糊度阈值，低于此值的帧将被直接丢弃
     * @param candidatePoolSize 候选帧批处理大小
     */
    PointCloudMapping(double resolution, double blurThreshold, int candidatePoolSize);

    ~PointCloudMapping();

    /**
     * @brief 外部接口，由Tracking线程调用，用于插入新的关键帧数据
     */
    void insertKeyFrame(ORB_SLAM3::KeyFrame* kf, cv::Mat& color, cv::Mat& depth);

    /**
     * @brief 请求关闭点云构建线程。应在程序结束时由System类调用。
     */
    void shutdown();

    /**
     * @brief 在shutdown后调用，执行最终的交叉验证并组装静态地图。
     * @return 最终构建的全局静态点云地图
     */
    PointCloud::Ptr assembleFinalMap();

private:
    /**
     * @brief 点云累积的主线程函数
     */
    void run();

    /**
     * @brief 计算图像的拉普拉斯响应方差
     */
    double calculateBlurScore(const cv::Mat& image);

    /**
     * @brief 处理一个批次的候选帧，将其分配到主、次地图中
     */
    void processCandidateBatch();

    /**
     * @brief 根据关键帧生成局部点云并转换到世界坐标系
     */
    PointCloud::Ptr generatePointCloud(ORB_SLAM3::KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth);

    /**
     * @brief 交叉验证的核心函数，从输入点云中过滤出静态点
     */
    void filterStaticPoints(const PointCloud::Ptr& cloud_in, 
                            octomap::OcTree& octree_other, 
                            PointCloud::Ptr& cloud_out
                            );
    
    // --- 成员变量 ---

    // 配置参数
    double mResolution;
    double mBlurThreshold;
    int mCandidatePoolSize;

    // 线程与同步
    std::mutex mMutexShutdown;
    bool mbShutdown;
    std::thread mMappingThread;

    std::mutex mMutexCandidates;
    std::condition_variable mCondKeyFrameAvailable;
    std::vector<CandidateFrame> mCandidateFrames;

    // 地图数据 - 现在分为主次两组，并且会持续累积
    PointCloud::Ptr mPrimaryPointCloud;   // 累积的主点云
    PointCloud::Ptr mSecondaryPointCloud; // 累积的次点云
    std::unique_ptr<octomap::OcTree> mPrimaryOctree;
    std::unique_ptr<octomap::OcTree> mSecondaryOctree;

    pcl::VoxelGrid<PointT> mVoxelGridFilter;
};

#endif // POINTCLOUDMAPPING_H