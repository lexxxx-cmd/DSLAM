#include "PointCloud.h"
#include <iostream>
#include <algorithm>
using PointT = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointT>;

Eigen::Matrix<double,4,4> toMatrix4d(const cv::Mat &cvMat4)
{
    Eigen::Matrix<double,4,4> M;

    M << cvMat4.at<float>(0,0), cvMat4.at<float>(0,1), cvMat4.at<float>(0,2), cvMat4.at<float>(0,3),
         cvMat4.at<float>(1,0), cvMat4.at<float>(1,1), cvMat4.at<float>(1,2), cvMat4.at<float>(1,3),
         cvMat4.at<float>(2,0), cvMat4.at<float>(2,1), cvMat4.at<float>(2,2), cvMat4.at<float>(2,3),
         cvMat4.at<float>(3,0), cvMat4.at<float>(3,1), cvMat4.at<float>(3,2), cvMat4.at<float>(3,3);
    return M;
}

Eigen::Matrix<double,4,4> toMatrix4d(const Sophus::SE3f &T)
{
    Eigen::Matrix<double,4,4> M = Eigen::Matrix4d::Identity();

    M.block<3, 3>(0, 0) = T.rotationMatrix().cast<double>();
    M.block<3, 1>(0, 3) = T.translation().cast<double>();
    return M;
}
PointCloudMapping::PointCloudMapping(double resolution, double blurThreshold, int candidatePoolSize)
    : mResolution(resolution), mBlurThreshold(blurThreshold), mCandidatePoolSize(candidatePoolSize), mbShutdown(false) {

    // 初始化主次点云和八叉树
    mPrimaryPointCloud = std::make_shared<PointCloud>();
    mSecondaryPointCloud = std::make_shared<PointCloud>();
    mPrimaryOctree = std::make_unique<octomap::OcTree>(mResolution);
    const double min_log_odds = -4.595; // log(0.01 / 0.99)
    const double max_log_odds =  4.595; // log(0.99 / 0.01)
   // mPrimaryOctree->setClampingThresholds(min_log_odds, max_log_odds); 
    mSecondaryOctree = std::make_unique<octomap::OcTree>(mResolution);
   // mSecondaryOctree->setClampingThresholds(min_log_odds, max_log_odds);  
    mVoxelGridFilter.setLeafSize(mResolution, mResolution, mResolution);

    // 启动内部处理线程
    mMappingThread = std::thread(&PointCloudMapping::run, this);
}

PointCloudMapping::~PointCloudMapping() {
    if (!mbShutdown) {
        shutdown();
    }
}

void PointCloudMapping::shutdown() {
    {
        std::unique_lock<std::mutex> lock(mMutexShutdown);
        mbShutdown = true;
    }
    mCondKeyFrameAvailable.notify_one(); // 唤醒线程以使其能检查到shutdown标志
    if (mMappingThread.joinable()) {
        mMappingThread.join();
    }
}

void PointCloudMapping::insertKeyFrame(ORB_SLAM3::KeyFrame* kf, cv::Mat& color, cv::Mat& depth) {
    double score = calculateBlurScore(color);
    if (score < mBlurThreshold) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(mMutexCandidates);
        mCandidateFrames.emplace_back(kf, color, depth, score);
    }
    
    if (mCandidateFrames.size() >= mCandidatePoolSize) {
        mCondKeyFrameAvailable.notify_one();
    }
}

void PointCloudMapping::run() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mMutexCandidates);
            mCondKeyFrameAvailable.wait(lock, [this] {
                return (mCandidateFrames.size() >= mCandidatePoolSize) || mbShutdown;
            });
        }

        if (mbShutdown) {
            // 在退出前，处理掉池中剩余的帧
            if (!mCandidateFrames.empty()) {
                processCandidateBatch();
            }
            break;
        }

        processCandidateBatch();
    }
}

void PointCloudMapping::processCandidateBatch() {
    std::vector<CandidateFrame> processingFrames;
    {
        std::unique_lock<std::mutex> lock(mMutexCandidates);
        processingFrames.swap(mCandidateFrames);
    }

    if (processingFrames.empty()) return;

    std::cout << "Processing a batch of " << processingFrames.size() << " frames for dense mapping." << std::endl;

    // 根据模糊度分数降序排序
    std::sort(processingFrames.begin(), processingFrames.end(), 
        [](const CandidateFrame& a, const CandidateFrame& b) {
        return a.mBlurScore > b.mBlurScore;
    });
    {
        const auto& frame = processingFrames[0];
        Eigen::Matrix4f Twc = toMatrix4d(frame.mpKF->GetPoseInverse()).cast<float>();
        PointCloud::Ptr localCloud = generatePointCloud(frame.mpKF, frame.mColorImg, frame.mDepthImg);
        PointCloud::Ptr worldCloud(new PointCloud());
        if(localCloud->width != 0){
            pcl::transformPointCloud(*localCloud, *worldCloud, Twc);
            *mPrimaryPointCloud += *worldCloud; 
        }
        if (!worldCloud->empty()) {
            octomap::Pointcloud cloud;
            for (const auto& p : worldCloud->points) {
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;
            
                cloud.push_back(p.x, p.y, p.z);
            }
            octomap::point3d sensor_origin(Twc(0, 3), Twc(1, 3),Twc(2, 3));
            mPrimaryOctree->insertPointCloud(cloud, sensor_origin);
            mPrimaryOctree->updateInnerOccupancy();
        }
    }
    {
        const auto& frame = processingFrames[1];
        PointCloud::Ptr localCloud = generatePointCloud(frame.mpKF, frame.mColorImg, frame.mDepthImg);
        PointCloud::Ptr worldCloud(new PointCloud());
        Eigen::Matrix4f Twc = toMatrix4d(frame.mpKF->GetPoseInverse()).cast<float>();
        if(localCloud->width != 0){
            pcl::transformPointCloud(*localCloud, *worldCloud, Twc);
            *mSecondaryPointCloud += *worldCloud;
        }
        if (!worldCloud->empty()) {
            octomap::Pointcloud cloud;
            for (const auto& p : worldCloud->points) {
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;
            
                cloud.push_back(p.x, p.y, p.z);
            }
            octomap::point3d sensor_origin(Twc(0, 3), Twc(1, 3),Twc(2, 3));
            mSecondaryOctree->insertPointCloud(cloud, sensor_origin);
            mSecondaryOctree->updateInnerOccupancy();
        }
    }
    
    
}

PointCloudMapping::PointCloud::Ptr PointCloudMapping::assembleFinalMap() {
    std::cout << "Assembling final static map..." << std::endl;
    std::cout << "Primary map has " << mPrimaryPointCloud->size() << " points." << std::endl;
    std::cout << "Secondary map has " << mSecondaryPointCloud->size() << " points." << std::endl;

    // 1. 双向交叉验证
    PointCloud::Ptr staticFromPrimary = std::make_shared<PointCloud>();
    PointCloud::Ptr staticFromSecondary = std::make_shared<PointCloud>();

    const double validationRadius = mResolution * 1.5;

    // 用次八叉树 M2 验证主点云 P1
    filterStaticPoints(mPrimaryPointCloud, *mSecondaryOctree, staticFromPrimary);
    
    // 用主八叉树 M1 验证次点云 P2
    filterStaticPoints(mSecondaryPointCloud, *mPrimaryOctree, staticFromSecondary);

    std::cout << "Filtering results: Static from Primary: " << staticFromPrimary->size() << "/" << mPrimaryPointCloud->size() << std::endl;
    std::cout << "Filtering results: Static from Secondary: " << staticFromSecondary->size() << "/" << mSecondaryPointCloud->size() << std::endl;

    // 2. 合并过滤后的静态点云
    PointCloud::Ptr finalStaticMap = std::make_shared<PointCloud>();
    *finalStaticMap += *staticFromPrimary;
    *finalStaticMap += *staticFromSecondary;
    // *finalStaticMap += *mPrimaryPointCloud;
    // *finalStaticMap += *mSecondaryPointCloud;

    // 3. 对最终地图进行下采样以统一密度
    PointCloud::Ptr downsampledMap = std::make_shared<PointCloud>();
    mVoxelGridFilter.setInputCloud(finalStaticMap);
    mVoxelGridFilter.filter(*downsampledMap);

    std::cout << "Final static map assembled with " << downsampledMap->size() << " points." << std::endl;
    return downsampledMap;
}


// 以下函数与之前版本基本相同，但为保持完整性而保留
double PointCloudMapping::calculateBlurScore(const cv::Mat& image) {
    cv::Mat gray, laplacian;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return stddev.val[0] * stddev.val[0];
}

PointCloudMapping::PointCloud::Ptr PointCloudMapping::generatePointCloud(ORB_SLAM3::KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth) {
    PointCloud::Ptr cloud_local(new PointCloud());
    // 1. 步长设置：参考代码中使用了 +=4 进行降采样，这里设置为 2 或 4 可以平衡性能和密度
    const int step = 1; 
    float scale_factor = 1.0f;
    if (depth.type() == CV_16U) {
        scale_factor = 5000.0f;
    }

    for (int m = 0; m < depth.rows; m += step) {
        // 提前获取行指针，比 .at() 快很多
        const unsigned short* depth_ptr_16u = (depth.type() == CV_16U) ? depth.ptr<unsigned short>(m) : nullptr;
        const float* depth_ptr_32f = (depth.type() == CV_32F) ? depth.ptr<float>(m) : nullptr;
        const uchar* color_ptr = color.ptr<uchar>(m);

        for (int n = 0; n < depth.cols; n += step) {
            float d = 0.0f;
            if (depth_ptr_16u) {
                d = depth_ptr_16u[n] / scale_factor; // 毫米 -> 米
            } else if (depth_ptr_32f) {
                d = depth_ptr_32f[n];
            }

            // 2. 深度过滤：参考代码使用了更严格的 6.0m 上限，有助于减少远处的噪声层
            if (d < 0.1 || d > 6.0) {
                continue;
            }
            // 检查掩码（如果你的逻辑需要）
            if (!kf->mMask.empty() && kf->mMask.at<uchar>(m, n) == 0) {
                PointT p;
                p.z = d;
                p.x = (n - kf->cx) * p.z / kf->fx;
                p.y = (m - kf->cy) * p.z / kf->fy;

                // 3. 颜色赋值：参考代码的指针写法
                p.b = color_ptr[n * 3];
                p.g = color_ptr[n * 3 + 1];
                p.r = color_ptr[n * 3 + 2];
                
                cloud_local->push_back(p);
            }
            // 如果没有掩码逻辑，直接通过：
            else if (kf->mMask.empty()) {
                PointT p;
                p.z = d;
                p.x = (n - kf->cx) * p.z / kf->fx;
                p.y = (m - kf->cy) * p.z / kf->fy;
                p.b = color_ptr[n * 3];
                p.g = color_ptr[n * 3 + 1];
                p.r = color_ptr[n * 3 + 2];
                cloud_local->push_back(p);
            }
        }
    }

    cloud_local->height = 1;
    cloud_local->width = cloud_local->points.size();
    cloud_local->is_dense = true;

    return cloud_local; // 返回的是相机坐标系的局部点云
}

void PointCloudMapping::filterStaticPoints(const PointCloud::Ptr& cloud_in, 
                            octomap::OcTree& octree_other, 
                            PointCloud::Ptr& cloud_out
                            ) {
    cloud_out->clear();

    cloud_out->points.reserve(cloud_in->size());

    for (const auto& point : cloud_in->points) {
        // 检查点是否有效
        if (!pcl::isFinite(point)) {
            continue;
        }
        octomap::OcTreeNode* node = octree_other.search(point.x ,point.y, point.z);
        if (node && octree_other.isNodeOccupied(node)) {
            // 如果在 searchRadius 范围内找到了至少一个点，则将该点视为静态点
            cloud_out->push_back(point);
        }
    }

    // 更新输出点云的元数据
    cloud_out->width = cloud_out->points.size();
    cloud_out->height = 1;
    cloud_out->is_dense = true;
}