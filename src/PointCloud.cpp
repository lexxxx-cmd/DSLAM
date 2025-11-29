#include "PointCloud.h"
#include <iostream>
#include <algorithm>
using PointT = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointT>;
PointCloudMapping::PointCloudMapping(double resolution, double blurThreshold, int candidatePoolSize)
    : mResolution(resolution), mBlurThreshold(blurThreshold), mCandidatePoolSize(candidatePoolSize), mbShutdown(false) {

    // 初始化主次点云和八叉树
    mPrimaryPointCloud = std::make_shared<PointCloud>();
    mSecondaryPointCloud = std::make_shared<PointCloud>();
    mPrimaryOctree = std::make_unique<pcl::octree::OctreePointCloudSearch<PointT>>(mResolution);
    const double min_log_odds = -4.595; // log(0.01 / 0.99)
    const double max_log_odds =  4.595; // log(0.99 / 0.01)
   // mPrimaryOctree->setClampingThresholds(min_log_odds, max_log_odds); 
    mSecondaryOctree = std::make_unique<pcl::octree::OctreePointCloudSearch<PointT>>(mResolution);
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
        PointCloud::Ptr localCloud = generatePointCloud(frame.mpKF, frame.mColorImg, frame.mDepthImg);
        *mPrimaryPointCloud += *localCloud; 
    }
    {
        const auto& frame = processingFrames[1];
        PointCloud::Ptr localCloud = generatePointCloud(frame.mpKF, frame.mColorImg, frame.mDepthImg);
        *mSecondaryPointCloud += *localCloud;
    }
    if (!mPrimaryPointCloud->empty()) {
        mPrimaryOctree->setInputCloud(mPrimaryPointCloud);
        mPrimaryOctree->addPointsFromInputCloud();
    }
    if (!mSecondaryPointCloud->empty()) {
        mSecondaryOctree->setInputCloud(mSecondaryPointCloud);
        mSecondaryOctree->addPointsFromInputCloud();
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
    filterStaticPoints(mPrimaryPointCloud, *mSecondaryOctree, staticFromPrimary, validationRadius);
    
    // 用主八叉树 M1 验证次点云 P2
    filterStaticPoints(mSecondaryPointCloud, *mPrimaryOctree, staticFromSecondary, validationRadius);

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
    PointCloud::Ptr cloud(new PointCloud());

    for (int m = 0; m < depth.rows; m++) {
        for (int n = 0; n < depth.cols; n++) {
            float d = depth.ptr<float>(m)[n];
            

            if (d < 0.1 || d > 10.0) {
                continue;
            }
            if(kf->mMask.at<uchar>(m, n) == 0) {
                PointT p;
                p.z = d;
                p.x = (n - kf->cx) * p.z / kf->fx;
                p.y = (m - kf->cy) * p.z / kf->fy;

                p.b = color.at<cv::Vec3b>(m, n)[0];
                p.g = color.at<cv::Vec3b>(m, n)[1];
                p.r = color.at<cv::Vec3b>(m, n)[2];
                cloud->push_back(p);
            }
        }
    }
    if (cloud->empty()) {
        // 返回一个空的 world cloud，保持接口一致性
        return PointCloud::Ptr(new PointCloud()); 
    }
    PointCloud::Ptr cloud_world(new PointCloud());
    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat(kf->GetPose());
    pcl::transformPointCloud(*cloud, *cloud_world, T.matrix());

    auto center = kf->GetCameraCenter();
    cloud_world->sensor_origin_ = Eigen::Vector4f(center(0,0), center(1,0), center(2,0), 1.0f);
    cloud_world->sensor_orientation_ = Eigen::Quaternionf(T.rotation().cast<float>());

    return cloud_world;
}

void PointCloudMapping::filterStaticPoints(const PointCloud::Ptr& cloud_in, 
                            pcl::octree::OctreePointCloudSearch<PointT>& octree_other, 
                            PointCloud::Ptr& cloud_out,
                            double searchRadius) {
    cloud_out->clear();

    cloud_out->points.reserve(cloud_in->size());

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    for (const auto& point : cloud_in->points) {
        // 检查点是否有效
        if (!pcl::isFinite(point)) {
            continue;
        }

        if (true || octree_other.radiusSearch(point, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance, 1) > 0) {
            // 如果在 searchRadius 范围内找到了至少一个点，则将该点视为静态点
            cloud_out->push_back(point);
        }
    }

    // 更新输出点云的元数据
    cloud_out->width = cloud_out->points.size();
    cloud_out->height = 1;
    cloud_out->is_dense = true;
}