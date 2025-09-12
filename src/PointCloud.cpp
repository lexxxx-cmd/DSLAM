#include "PointCloud.h"
#include <iostream>
#include <algorithm>

PointCloudMapping::PointCloudMapping(double resolution, double blurThreshold, int candidatePoolSize)
    : mResolution(resolution), mBlurThreshold(blurThreshold), mCandidatePoolSize(candidatePoolSize), mbShutdown(false) {

    // 初始化主次点云和八叉树
    mPrimaryPointCloud = std::make_shared<PointCloud>();
    mSecondaryPointCloud = std::make_shared<PointCloud>();
    mPrimaryOctree = std::make_unique<pcl::octree::OctreePointCloudChangeDetector<PointT>>(mResolution);
    mSecondaryOctree = std::make_unique<pcl::octree::OctreePointCloudChangeDetector<PointT>>(mResolution);

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

    // 将批次分为两半
    size_t midPoint = processingFrames.size() / 2;

    // 前一半加入主地图
    for (size_t i = 0; i < midPoint; ++i) {
        const auto& frame = processingFrames[i];
        PointCloud::Ptr localCloud = generatePointCloud(frame.mpKF, frame.mColorImg, frame.mDepthImg);
        *mPrimaryPointCloud += *localCloud;
        mPrimaryOctree->setInputCloud(localCloud);
        mPrimaryOctree->addPointsFromInputCloud();
        mPrimaryOctree->switchBuffers(); // 固化本次添加，为下次比较做准备
    }

    // 后一半加入次地图
    for (size_t i = midPoint; i < processingFrames.size(); ++i) {
        const auto& frame = processingFrames[i];
        PointCloud::Ptr localCloud = generatePointCloud(frame.mpKF, frame.mColorImg, frame.mDepthImg);
        *mSecondaryPointCloud += *localCloud;
        mSecondaryOctree->setInputCloud(localCloud);
        mSecondaryOctree->addPointsFromInputCloud();
        mSecondaryOctree->switchBuffers();
    }
}

PointCloudMapping::PointCloud::Ptr PointCloudMapping::assembleFinalMap() {
    std::cout << "Assembling final static map..." << std::endl;
    std::cout << "Primary map has " << mPrimaryPointCloud->size() << " points." << std::endl;
    std::cout << "Secondary map has " << mSecondaryPointCloud->size() << " points." << std::endl;

    // 1. 双向交叉验证
    PointCloud::Ptr staticFromPrimary = std::make_shared<PointCloud>();
    PointCloud::Ptr staticFromSecondary = std::make_shared<PointCloud>();

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
    cloud->height = depth.rows;
    cloud->width = depth.cols;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);

    for (int m = 0; m < depth.rows; m++) {
        for (int n = 0; n < depth.cols; n++) {
            float d = depth.ptr<float>(m)[n];
            PointT& p = cloud->at(n, m);

            if (d < 0.1 || d > 10.0) {
                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            p.z = d;
            p.x = (n - kf->cx) * p.z / kf->fx;
            p.y = (m - kf->cy) * p.z / kf->fy;

            p.b = color.at<cv::Vec3b>(m, n)[0];
            p.g = color.at<cv::Vec3b>(m, n)[1];
            p.r = color.at<cv::Vec3b>(m, n)[2];
        }
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
                                           pcl::octree::OctreePointCloudChangeDetector<PointT>& octree_other, 
                                           PointCloud::Ptr& cloud_out) {
    std::vector<int> dynamic_indices;
    octree_other.setInputCloud(cloud_in);
    octree_other.getPointIndicesFromNewVoxels(dynamic_indices);

    pcl::PointIndices::Ptr dynamic_indices_ptr(new pcl::PointIndices());
    dynamic_indices_ptr->indices = dynamic_indices;

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud_in);
    extract.setIndices(dynamic_indices_ptr);
    extract.setNegative(true);
    extract.filter(*cloud_out);
}