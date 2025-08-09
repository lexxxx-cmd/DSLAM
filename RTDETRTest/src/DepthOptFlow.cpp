#include "DepthOptFlow.h"
    std::vector<cv::Point2f> DynamicDetector::detect(const cv::Mat& curr_frame) {
        if(!mImGrayPre.data) {
            mImGrayPre = curr_frame.clone();
            return std::vector<cv::Point2f>();
        }

        // 清除上一帧数据
        F_prepoint.clear();
        F_nextpoint.clear();
        F2_prepoint.clear();
        F2_nextpoint.clear();
        T_M.clear();

        // 特征点检测
        cv::goodFeaturesToTrack(mImGrayPre, prepoint, 
            params.maxCorners, params.qualityLevel, params.minDistance,
            cv::Mat(), params.blockSize, params.useHarris, params.k);

        // 亚像素角点优化
        cv::cornerSubPix(mImGrayPre, prepoint, 
            cv::Size(10, 10), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.03));

        // 光流跟踪
        cv::calcOpticalFlowPyrLK(mImGrayPre, curr_frame, prepoint, nextpoint, 
            state, err, params.winSize, params.maxLevel,
            cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.01));

        // 特征点筛选
        filterPoints(curr_frame);

        // 基础矩阵估计
        cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
        cv::Mat F = cv::findFundamentalMat(F_prepoint, F_nextpoint, mask, 
            cv::FM_RANSAC, params.ransacReprojThreshold, params.confidence);

        // 提取高质量静态点
        findStaticPoints(F, mask);

        // 检测动态点
        findDynamicPoints(F);

        // 更新前一帧
        mImGrayPre = curr_frame.clone();

        return T_M;
    }

    // 可视化函数
    cv::Mat DynamicDetector::visualize(const cv::Mat& frame) {
        cv::Mat visual = frame.clone();
        if(visual.channels() == 1) {
            cv::cvtColor(visual, visual, cv::COLOR_GRAY2BGR);
        }

        // 绘制动态点和运动轨迹
        for(size_t i = 0; i < T_M.size(); i++) {
            // 在原始位置画点
            cv::circle(visual, prepoint[i], 3, cv::Scalar(0,0,255), -1);
            
            // 画箭头表示运动
            cv::arrowedLine(visual, prepoint[i], T_M[i], 
                cv::Scalar(0,0,255), 2, 8, 0, 0.2);
        }

        return visual;
    }
    void DynamicDetector::filterPoints(const cv::Mat& curr_frame) {
        for (size_t i = 0; i < state.size(); i++) {
            if(state[i] != 0) {
                // 边界检查
                int x1 = prepoint[i].x, y1 = prepoint[i].y;
                int x2 = nextpoint[i].x, y2 = nextpoint[i].y;
                
                if (isOutOfBounds(x1, y1, x2, y2, curr_frame)) {
                    state[i] = 0;
                    continue;
                }

                // 灰度一致性检查
                if (!checkGrayConsistency(x1, y1, x2, y2, curr_frame)) {
                    state[i] = 0;
                    continue;
                }

                if (state[i]) {
                    F_prepoint.push_back(prepoint[i]);
                    F_nextpoint.push_back(nextpoint[i]);
                }
            }
        }
    }

    bool DynamicDetector::isOutOfBounds(int x1, int y1, int x2, int y2, const cv::Mat& frame) {
        return (x1 < limit_edge_corner || x1 >= frame.cols - limit_edge_corner ||
                x2 < limit_edge_corner || x2 >= frame.cols - limit_edge_corner ||
                y1 < limit_edge_corner || y1 >= frame.rows - limit_edge_corner ||
                y2 < limit_edge_corner || y2 >= frame.rows - limit_edge_corner);
    }

    bool DynamicDetector::checkGrayConsistency(int x1, int y1, int x2, int y2, const cv::Mat& curr_frame) {
        int dx[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        int dy[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
        
        double sum_check = 0;
        for (int j = 0; j < 9; j++) {
            sum_check += abs(mImGrayPre.at<uchar>(y1 + dy[j], x1 + dx[j]) - 
                           curr_frame.at<uchar>(y2 + dy[j], x2 + dx[j]));
        }
        return sum_check <= limit_of_check;
    }

    void DynamicDetector::findStaticPoints(const cv::Mat& F, const cv::Mat& mask) {
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<uchar>(i, 0) != 0) {
                double dd = calculateEpipolarDistance(F, F_prepoint[i], F_nextpoint[i]);
                if (dd <= 0.1) {
                    F2_prepoint.push_back(F_prepoint[i]);
                    F2_nextpoint.push_back(F_nextpoint[i]);
                }
            }
        }
        F_prepoint = F2_prepoint;
        F_nextpoint = F2_nextpoint;
    }

    void DynamicDetector::findDynamicPoints(const cv::Mat& F) {
        for (size_t i = 0; i < prepoint.size(); i++) {
            if (state[i] != 0) {
                double dd = calculateEpipolarDistance(F, prepoint[i], nextpoint[i]);
                if (dd > limit_dis_epi) {
                    T_M.push_back(nextpoint[i]);
                }
            }
        }
    }

    double DynamicDetector::calculateEpipolarDistance(const cv::Mat& F, const cv::Point2f& p1, const cv::Point2f& p2) {
        double A = F.at<double>(0,0)*p1.x + F.at<double>(0,1)*p1.y + F.at<double>(0,2);
        double B = F.at<double>(1,0)*p1.x + F.at<double>(1,1)*p1.y + F.at<double>(1,2);
        double C = F.at<double>(2,0)*p1.x + F.at<double>(2,1)*p1.y + F.at<double>(2,2);
        return fabs(A*p2.x + B*p2.y + C) / sqrt(A*A + B*B);
    }
