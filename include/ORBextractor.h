/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <queue>
#include <list>
#include <opencv2/opencv.hpp>


namespace ORB_SLAM3
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<cv::Mat> mvLastImagePyramid;
    std::queue<cv::Mat> mvMaskWindow,mvStaticMaskWindows;
    
    cv::Mat mMask;
    cv::Mat mLKMask;
    bool allDelete;
    cv::Mat F;
    std::vector<cv::Rect2i> mvDynamicArea;
    //光流的动态点集
    cv::Mat mDepth;
    std::vector<cv::Point2f> prepoint, nextpoint;
    std::vector<cv::Point2f> F_prepoint, F_nextpoint;
    std::vector<cv::Point2f> F2_prepoint, F2_nextpoint;
    std::vector<cv::Point2f> mT_M;
    std::unique_ptr<cv::flann::Index> flannIndex;
    std::vector<int> pointMask; 
    std::vector<uchar> forward_state;
    std::vector<float> err;
    int limit_edge_corner = 5;
    std::queue<cv::Mat> mLKMaskWin;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    void PointCulling(std::vector<cv::Point2f>& curr_pts,std::vector<cv::KeyPoint>& vToDistributeKeys,std::vector<cv::Point2f>& vLastToDistributeKeys,
        const int &level,const int &minBorderX,const int &minBorderY);
    bool FlannForDynaKeyPoints(const cv::KeyPoint& vit_kp,const int& num);
    std::vector<cv::Point2f> ConvertKeypointsToPoints(const std::vector<cv::KeyPoint>& keypoints);
    bool isDynaKeyPoints(
        std::vector<cv::Point2f>& vToDistributeKeys,std::vector<cv::Point2f>& vLastToDistributeKeys,
        const cv::KeyPoint& vit_kp,const int& i);
    //bool hasNearbyDynamicPoints(const cv::Point2f& pt);
    bool inDynamicMask(const cv::Point2f& pt);
    bool inDynamicLKMask(const cv::Point2f& pt);
    bool inSPMat(const cv::Point2f& pt);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;
    

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

#endif

