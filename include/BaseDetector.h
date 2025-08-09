//
// Created by yuwenlu on 2022/3/14.
//
#ifndef BASE_DETECT_H
#define BASE_DETECT_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
using namespace std;

class BaseDetection
{
public:
    BaseDetection():mRGB() {}
    virtual ~BaseDetection() = default;
    //禁用拷贝
    BaseDetection(const BaseDetection&) = delete;
    BaseDetection& operator=(const BaseDetection&) = delete;

    virtual void GetImage(cv::Mat& RGB) final {
        mRGB = RGB;
    }
    virtual void GetImageDep(cv::Mat& Depth) final {
        mDepth = Depth;
    }
    virtual void ClearImage() final {
        mRGB = 0;
    }
    virtual bool Detect() = 0;
    virtual void ClearArea() final {
        mvPersonArea.clear();
    }
    vector<cv::Rect2i> mvPersonArea = {};
    virtual vector<at::Tensor> PostProcess(const at::Tensor& preds) = 0;
    virtual void maskUpdate(cv::Rect2i& DynamicArea) = 0;

public:
    cv::Mat mRGB;
    cv::Mat mDepth;
    torch::jit::script::Module mModule;
    std::vector<std::string> mClassnames;

    // 6-28
    vector<string> mvDynamicNames;
    vector<cv::Rect2i> mvDynamicArea;
    map<string, vector<cv::Rect2i>> mmDetectMap;
    cv::Mat mMask;
    bool needUpdateMask;
    bool allDelete;


};


#endif //BASE_DETECT_H
