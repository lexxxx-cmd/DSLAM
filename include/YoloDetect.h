//
// Created by yuwenlu on 2022/3/14.
//
#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include "BaseDetector.h"
using namespace std;

class YoloDetection : public BaseDetection
{
public:
    YoloDetection();
    bool Detect() override;
    vector<at::Tensor> PostProcess(const at::Tensor& preds) override;
    void maskUpdate(cv::Rect2i& DynamicArea) override {
        mMask = cv::Mat::zeros(mRGB.size(), CV_8UC1);
    }
    /*
    ~YoloDetection();
    void GetImage(cv::Mat& RGB);
    void ClearImage();
    
    void ClearArea();
    vector<cv::Rect2i> mvPersonArea = {};
    

public:
    cv::Mat mRGB;
    torch::jit::script::Module mModule;
    std::vector<std::string> mClassnames;

    // 6-28
    vector<string> mvDynamicNames;
    vector<cv::Rect2i> mvDynamicArea;
    map<string, vector<cv::Rect2i>> mmDetectMap;
    */

};


#endif //YOLO_DETECT_H
