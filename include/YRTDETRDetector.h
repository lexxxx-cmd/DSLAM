#ifndef YDETR_DETECTOR
#define YDETR_DETECTOR

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
#include <map>
#include <vector>
#include <string>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include "BaseDetector.h"

using namespace std;

class YDETRDetector : public BaseDetection
{
    public:
    //typedef std::shared_ptr<DETRDetector> Ptr;

    YDETRDetector();
    YDETRDetector(const std::string& model_path, const std::string& class_path);
    
    
    bool Detect() override; 
    std::vector<at::Tensor> PostProcess(const at::Tensor& preds) override;
    void maskUpdate(cv::Rect2i& DynamicArea) override;
    /*
    void GetImage(cv::Mat& RGB);
    void ClearArea();
    void ClearImage();

    std::vector<cv::Rect2i> mvPersonArea = {};
public:
    cv::Mat mRGB;
    torch::jit::script::Module mModule;
    std::vector<std::string> mClassnames;


    std::vector<std::string> mvDynamicNames;
    std::vector<cv::Rect2i> mvDynamicArea;
    std::map<std::string, std::vector<cv::Rect2i>> mmDetectMap;
    */

};

#endif