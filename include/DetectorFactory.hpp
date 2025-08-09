
#ifndef DFACTORY_HPP
#define DFACTORY_HPP
#include <memory>
#include "BaseDetector.h"
#include "YoloDetect.h"
#include "YRTDETRDetector.h"

class DetectorFactory {
public:
    static std::unique_ptr<BaseDetection> createDetector(const std::string& type) {
        if (type == "yolo") return std::make_unique<YoloDetection>();
        if (type == "detr") return std::make_unique<YDETRDetector>();
        return nullptr;
    }
};

#endif