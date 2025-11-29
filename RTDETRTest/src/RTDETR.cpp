#pragma once
#include "RTDETR.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cassert>


RTDETRv4::RTDETRv4(const std::string& engine_path, int input_size) 
    : input_size_(input_size) {
    
    runtime_ = nvinfer1::createInferRuntime(logger_);
    assert(runtime_ != nullptr);
    
    buildEngine(engine_path);
    
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
    
    CHECK(cudaStreamCreate(&stream_));
    
    allocateBuffers();
    //debug
    for (const auto& output : outputs_) {
        std::cout << output.name << ": dims = ";
        for (int i = 0; i < output.dims.nbDims; ++i) {
            std::cout << output.dims.d[i] << " ";
    }
    std::cout << std::endl;
}
}

RTDETRv4::~RTDETRv4() {
    for (auto& input : inputs_) {
        if (input.device_ptr) CHECK(cudaFree(input.device_ptr));
        if (input.host_ptr) CHECK(cudaFreeHost(input.host_ptr));
    }
    for (auto& output : outputs_) {
        if (output.device_ptr) CHECK(cudaFree(output.device_ptr));
        if (output.host_ptr) CHECK(cudaFreeHost(output.host_ptr));
    }
    
    cudaStreamDestroy(stream_);
    delete context_;
    delete engine_;
    delete runtime_;
}

void RTDETRv4::buildEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    assert(file.good());
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    
    
    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    assert(engine_ != nullptr);
}

void RTDETRv4::allocateBuffers() {
    int num_io_tensors = engine_->getNbIOTensors();
    
    for (int i = 0; i < num_io_tensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        nvinfer1::DataType dtype = engine_->getTensorDataType(name);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        
        TensorInfo info;
        info.name = name;
        info.dims = dims;
        
        if (dims.d[0] == -1) {
            dims.d[0] = 1;
        }
        
        info.size = getTensorSize(dims) * getElementSize(dtype);
        
        CHECK(cudaMalloc(&info.device_ptr, info.size));
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputs_.push_back(info);
        } else {
            CHECK(cudaHostAlloc(&info.host_ptr, info.size, cudaHostAllocDefault));
            outputs_.push_back(info);
        }
        std::cout << "Tensor " << info.name << ", dims: ";
        for (int i = 0; i < dims.nbDims; ++i)
            std::cout << dims.d[i] << " ";
        std::cout << ", size(bytes): " << info.size << std::endl;
    }
    

}

cv::Mat RTDETRv4::preprocess(const cv::Mat& image) {
    preprocess_params_.orig_width = image.cols;
    preprocess_params_.orig_height = image.rows;
    
    float scale_w = static_cast<float>(input_size_) / image.cols;
    float scale_h = static_cast<float>(input_size_) / image.rows;
    preprocess_params_.scale = std::min(scale_w, scale_h);
    
    int new_w = static_cast<int>(image.cols * preprocess_params_.scale);
    int new_h = static_cast<int>(image.rows * preprocess_params_.scale);
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));
    
    preprocess_params_.pad_w = (input_size_ - new_w) / 2;
    preprocess_params_.pad_h = (input_size_ - new_h) / 2;
    
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, 
                       preprocess_params_.pad_h, input_size_ - new_h - preprocess_params_.pad_h,
                       preprocess_params_.pad_w, input_size_ - new_w - preprocess_params_.pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    cv::Mat blob;
    blob.create({1, 3, input_size_, input_size_}, CV_32F);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(input_size_, input_size_, CV_32F, 
                              blob.ptr<float>() + i * input_size_ * input_size_);
    }
    cv::split(normalized, channels);
    
    return blob;
}

void RTDETRv4::copyInputToDevice(const cv::Mat& preprocessed) {
    CHECK(cudaMemcpyAsync(inputs_[0].device_ptr, preprocessed.ptr<float>(), 
                          inputs_[0].size, cudaMemcpyHostToDevice, stream_));
    
    context_->setInputShape(inputs_[0].name.c_str(), 
                           nvinfer1::Dims4{1, 3, input_size_, input_size_});
    context_->setTensorAddress(inputs_[0].name.c_str(), inputs_[0].device_ptr);
    
    if (inputs_.size() > 1) {
        float orig_sizes[2] = {
            static_cast<float>(preprocess_params_.orig_width), 
            static_cast<float>(preprocess_params_.orig_height)
        };
        CHECK(cudaMemcpyAsync(inputs_[1].device_ptr, orig_sizes, 
                              2 * sizeof(float), cudaMemcpyHostToDevice, stream_));
        context_->setInputShape(inputs_[1].name.c_str(), nvinfer1::Dims2{1, 2});
        context_->setTensorAddress(inputs_[1].name.c_str(), inputs_[1].device_ptr);
    }
    
    for (auto& output : outputs_) {
        context_->setTensorAddress(output.name.c_str(), output.device_ptr);
    }
}

void RTDETRv4::infer() {
    for (int i = 0; i < inputs_.size(); ++i) {
        std::cout << "Input " << inputs_[i].name << ": device_ptr = " << inputs_[i].device_ptr
                << ", dims = ";
        for (int d=0; d < inputs_[i].dims.nbDims; ++d)
        std::cout << inputs_[i].dims.d[d] << " ";
        std::cout << std::endl;
    }

    bool success = context_->enqueueV3(stream_);
    if (!success) {
        std::cerr << "TensorRT inference failed!" << std::endl;
    }
    
    for (auto& output : outputs_) {
        CHECK(cudaMemcpyAsync(output.host_ptr, output.device_ptr, 
                              output.size, cudaMemcpyDeviceToHost, stream_));
    }
    
    CHECK(cudaStreamSynchronize(stream_));
}

std::vector<Detection> RTDETRv4::postprocess(float conf_thresh) {
    std::vector<Detection> detections;
    
    float* labels_ptr = nullptr;
    float* boxes_ptr = nullptr;
    float* scores_ptr = nullptr;
    
    for (auto& output : outputs_) {
        if (output.name.find("labels") != std::string::npos) {
            labels_ptr = static_cast<float*>(output.host_ptr);
        } else if (output.name.find("boxes") != std::string::npos) {
            boxes_ptr = static_cast<float*>(output.host_ptr);
        } else if (output.name.find("scores") != std::string::npos) {
            scores_ptr = static_cast<float*>(output.host_ptr);
        }
    }
    //debug
    for (auto& output : outputs_) {
        std::cout << "Output " << output.name << " host_ptr: " << output.host_ptr
                << ", device_ptr: " << output.device_ptr << ", size: " << output.size << std::endl;
    }

    if (boxes_ptr) {
        std::cout << "Boxes first 8 floats: ";
        for (int i = 0; i < 8; ++i) {
            std::cout << boxes_ptr[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Boxes pointer not found!" << std::endl;
    }//
    
    if (!labels_ptr || !boxes_ptr || !scores_ptr) return detections;
    
    int num_detections = outputs_[0].dims.d[1];
    
    for (int i = 0; i < num_detections; ++i) {
        float score = scores_ptr[i];
        if (score < conf_thresh) continue;
        
        Detection det;
        det.confidence = score;
        det.class_id = static_cast<int>(labels_ptr[i]);

        float x1 = boxes_ptr[i * 4 + 0];
        float y1 = boxes_ptr[i * 4 + 1];
        float x2 = boxes_ptr[i * 4 + 2];
        float y2 = boxes_ptr[i * 4 + 3];
        
        det.bbox = cv::Rect2f(x1, y1, x2-x1, y2-y1);
        
        detections.push_back(det);
    }
    
    return detections;
}

std::vector<Detection> RTDETRv4::detect(const cv::Mat& image, float conf_thresh) {
    cv::Mat preprocessed = preprocess(image);
    copyInputToDevice(preprocessed);
    infer();
    return postprocess(conf_thresh);
}

void RTDETRv4::warmup(int iterations) {
    cv::Mat dummy(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    for (int i = 0; i < iterations; ++i) {
        detect(dummy, 0.5);
    }
}

size_t RTDETRv4::getElementSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT8: return 1;
        default: return 0;
    }
}

size_t RTDETRv4::getTensorSize(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}