cmake_minimum_required(VERSION 3.18)
project(RTDETR_Test)
SET(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_definitions(-DTRT_10)
# 1. 查找 TensorRT 头文件路径
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS /usr/include /usr/include/x86_64-linux-gnu # 增加可能的搜索路径
)

# 2. 查找核心库文件
find_library(TENSORRT_INFER_LIBRARY
    NAMES nvinfer
)

# 3. 查找 ONNX 解析器库文件
find_library(TENSORRT_ONNX_PARSER_LIBRARY
    NAMES nvonnxparser
)

# 可选：查找插件库（RTDETR可能需要）
find_library(TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
)

# 4. 检查是否找到所有必需文件
if (TENSORRT_INCLUDE_DIR AND TENSORRT_INFER_LIBRARY AND TENSORRT_ONNX_PARSER_LIBRARY)
    message(STATUS "Found TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "Found TensorRT Infer Lib: ${TENSORRT_INFER_LIBRARY}")
    
    # 5. 添加头文件路径
    include_directories(${TENSORRT_INCLUDE_DIR})

    # 6. 定义链接库变量
    set(RTDETR_TENSORRT_LIBRARIES 
        ${TENSORRT_INFER_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
        ${TENSORRT_PLUGIN_LIBRARY} # 包含插件库
    )
else()
    message(FATAL_ERROR "TensorRT libraries or headers not found. Check if libnvinfer-dev and libnvonnxparser-dev are installed.")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include  # 放RTDETR.h等头文件
)

add_executable(test_rt_detr main.cc include/rtdetr_opencv.h src/rtdetr_opencv.cpp)

message(STATUS "CUDAToolkit_LIBRARIES = ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS = ${CUDAToolkit_INCLUDE_DIRS}")
# message(STATUS "CUDAToolkit_LIBRARIES = ${CUDAToolkit_LIBRARY_DIR}")
target_link_directories(test_rt_detr PUBLIC ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(test_rt_detr 
    ${OpenCV_LIBS}
    ${RTDETR_TENSORRT_LIBRARIES}
    cudart
)
