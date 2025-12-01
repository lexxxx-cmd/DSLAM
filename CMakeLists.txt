cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})



set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCOMPILEDWITHC14)
    message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

add_definitions(-w)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/Thirdparty/libtorch)


# if(NOT OpenCV_CUDA_VERSION)
#     message(FATAL_ERROR "未检测到支持 CUDA 的 OpenCV 构建。请重新编译 OpenCV 并设置 -DWITH_CUDA=ON")
# else()
#     message(STATUS "Found OpenCV CUDA Version: ${OpenCV_CUDA_VERSION}")
# endif()


find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/mnt/workspace/YOLO_ORB_SLAM3/RTDETRTest/Thirdparty/libtorch")
find_package(Torch REQUIRED)
# find_package(CUDAToolkit REQUIRED)
# add_definitions(-DTRT_10)
# # 1. 查找 TensorRT 头文件路径
# find_path(TENSORRT_INCLUDE_DIR
#     NAMES NvInfer.h
#     PATHS /usr/include /usr/include/x86_64-linux-gnu # 增加可能的搜索路径
# )

# # 2. 查找核心库文件
# find_library(TENSORRT_INFER_LIBRARY
#     NAMES nvinfer
# )

# # 3. 查找 ONNX 解析器库文件
# find_library(TENSORRT_ONNX_PARSER_LIBRARY
#     NAMES nvonnxparser
# )

# # 可选：查找插件库（RTDETR可能需要）
# find_library(TENSORRT_PLUGIN_LIBRARY
#     NAMES nvinfer_plugin
# )

# # 4. 检查是否找到所有必需文件
# if (TENSORRT_INCLUDE_DIR AND TENSORRT_INFER_LIBRARY AND TENSORRT_ONNX_PARSER_LIBRARY)
#     message(STATUS "Found TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
#     message(STATUS "Found TensorRT Infer Lib: ${TENSORRT_INFER_LIBRARY}")
    
#     # 5. 添加头文件路径
#     include_directories(${TENSORRT_INCLUDE_DIR})

#     # 6. 定义链接库变量
#     set(RTDETR_TENSORRT_LIBRARIES 
#         ${TENSORRT_INFER_LIBRARY}
#         ${TENSORRT_ONNX_PARSER_LIBRARY}
#         ${TENSORRT_PLUGIN_LIBRARY} # 包含插件库
#     )
# else()
#     message(FATAL_ERROR "TensorRT libraries or headers not found. Check if libnvinfer-dev and libnvonnxparser-dev are installed.")
# endif()
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
# 强制指定编译器，防止 CMake 自动去抓 /usr/bin/nvcc (即 CUDA 12)
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

enable_language(CUDA)
set(TRT_ROOT "~/Softwares/TensorRT-10.12.0.36/")

find_package(OpenCV 4.2)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()
find_package(PCL 1.12 REQUIRED)
add_library(PointCloudMapper STATIC
        src/PointCloud.cpp
)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(${TRT_ROOT}/include)

set_target_properties(PointCloudMapper PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(PointCloudMapper PUBLIC
        ${PCL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${OPENCV_INCLUDE_DIR}
)
target_compile_definitions(PointCloudMapper PUBLIC
        ${PCL_DEFINITIONS}
)
target_link_libraries(PointCloudMapper PUBLIC ${PCL_LIBRARIES} ${OpenCV_LIBS})

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
        ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        #${CUDAToolkit_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Atlas.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Viewer.cc
        src/ImuTypes.cc
        src/G2oTypes.cc
        src/CameraModels/Pinhole.cpp
        src/CameraModels/KannalaBrandt8.cpp
        src/OptimizableTypes.cpp
        src/MLPnPsolver.cpp
        src/GeometricTools.cc
        src/TwoViewReconstruction.cc
        src/Config.cc
        src/Settings.cc
        src/YoloDetect.cpp
        src/YRTDETRDetector.cpp
        src/RTdetr_opencv.cpp
        include/System.h
        include/Tracking.h
        include/LocalMapping.h
        include/LoopClosing.h
        include/ORBextractor.h
        include/ORBmatcher.h
        include/FrameDrawer.h
        include/Converter.h
        include/MapPoint.h
        include/KeyFrame.h
        include/Atlas.h
        include/Map.h
        include/MapDrawer.h
        include/Optimizer.h
        include/Frame.h
        include/KeyFrameDatabase.h
        include/Sim3Solver.h
        include/Viewer.h
        include/ImuTypes.h
        include/G2oTypes.h
        include/CameraModels/GeometricCamera.h
        include/CameraModels/Pinhole.h
        include/CameraModels/KannalaBrandt8.h
        include/OptimizableTypes.h
        include/MLPnPsolver.h
        include/GeometricTools.h
        include/TwoViewReconstruction.h
        include/SerializationUtils.h
        include/Config.h
        include/Settings.h
        include/BaseDetector.h
        include/DetectorFactory.hpp
        include/YRTDETRDetector.h
        include/RTdetr_opencv.h
        include/Common.hpp)

add_subdirectory(Thirdparty/g2o)
#target_link_directories(${PROJECT_NAME} PUBLIC ${CUDAToolkit_LIBRARY_DIR})
link_directories(${TRT_ROOT}/lib)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_libraries(${PROJECT_NAME} PUBLIC
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${TORCH_LIBRARIES}
        -lboost_serialization
        -lcrypto
        PointCloudMapper
        # ${RTDETR_TENSORRT_LIBRARIES}
        /usr/local/cuda-11.8/lib64/libcudart.so
        ${TRT_ROOT}/lib/libnvinfer.so
        ${TRT_ROOT}/lib/libnvinfer_plugin.so
        ${TRT_ROOT}/lib/libnvonnxparser.so
    )

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
            ${realsense_INCLUDE_DIR}
            )
    target_link_libraries(${PROJECT_NAME}
            ${realsense2_LIBRARY}
            )
endif()


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME} PointCloudMapper)

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Examples/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i
            Examples/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Examples/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_realsense_D435i
            Examples/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i
            Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i
            Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Examples/Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_T265
            Examples/Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()
