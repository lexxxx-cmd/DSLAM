#!/bin/bash

# TUM 数据集序列列表
sequences=(
    "walking_xyz"
    "walking_rpy"
    "walking_static"
    "walking_half"
    "sitting_xyz"
    "sitting_rpy"
    "sitting_static"
    "sitting_half"
)

# 基础路径设置
vocabulary="Vocabulary/ORBvoc.txt"
settings="Examples/RGB-D/TUM3.yaml"
data_root="/home/data/TUM/rgbd_dataset_freiburg3_"
traj_save_root="TrajSave/tum"

# 遍历所有序列
for seq in "${sequences[@]}"; do
    echo "=============================================="
    echo "处理序列: $seq"
    echo "=============================================="
    
    # 构造路径
    dataset_path="${data_root}${seq}"
    association_file="${dataset_path}/associations.txt"  # 移除下划线
    output_dir="${traj_save_root}/${seq}/"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 运行 ORB-SLAM3 生成轨迹
    echo "生成轨迹: $seq"
    ./Examples/RGB-D/rgbd_tum \
        "$vocabulary" \
        "$settings" \
        "$dataset_path" \
        "$association_file" \
        "$output_dir"
    
    # 重命名轨迹文件 (CameraTrajectory.txt -> oldCameraTrajectory.txt)
    if [ -f "${output_dir}/CameraTrajectory.txt" ]; then
        mv "${output_dir}/CameraTrajectory.txt" "${output_dir}/thisCameraTrajectory.txt"
        echo "轨迹文件已重命名"
    else
        echo "警告: 未找到 CameraTrajectory.txt 文件"
        continue
    fi
    
    # 运行轨迹评估
    echo "评估轨迹: $seq"
    evo_ape tum \
        "${dataset_path}/groundtruth.txt" \
        "${traj_save_root}/${seq}/thisCameraTrajectory.txt" \
        -va \
        --plot \
        --plot_mode xy \
        --save_plot "${traj_save_root}/${seq}/plot_xy.png" \
        --save_results "${traj_save_root}/${seq}/results.zip"
    
    echo "序列 $seq 处理完成"
    echo ""
done

echo "所有序列处理完成！"