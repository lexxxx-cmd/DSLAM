#!/bin/bash

# TUM 数据集序列列表（与轨迹保存文件夹名称对应）
sequences=(
    "walking_xyz"
    "walking_rpy"
    "walking_static"
    "walking_halfsphere"
    "sitting_xyz"
    "sitting_rpy"
    "sitting_static"
    "sitting_halfsphere"
)

sequences2=(
    "balloon"
    "crowd"
    "moving_nonobstructing_box"
    "moving_obstructing_box"
    "person_tracking"
    "placing_nonobstructing_box"
)

# 基础路径设置
data_root="/home/data/TUM/rgbd_dataset_freiburg3_"  # 地面真值数据根路径
data_root2="/home/data/TUM/Bonn/rgbd_bonn_"
traj_save_root="TrajSave/tum"
traj_save_root2="TrajSave/bonn"                          # 轨迹文件保存根路径

# 遍历所有序列，生成轨迹对比图
for seq in "${sequences[@]}"; do
    echo "=============================================="
    echo "处理序列: $seq"
    echo "=============================================="
    
    # 构造路径
    dataset_path="${data_root}${seq}"                   # 数据集完整路径（包含地面真值）
    groundtruth_file="${dataset_path}/groundtruth.txt"   # 地面真值文件路径
    trajectory_file="${traj_save_root}/${seq}/thisCameraTrajectory.txt"  # 待评估轨迹文件路径
    plot_save_dir="${traj_save_root}/${seq}/"            # 对比图保存目录
    
    # 检查地面真值文件是否存在
    if [ ! -f "$groundtruth_file" ]; then
        echo "警告: 未找到地面真值文件 $groundtruth_file"
        continue
    fi
    
    # 检查轨迹文件是否存在
    if [ ! -f "$trajectory_file" ]; then
        echo "警告: 未找到轨迹文件 $trajectory_file"
        continue
    fi
    
    # 创建对比图保存目录（确保存在）
    mkdir -p "$plot_save_dir"
    
    # # 运行 evo_traj 生成轨迹对比图（xyz模式）并保存
    # echo "生成轨迹对比图: $seq"
    # evo_traj tum \
    #     "$groundtruth_file" \
    #     "$trajectory_file" \
    #     --ref="$groundtruth_file" \
    #     -p \
    #     --plot_mode=xyz \
    #     --save_plot="$plot_save_dir" \
    #     -as
    # 运行 evo_traj 生成轨迹对比图（xyz模式）并保存
    echo "生成轨迹对比图: $seq"
    evo_rpe tum \
        "$groundtruth_file" \
        "$trajectory_file" \
        -as \
    
    echo "序列 $seq 对比图生成完成，保存至: $plot_save_dir"
    echo ""
done

echo "所有序列轨迹对比图生成完成！"