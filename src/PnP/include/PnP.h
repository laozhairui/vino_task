
#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace PnP
{

// 保存一次 PnP 姿态解算结果。
struct PnPResult
{
    // Rodrigues 旋转向量。
    cv::Vec3d rotation_vector{0.0, 0.0, 0.0};

    // 平移向量，单位：mm。
    cv::Vec3d translation_vector{0.0, 0.0, 0.0};

    // 灯条模型坐标系到相机坐标系的旋转矩阵。
    cv::Matx33d rotation_world_to_camera =
        cv::Matx33d::eye();

    // 相机坐标系到灯条模型坐标系的旋转矩阵。
    cv::Matx33d rotation_camera_to_world =
        cv::Matx33d::eye();

    // ZYX 欧拉角，单位：度。
    double yaw = 0.0;
    double pitch = 0.0;
    double roll = 0.0;

    // 相机光心到灯条模型原点的直线距离，单位：mm。
    double distance = 0.0;

    // 沿相机 Z 轴的深度，单位：mm。
    double depth = 0.0;

    // 相机在灯条模型坐标系中的位置，单位：mm。
    cv::Vec3d camera_position{0.0, 0.0, 0.0};

    // 四个端点的 RMS 重投影误差，单位：像素。
    double reprojection_error = 0.0;
};


class PnPSolver
{
public:

    PnPSolver() = default;

    // 从 YAML 文件加载相机内参和畸变参数。
    bool load_camera_parameters(
        const std::string& file_path
    );

    // 直接设置相机内参和畸变参数。
    bool set_camera_parameters(
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs
    );

    // 使用四个灯条端点进行 PnP 解算。
    //
    // image_points 顺序必须为：
    // 0：左灯条上端
    // 1：左灯条下端
    // 2：右灯条下端
    // 3：右灯条上端
    //
    // 默认模型尺寸：
    // 左右灯条中心距离：130 mm
    // 单根灯条上下端点距离：60 mm
    //
    // 解算成功返回 true，否则返回 false。
    bool solve(
        const std::vector<cv::Point2f>& image_points,
        PnPResult& result,
        double light_bar_center_distance = 130.0,
        double light_bar_length = 60.0
    ) const;

    // 检查相机参数是否已经成功加载。
    bool is_ready() const;

    // 打印相机内参和畸变参数。
    void print_camera_parameters() const;

    // 打印 PnP 解算结果。
    static void print_result(
        const PnPResult& result
    );

private:

    // 验证输入的四个二维关键点。
    static bool validate_image_points(
        const std::vector<cv::Point2f>& image_points
    );

    // 将旋转矩阵转换为 ZYX 欧拉角。
    static void rotation_matrix_to_euler_zyx(
        const cv::Matx33d& rotation_matrix,
        double& yaw,
        double& pitch,
        double& roll
    );

private:

    // 相机内参矩阵。
    cv::Mat m_camera_matrix;

    // 相机畸变系数。
    cv::Mat m_dist_coeffs;
};

}  // namespace PnP
