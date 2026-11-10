
#include "PnP.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <limits>

namespace PnP
{

// ============================================================
// 1. 加载相机参数
// ============================================================

bool PnPSolver::load_camera_parameters(
    const std::string& file_path
)
{
    cv::FileStorage file_storage(
        file_path,
        cv::FileStorage::READ
    );

    if (!file_storage.isOpened())
    {
        std::cerr << "[PnP Error] Cannot open camera calibration file: "
                  << file_path << std::endl;
        return false;
    }

    file_storage["camera_matrix"] >> m_camera_matrix;
    file_storage["dist_coeffs"] >> m_dist_coeffs;

    file_storage.release();

    if (m_camera_matrix.empty() || m_dist_coeffs.empty())
    {
        std::cerr << "[PnP Error] Camera parameters are missing."
                  << std::endl;
        return false;
    }

    // 统一转换为 double 类型，便于后续计算。
    m_camera_matrix.convertTo(m_camera_matrix, CV_64F);
    m_dist_coeffs.convertTo(m_dist_coeffs, CV_64F);

    if (m_camera_matrix.rows != 3
        || m_camera_matrix.cols != 3)
    {
        std::cerr << "[PnP Error] Invalid camera matrix size."
                  << std::endl;
        return false;
    }

    if (m_dist_coeffs.total() < 4)
    {
        std::cerr << "[PnP Error] Invalid distortion coefficients."
                  << std::endl;
        return false;
    }

    const double fx = m_camera_matrix.at<double>(0, 0);
    const double fy = m_camera_matrix.at<double>(1, 1);

    if (fx <= 0.0 || fy <= 0.0)
    {
        std::cerr << "[PnP Error] Invalid focal length."
                  << std::endl;
        return false;
    }

    std::cout << "[PnP] Camera parameters loaded successfully."
              << std::endl;

    return true;
}


// ============================================================
// 2. 手动设置相机参数
// ============================================================

bool PnPSolver::set_camera_parameters(
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs
)
{
    if (camera_matrix.empty() || dist_coeffs.empty())
    {
        std::cerr << "[PnP Error] Empty camera parameters."
                  << std::endl;
        return false;
    }

    if (camera_matrix.rows != 3
        || camera_matrix.cols != 3
        || dist_coeffs.total() < 4)
    {
        std::cerr << "[PnP Error] Invalid camera parameter dimensions."
                  << std::endl;
        return false;
    }

    camera_matrix.convertTo(m_camera_matrix, CV_64F);
    dist_coeffs.convertTo(m_dist_coeffs, CV_64F);

    return is_ready();
}


// ============================================================
// 3. 检查相机参数
// ============================================================

bool PnPSolver::is_ready() const
{
    if (m_camera_matrix.empty() || m_dist_coeffs.empty())
    {
        return false;
    }

    if (m_camera_matrix.rows != 3
        || m_camera_matrix.cols != 3)
    {
        return false;
    }

    const double fx = m_camera_matrix.at<double>(0, 0);
    const double fy = m_camera_matrix.at<double>(1, 1);

    return fx > 0.0 && fy > 0.0;
}


// ============================================================
// 4. 验证输入的四个二维关键点
// ============================================================

bool PnPSolver::validate_image_points(
    const std::vector<cv::Point2f>& image_points
)
{
    if (image_points.size() != 4)
    {
        std::cerr << "[PnP Error] Exactly four image points are required."
                  << std::endl;
        return false;
    }

    // 检查坐标是否有效。
    for (const auto& point : image_points)
    {
        if (!std::isfinite(point.x)
            || !std::isfinite(point.y))
        {
            std::cerr << "[PnP Error] Invalid image point."
                      << std::endl;
            return false;
        }
    }

    // 检查是否存在过于接近的点。
    for (std::size_t i = 0; i < image_points.size(); ++i)
    {
        for (std::size_t j = i + 1; j < image_points.size(); ++j)
        {
            const double distance = cv::norm(
                image_points[i] - image_points[j]
            );

            if (distance < 5.0)
            {
                std::cerr << "[PnP Error] Two image points are too close."
                          << std::endl;
                return false;
            }
        }
    }

    // 当前输入顺序：
    // 左上、左下、右下、右上。
    // 该顺序沿四边形轮廓排列，可以直接计算面积。
    const double area = std::abs(
        cv::contourArea(image_points)
    );

    if (area < 20.0)
    {
        std::cerr << "[PnP Error] Image quadrilateral area is too small."
                  << std::endl;
        return false;
    }

    return true;
}


// ============================================================
// 5. 旋转矩阵转换为 ZYX 欧拉角
// ============================================================

void PnPSolver::rotation_matrix_to_euler_zyx(
    const cv::Matx33d& rotation_matrix,
    double& roll,  //原本为z,y,x  
    double& yaw,
    double& pitch
)
{
    const double r31 = rotation_matrix(2, 0);
    const double r32 = rotation_matrix(2, 1);
    const double r33 = rotation_matrix(2, 2);

    const double r21 = rotation_matrix(1, 0);
    const double r11 = rotation_matrix(0, 0);

    // pitch = asin(-r31)  y
    yaw = std::asin(
        std::clamp(-r31, -1.0, 1.0)
    );

    const double cos_yaw = std::cos(yaw);

    // 处理万向节锁定附近的情况。
    if (std::abs(cos_yaw) > 1e-6)
    {
        pitch = std::atan2(r32, r33);//x
        roll = std::atan2(r21, r11);//z
    }
    else
    {
        // 万向节锁定时，将 roll 固定为 0。
        pitch = 0.0; //x

        roll = std::atan2( //z
            -rotation_matrix(0, 1),
            rotation_matrix(1, 1)
        );
    }

    // 弧度转换为角度。
    constexpr double RAD_TO_DEG = 180.0 / CV_PI;

    roll *= RAD_TO_DEG;
    yaw *= RAD_TO_DEG;
    pitch *= RAD_TO_DEG;
}


// ============================================================
// 6. PnP 核心解算
// ============================================================

bool PnPSolver::solve(
    const std::vector<cv::Point2f>& image_points,
    PnPResult& result,
    double light_bar_center_distance,
    double light_bar_length
) const
{
    if (!is_ready())
    {
        std::cerr << "[PnP Error] Camera parameters are not ready."
                  << std::endl;
        return false;
    }

    if (!validate_image_points(image_points))
    {
        return false;
    }

    if (!std::isfinite(light_bar_center_distance)
        || !std::isfinite(light_bar_length)
        || light_bar_center_distance <= 0.0
        || light_bar_length <= 0.0)
    {
        std::cerr << "[PnP Error] Invalid light bar model dimensions."
                  << std::endl;
        return false;
    }

    // --------------------------------------------------------
    // 6.1 建立灯条端点的三维模型
    // --------------------------------------------------------

    const float half_width = static_cast<float>(
        light_bar_center_distance / 2.0
    );

    const float half_height = static_cast<float>(
        light_bar_length / 2.0
    );

    // 单位：mm。
    //
    // 顺序必须与 image_points 完全一致：
    // 0：左灯条上端
    // 1：左灯条下端
    // 2：右灯条下端
    // 3：右灯条上端
    //
    // 坐标系：
    // X：向右
    // Y：向下
    // Z：垂直灯条所在平面
    const std::vector<cv::Point3f> object_points = {
        {-half_width, -half_height, 0.0F},
        {-half_width,  half_height, 0.0F},
        { half_width,  half_height, 0.0F},
        { half_width, -half_height, 0.0F}
    };

    // --------------------------------------------------------
    // 6.2 使用 solvePnP 求解旋转和平移
    // --------------------------------------------------------

    cv::Vec3d rotation_vector;
    cv::Vec3d translation_vector;

    const bool success = cv::solvePnP(
        object_points,
        image_points,
        m_camera_matrix,
        m_dist_coeffs,
        rotation_vector,
        translation_vector,
        false,
        cv::SOLVEPNP_ITERATIVE
    );

    if (!success)
    {
        std::cerr << "[PnP Error] solvePnP failed."
                  << std::endl;
        return false;
    }

    // --------------------------------------------------------
    // 6.3 Rodrigues 旋转向量转换为旋转矩阵
    // --------------------------------------------------------

    cv::Mat rotation_matrix_mat;

    cv::Rodrigues(
        rotation_vector,
        rotation_matrix_mat
    );

    cv::Mat rotation_matrix_64;

    rotation_matrix_mat.convertTo(
        rotation_matrix_64,
        CV_64F
    );

    cv::Matx33d rotation_world_to_camera;

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            rotation_world_to_camera(row, col) =
                rotation_matrix_64.at<double>(row, col);
        }
    }

    // 旋转矩阵正交，逆矩阵等于转置矩阵。
    const cv::Matx33d rotation_camera_to_world =
        rotation_world_to_camera.t();

    
    // --------------------------------------------------------
    // 6.4 计算 ZYX 欧拉角
    // --------------------------------------------------------

    double yaw = 0.0;
    double pitch = 0.0;
    double roll = 0.0;

    rotation_matrix_to_euler_zyx(
        rotation_world_to_camera,
        roll,
        yaw,
        pitch
    );

    // --------------------------------------------------------
    // 6.5 计算距离
    // --------------------------------------------------------

    // 相机光心到灯条模型原点的直线距离。
    const double distance = cv::norm(translation_vector);

    // 沿相机 Z 轴的深度。
    const double depth = translation_vector[2];

    // 相机在灯条模型坐标系中的位置：
    // C = -R^T * t
    const cv::Vec3d camera_position =
        -(rotation_camera_to_world * translation_vector);

    // --------------------------------------------------------
    // 6.6 计算重投影误差
    // --------------------------------------------------------

    std::vector<cv::Point2f> projected_points;

    cv::projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        m_camera_matrix,
        m_dist_coeffs,
        projected_points
    );

    double squared_error = 0.0;

    for (std::size_t i = 0; i < image_points.size(); ++i)
    {
        const double error = cv::norm(
            image_points[i] - projected_points[i]
        );

        squared_error += error * error;
    }

    const double reprojection_error = std::sqrt(
        squared_error / image_points.size()
    );

    // --------------------------------------------------------
    // 6.7 保存结果
    // --------------------------------------------------------

    result.rotation_vector = rotation_vector;
    result.translation_vector = translation_vector;

    result.rotation_world_to_camera =
        rotation_world_to_camera;

    result.rotation_camera_to_world =
        rotation_camera_to_world;

    result.yaw = yaw;
    result.pitch = pitch;
    result.roll = roll;

    result.distance = distance;
    result.depth = depth;

    result.camera_position = camera_position;

    result.reprojection_error = reprojection_error;

    return true;
}


// ============================================================
// 7. 打印相机参数
// ============================================================

void PnPSolver::print_camera_parameters() const
{
    if (!is_ready())
    {
        std::cout << "[PnP] Camera parameters are not ready."
                  << std::endl;
        return;
    }

    std::cout << "\n========== Camera Parameters =========="
              << std::endl;

    std::cout << "Camera Matrix:\n"
              << m_camera_matrix << std::endl;

    std::cout << "Distortion Coefficients:\n"
              << m_dist_coeffs << std::endl;

    std::cout << "=======================================\n"
              << std::endl;
}


// ============================================================
// 8. 打印 PnP 结果
// ============================================================

void PnPSolver::print_result(const PnPResult& result)
{
    std::cout << "\n========== PnP Result =========="
              << std::endl;

    std::cout << "Yaw: "
              << result.yaw << " deg"
              << std::endl;

    std::cout << "Pitch: "
              << result.pitch << " deg"
              << std::endl;

    std::cout << "Roll: "
              << result.roll << " deg"
              << std::endl;

    std::cout << "Distance: "
              << result.distance << " mm"
              << std::endl;

    std::cout << "Depth: "
              << result.depth << " mm"
              << std::endl;

    std::cout << "Translation: "
              << result.translation_vector
              << std::endl;

    std::cout << "Camera Position: "
              << result.camera_position
              << std::endl;

    std::cout << "Reprojection Error: "
              << result.reprojection_error << " px"
              << std::endl;

    std::cout << "================================\n"
              << std::endl;

    std::cout << "Armor -> Camera:\n"
              << result.rotation_world_to_camera;

    std::cout << "================================\n"
              << std::endl;

    std::cout << "================================\n"
              << std::endl;

    std::cout << "Camera -> world:\n"
              << result.rotation_camera_to_world;

    std::cout << "================================\n"
              << std::endl;
}

bool PnPSolver::project_points(
    const PnPResult& result,
    std::vector<cv::Point2f>& image_points,
    double armor_length,
    double armor_width
) const
{

    const float armor_half_width = static_cast<float>(
        armor_width / 2.0
    );

    const float armor_half_length = static_cast<float>(
        armor_length / 2.0
    );

    const std::vector<cv::Point3f> armor_points = {
        {-armor_half_length, -armor_half_width, 0.0F},
        {-armor_half_length,  armor_half_width, 0.0F},
        { armor_half_length,  armor_half_width, 0.0F},
        { armor_half_length, -armor_half_width, 0.0F}
    };

    if (!is_ready() || armor_points.empty())
    {
        return false;
    }

    cv::projectPoints(
        armor_points,
        result.rotation_vector,
        result.translation_vector,
        m_camera_matrix,
        m_dist_coeffs,
        image_points
    );

    return image_points.size() == armor_points.size();
}

}  // namespace PnP
