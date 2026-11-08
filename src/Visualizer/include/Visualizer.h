#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

// Visualizer 只负责图像绘制和显示。
// 不负责 YOLO 推理，也不负责 PnP 位姿解算。
class Visualizer
{
public:
    Visualizer() = default;

    // 窗口资源由 Visualizer 管理。
    ~Visualizer();

    // 禁止复制，避免多个对象共同管理同一个窗口。
    Visualizer(const Visualizer&) = delete;
    Visualizer& operator=(const Visualizer&) = delete;

    // 创建显示窗口。
    void initialize(
        const std::string& window_name,
        int width,
        int height
    );

    // 绘制关键点及其序号。
    void draw_keypoints(
        cv::Mat& image,
        const std::vector<cv::Point2f>& points
    ) const;

    // 绘制绿色四边形。
    void draw_quadrilateral(
        cv::Mat& image,
        const std::vector<cv::Point2f>& points
    ) const;

    // 绘制青色投影矩形。
    void draw_projected_rectangle(
        cv::Mat& image,
        const std::vector<cv::Point2f>& points
    ) const;

    // 显示图像并返回键盘输入。
    int show(
        const cv::Mat& image,
        int delay = 1
    ) const;

    // 仅处理窗口事件和键盘输入，不显示新图像。
    int wait_key(int delay = 1) const;

    // 关闭当前窗口。
    void close();

private:
    std::string m_window_name;
    bool m_initialized = false;
};