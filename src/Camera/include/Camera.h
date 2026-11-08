
#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

class Camera
{
public:

    bool start(int device_id = 0);

    Camera();

    ~Camera();

    // 禁止拷贝，避免多个 Camera 对象错误共享同一个实现实例
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;

    // 初始化和释放 MVS SDK
    bool initialize();
    bool finalize();

    // 设备枚举和句柄创建
    bool enumerate_devices();
    int device_count() const;

    bool create_handle(int device_id = 0);

    // 相机设备控制
    bool open();
    bool set_camera_parameters();

    // 图像采集
    bool start_grabbing();
    bool get_frame(cv::Mat& image);

    bool stop_grabbing();
    bool close();

    bool is_opened() const;

    // 获取最近一次错误信息
    std::string last_error() const;

private:
    // PIMPL：内部实现类只在 Camera.cpp 中定义
    class Impl;

    // Camera 对象通过 unique_ptr 持有内部实现
    std::unique_ptr<Impl> m_impl;
};
