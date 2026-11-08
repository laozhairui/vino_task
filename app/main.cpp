
#include "Camera.h"
#include "yolo_vino.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>


int main()
{
    // ========================================================
    // 1. 初始化相机
    // ========================================================

    Camera camera;

    // 初始化 MVS SDK。
    if (!camera.initialize())
    {
        std::cerr << "[Camera Error] "
                  << camera.last_error() << std::endl;
        return -1;
    }

    // 枚举当前连接的相机。
    if (!camera.enumerate_devices())
    {
        std::cerr << "[Camera Error] "
                  << camera.last_error() << std::endl;
        return -1;
    }

    std::cout << "[Camera] Found "
              << camera.device_count()
              << " camera(s)." << std::endl;

    // 创建第 0 个相机设备的句柄。
    if (!camera.create_handle(0))
    {
        std::cerr << "[Camera Error] "
                  << camera.last_error() << std::endl;
        return -1;
    }

    // 打开相机，并设置曝光、增益等参数。
    if (!camera.open())
    {
        std::cerr << "[Camera Error] "
                  << camera.last_error() << std::endl;
        return -1;
    }

    // 开始图像采集。
    if (!camera.start_grabbing())
    {
        std::cerr << "[Camera Error] "
                  << camera.last_error() << std::endl;
        return -1;
    }

    std::cout << "[Camera] Image acquisition started."
              << std::endl;


    // ========================================================
    // 2. 初始化 YOLOv8 Pose
    // ========================================================

    auto logger = std::make_unique<YoloVino::YoloVinoLogger>();

    // basic_info：显示基本运行信息和警告。
    // 避免实时推理时输出过多 debug 信息。
    logger->set_info_level(YoloVino::LoggerInfoLevel::basic_info);
    logger->print_yaml_info();

    YoloVino::Yolov8poseVino vino(std::move(logger));

    std::cout << "[YOLO] Model initialized."
              << std::endl;


    // ========================================================
    // 3. 实时采集、推理和可视化
    // ========================================================

    cv::namedWindow("Armor Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Armor Detection", 1200, 900);

    cv::Mat frame;

    while (true)
    {
        // ----------------------------------------------------
        // 3.1 从相机获取一帧图像
        // ----------------------------------------------------

        if (!camera.get_frame(frame))
        {
            std::cerr << "[Camera Warning] "
                      << camera.last_error() << std::endl;

            // 获取失败时跳过本次循环，继续尝试下一帧。
            if (cv::waitKey(1) == 27)
            {
                break;
            }

            continue;
        }

        // ----------------------------------------------------
        // 3.2 YOLOv8 Pose 推理
        // ----------------------------------------------------

        cv::Rect roi(
            0,
            0,
            frame.cols,
            frame.rows
        );

        std::vector<YoloVino::NNDetectData> results =
            vino.safe_predict(frame, roi);

        // ----------------------------------------------------
        // 3.3 绘制检测框和关键点
        // ----------------------------------------------------

        for (const auto& det : results)
        {
            // 绘制装甲板检测框。
            if (det.rect.width > 0 && det.rect.height > 0)
            {
                cv::rectangle(
                    frame,
                    det.rect,
                    cv::Scalar(0, 255, 0),
                    2
                );
            }

            // 绘制类别和置信度。
            std::string label =
                "ID:" + std::to_string(det.class_id) +
                " conf:" + cv::format("%.2f", det.confidence);

            int base_line = 0;

            cv::Size label_size = cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &base_line
            );

            int label_y = std::max(
                det.rect.y,
                label_size.height + base_line
            );

            cv::putText(
                frame,
                label,
                cv::Point(det.rect.x, label_y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 255, 0),
                1
            );

            // 绘制 YOLOv8 Pose 输出的关键点。
            for (std::size_t i = 0; i < det.keypoints.size(); ++i)
            {
                const auto& keypoint = det.keypoints[i];

                int x = static_cast<int>(keypoint.x);
                int y = static_cast<int>(keypoint.y);

                // 只绘制位于图像范围内的关键点。
                if (x < 0 || x >= frame.cols
                    || y < 0 || y >= frame.rows)
                {
                    continue;
                }

                cv::circle(
                    frame,
                    cv::Point(x, y),
                    4,
                    cv::Scalar(0, 0, 255),
                    cv::FILLED
                );

                // 标记关键点序号，方便后续核对 PnP 点的顺序。
                cv::putText(
                    frame,
                    std::to_string(i),
                    cv::Point(x + 5, y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.4,
                    cv::Scalar(255, 0, 0),
                    1
                );
            }
        }

        // ----------------------------------------------------
        // 3.4 显示实时画面
        // ----------------------------------------------------

        cv::imshow("Armor Detection", frame);

        // 等待 1 ms，检测退出按键。
        int key = cv::waitKey(1);

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }


    // ========================================================
    // 4. 释放资源
    // ========================================================

    camera.stop_grabbing();
    camera.close();
    camera.finalize();

    cv::destroyAllWindows();

    std::cout << "[System] Program exited."
              << std::endl;

    return 0;
}
