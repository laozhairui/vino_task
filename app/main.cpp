
#include "Camera.h"
#include "yolo_vino.hpp"
#include "PnP.h"
#include "Visualizer.h"

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

    if (!camera.start())
    {
        std::cerr << "[Camera Error]" << camera.last_error() << '\n';

        return -1;
    }

    std::cout << "[Camera] Found " << camera.device_count() << " camera(s)." << '\n';

    std::cout << "[Camera] Image acquisition started."
              << std::endl;


    // ========================================================
    // 2. 初始化 YOLOv8 Pose
    // ========================================================

    auto logger = std::make_unique<YoloVino::YoloVinoLogger>(
        "../src/YoloVino/config/yolov5fourpoint_vino_config.yaml"
    );

    // basic_info：显示基本运行信息和警告。
    // 避免实时推理时输出过多 debug 信息。
    logger->set_info_level(YoloVino::LoggerInfoLevel::basic_info);
    logger->print_yaml_info();

    YoloVino::Yolov5fourpointVino vino(std::move(logger));

    std::cout << "[YOLO] Model initialized."
              << std::endl;


    // ========================================================
    // 3. 实时采集、推理和可视化
    // ========================================================

    Visualizer visualizer;

    visualizer.initialize(
        "Armor Detection",
        1200,
        900
    );
    
    cv::Mat frame;

    PnP::PnPSolver pnp_solver;

    if (!pnp_solver.load_camera_parameters(
        "../config/camera_calibration.yml"
    ))
    {
        std::cerr << "[PnP ERROR] Failied to load camera calivreation" << std::endl;

        camera.stop_grabbing();
        camera.close();
        camera.finalize();

        return -1;
    }

    pnp_solver.print_camera_parameters();

    const std::vector<cv::Point3f> armor_points = {
        {-65.0F, -65.0F, 0.0F},
        {-65.0F,  65.0F, 0.0F},
        { 65.0F,  65.0F, 0.0F},
        { 65.0F, -65.0F, 0.0F}
    };

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
            if (visualizer.wait_key(1) == 27)
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
            std::vector<cv::Point2f> image_points;

            image_points.reserve(det.keypoints.size());

            for (const auto& keypoint : det.keypoints)
            {
                image_points.emplace_back(
                    keypoint.x,
                    keypoint.y
                );
            }

            visualizer.draw_keypoints(
                frame,
                image_points
            );

            if (image_points.size() != 4)
            {
                continue;
            }

            visualizer.draw_quadrilateral(
                frame,
                image_points
            );

            PnP::PnPResult pnp_result;

            if (!pnp_solver.solve(image_points, pnp_result))
            {
                continue;
            }

            PnP::PnPSolver::print_result(pnp_result);

            std::vector<cv::Point2f> projected_armor_points;

            if (pnp_solver.project_points(
                pnp_result,
                armor_points,
                projected_armor_points
            ))
            {
                visualizer.draw_projected_rectangle(
                    frame,
                    projected_armor_points
                );
            }
        }

        // ----------------------------------------------------
        // 3.4 显示实时画面
        // ----------------------------------------------------

        int key = visualizer.show(frame);

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }

    visualizer.close();

    std::cout << "[System] Program exited."
              << std::endl;

    return 0;
}
