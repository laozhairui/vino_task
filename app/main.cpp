
#include "Camera.h"
#include "yolo_vino.hpp"
#include "PnP.h"

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

    cv::namedWindow("Armor Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Armor Detection", 1200, 900);

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

            // 绘制输出的关键点。
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

                if (det.keypoints.size() == 4)
                {

                    std::vector<cv::Point2f> image_points = {
                        cv::Point2f(det.keypoints[0].x, det.keypoints[0].y),
                        cv::Point2f(det.keypoints[1].x, det.keypoints[1].y),
                        cv::Point2f(det.keypoints[2].x, det.keypoints[2].y),
                        cv::Point2f(det.keypoints[3].x, det.keypoints[3].y)
                        
                        
                    };

                    PnP::PnPResult pnp_result;

                    if (pnp_solver.solve(image_points, pnp_result))
                    {
                        PnP::PnPSolver::print_result(pnp_result);

                        const std::vector<cv::Point3f> armor_points = {
                            {-65.0F, -65.0F, 0.0F},
                            {-65.0F,  65.0F, 0.0F},
                            { 65.0F,  65.0F, 0.0F},
                            { 65.0F, -65.0F, 0.0F}
                        };

                        std::vector<cv::Point2f>projected_armor_points;

                        if (pnp_solver.project_points(
                            pnp_result,
                            armor_points,
                            projected_armor_points
                        ))
                        {
                            std::vector<cv::Point> armor_corners;
                            
                            for (const auto& point : projected_armor_points)
                            {
                                armor_corners.emplace_back(
                                    cvRound(point.x),
                                    cvRound(point.y)
                                );
                            }

                            cv::polylines(
                                frame,
                                armor_corners,
                                true,
                                cv::Scalar(255, 255, 0),
                                2
                            );
                        }

                    }

                    cv::Point p0(
                        static_cast<int>(det.keypoints[0].x),
                        static_cast<int>(det.keypoints[0].y)
                    );

                    cv::Point p1(
                        static_cast<int>(det.keypoints[1].x),
                        static_cast<int>(det.keypoints[1].y)
                    );

                    cv::Point p2(
                        static_cast<int>(det.keypoints[2].x),
                        static_cast<int>(det.keypoints[2].y)
                    );

                    cv::Point p3(
                        static_cast<int>(det.keypoints[3].x),
                        static_cast<int>(det.keypoints[3].y)
                    );

                    cv::line(frame, p0, p1, cv::Scalar(0, 255, 0), 2);
                    cv::line(frame, p1, p2, cv::Scalar(0, 255, 0), 2);
                    cv::line(frame, p2, p3, cv::Scalar(0, 255, 0), 2);
                    cv::line(frame, p3, p0, cv::Scalar(0, 255, 0), 2);
                    
                }
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
    
    cv::destroyAllWindows();

    std::cout << "[System] Program exited."
              << std::endl;

    return 0;
}
