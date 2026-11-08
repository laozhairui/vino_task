#include "Visualizer.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>


Visualizer::~Visualizer()
{
    close();
}


void Visualizer::initialize(
    const std::string& window_name,
    int width,
    int height
)
{
    // 如果之前已经初始化过窗口，先关闭旧窗口。
    close();

    m_window_name = window_name;

    cv::namedWindow(
        m_window_name,
        cv::WINDOW_NORMAL
    );

    cv::resizeWindow(
        m_window_name,
        width,
        height
    );

    m_initialized = true;
}


void Visualizer::draw_keypoints(
    cv::Mat& image,
    const std::vector<cv::Point2f>& points
) const
{
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto& point = points[i];

        const int x = cvRound(point.x);
        const int y = cvRound(point.y);

        // 忽略图像范围以外的关键点。
        if (x < 0 || x >= image.cols
            || y < 0 || y >= image.rows)
        {
            continue;
        }

        // 绘制红色关键点。
        cv::circle(
            image,
            cv::Point(x, y),
            4,
            cv::Scalar(0, 0, 255),
            cv::FILLED
        );

        // 绘制关键点序号。
        cv::putText(
            image,
            std::to_string(i),
            cv::Point(x + 5, y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(255, 0, 0),
            1
        );
    }
}


void Visualizer::draw_quadrilateral(
    cv::Mat& image,
    const std::vector<cv::Point2f>& points
) const
{
    // 当前四边形只适用于四个关键点。
    if (points.size() != 4)
    {
        return;
    }

    std::vector<cv::Point> corners;

    corners.reserve(points.size());

    for (const auto& point : points)
    {
        corners.emplace_back(
            cvRound(point.x),
            cvRound(point.y)
        );
    }

    // 按照关键点给定的顺序连接四个顶点。
    cv::line(image, corners[0], corners[1], cv::Scalar(0, 255, 0), 2);
    cv::line(image, corners[1], corners[2], cv::Scalar(0, 255, 0), 2);
    cv::line(image, corners[2], corners[3], cv::Scalar(0, 255, 0), 2);
    cv::line(image, corners[3], corners[0], cv::Scalar(0, 255, 0), 2);
}


void Visualizer::draw_projected_rectangle(
    cv::Mat& image,
    const std::vector<cv::Point2f>& points
) const
{
    if (points.size() != 4)
    {
        return;
    }

    std::vector<cv::Point> corners;

    corners.reserve(points.size());

    for (const auto& point : points)
    {
        corners.emplace_back(
            cvRound(point.x),
            cvRound(point.y)
        );
    }

    // 绘制闭合的青色投影矩形。
    const std::vector<std::vector<cv::Point>> contours = {
        corners
    };

    cv::polylines(
        image,
        contours,
        true,
        cv::Scalar(255, 255, 0),
        2
    );
}


int Visualizer::show(
    const cv::Mat& image,
    int delay
) const
{
    if (!m_initialized)
    {
        return -1;
    }

    cv::imshow(m_window_name, image);

    return cv::waitKey(delay);
}


int Visualizer::wait_key(int delay) const
{
    return cv::waitKey(delay);
}


void Visualizer::close()
{
    if (m_initialized)
    {
        cv::destroyWindow(m_window_name);
        m_initialized = false;
    }
}