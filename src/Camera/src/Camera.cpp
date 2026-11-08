#include "Camera.h"

#include <MvCameraControl.h>

#include <opencv2/imgproc.hpp>

#include <memory>
#include <string>


// ============================================================
// 1. Impl：Camera 的内部实现类
// ============================================================

class Camera::Impl
{
public:
    Impl()
        : m_device_list{}
        , m_handle(nullptr)
        , m_sdk_acquired(false)
        , m_device_opened(false)
        , m_grabbing(false)
        , m_last_error()
    {
    }

    ~Impl()
    {
        // 对象销毁时，尽可能释放相机和 SDK 资源。
        finalize();
    }

    // ========================================================
    // 1.1 一键启动相机
    // ========================================================

    bool start(int device_id)
    {
        m_last_error.clear();

        // 已经开始采集时，不重复初始化。
        if (m_grabbing)
        {
            return true;
        }

        // 不允许在存在未完成的设备状态时重复启动。
        if (m_handle != nullptr || m_device_opened)
        {
            m_last_error =
                "Camera is partially initialized. Release it first.";

            return false;
        }

        // 初始化 SDK。
        if (!initialize())
        {
            return false;
        }

        // 枚举设备。
        if (!enumerate_devices())
        {
            return rollback_start();
        }

        // 检查设备索引。
        if (device_id < 0 || device_id >= device_count())
        {
            m_last_error = "Invalid camera device ID.";
            return rollback_start();
        }

        // 创建相机句柄。
        if (!create_handle(device_id))
        {
            return rollback_start();
        }

        // 打开相机，并设置相机参数。
        if (!open())
        {
            return rollback_start();
        }

        // 开始图像采集。
        if (!start_grabbing())
        {
            return rollback_start();
        }

        return true;
    }

    // ========================================================
    // 1.2 初始化和释放 SDK
    // ========================================================

    bool initialize()
    {
        if (m_sdk_acquired)
        {
            return true;
        }

        m_last_error.clear();

        // 只有第一个 Camera 实例需要初始化 MVS SDK。
        if (m_sdk_reference_count == 0)
        {
            int ret = MV_CC_Initialize();

            if (ret != MV_OK)
            {
                m_last_error = "Failed to initialize MVS SDK.";
                return false;
            }
        }

        ++m_sdk_reference_count;
        m_sdk_acquired = true;

        return true;
    }

    bool finalize()
    {
        if (!m_sdk_acquired)
        {
            return true;
        }

        // 尝试释放设备资源。
        bool close_success = close();

        // 如果句柄仍然存在，说明设备资源没有完全释放。
        // 此时不能结束 SDK。
        if (m_handle != nullptr)
        {
            return false;
        }

        if (m_sdk_reference_count <= 0)
        {
            m_last_error = "Invalid SDK reference count.";
            return false;
        }

        // 只有最后一个 Camera 实例才真正结束 SDK。
        if (m_sdk_reference_count == 1)
        {
            int ret = MV_CC_Finalize();

            if (ret != MV_OK)
            {
                m_last_error = "Failed to finalize MVS SDK.";
                return false;
            }
        }

        --m_sdk_reference_count;
        m_sdk_acquired = false;

        // 即使设备资源最终释放成功，
        // 如果之前的停止采集操作失败，仍然向调用者报告失败。
        return close_success;
    }

    // ========================================================
    // 1.3 设备枚举和句柄创建
    // ========================================================

    bool enumerate_devices()
    {
        m_last_error.clear();
        m_device_list = {};

        if (!m_sdk_acquired)
        {
            m_last_error = "Initialize the MVS SDK first.";
            return false;
        }

        int ret = MV_CC_EnumDevices(
            MV_GIGE_DEVICE | MV_USB_DEVICE,
            &m_device_list
        );

        if (ret != MV_OK)
        {
            m_last_error = "Failed to enumerate camera devices.";
            return false;
        }

        if (m_device_list.nDeviceNum == 0)
        {
            m_last_error = "No camera device found.";
            return false;
        }

        return true;
    }

    int device_count() const
    {
        return static_cast<int>(m_device_list.nDeviceNum);
    }

    bool create_handle(int device_id)
    {
        m_last_error.clear();

        if (!m_sdk_acquired)
        {
            m_last_error = "Initialize the MVS SDK first.";
            return false;
        }

        if (m_handle != nullptr)
        {
            m_last_error = "Camera handle already exists.";
            return false;
        }

        if (device_id < 0
            || device_id >= static_cast<int>(m_device_list.nDeviceNum))
        {
            m_last_error = "Invalid camera device ID.";
            return false;
        }

        MV_CC_DEVICE_INFO* device_info =
            m_device_list.pDeviceInfo[device_id];

        if (device_info == nullptr)
        {
            m_last_error = "Camera device information is null.";
            return false;
        }

        int ret = MV_CC_CreateHandle(&m_handle, device_info);

        if (ret != MV_OK)
        {
            m_handle = nullptr;
            m_last_error = "Failed to create camera handle.";
            return false;
        }

        return true;
    }

    // ========================================================
    // 1.4 打开相机和设置参数
    // ========================================================

    bool open()
    {
        m_last_error.clear();

        if (!m_sdk_acquired)
        {
            m_last_error = "Initialize the MVS SDK first.";
            return false;
        }

        if (m_handle == nullptr)
        {
            m_last_error = "Create the camera handle before opening.";
            return false;
        }

        if (m_device_opened)
        {
            return true;
        }

        int ret = MV_CC_OpenDevice(m_handle);

        if (ret != MV_OK)
        {
            m_last_error = "Failed to open camera device.";
            return false;
        }

        m_device_opened = true;

        // 打开相机后设置参数。
        if (!set_camera_parameters())
        {
            // 保存原始错误，避免清理操作覆盖错误信息。
            std::string error_message = m_last_error;

            close();

            m_last_error = error_message;
            return false;
        }

        return true;
    }

    bool set_camera_parameters()
    {
        m_last_error.clear();

        if (!m_device_opened)
        {
            m_last_error = "Open camera before setting parameters.";
            return false;
        }

        // 关闭自动曝光。
        int ret = MV_CC_SetEnumValue(
            m_handle,
            "ExposureAuto",
            0
        );

        if (ret != MV_OK)
        {
            m_last_error = "Failed to disable auto exposure.";
            return false;
        }

        // 设置曝光时间：20000 微秒。
        ret = MV_CC_SetFloatValue(
            m_handle,
            "ExposureTime",
            20000.0F
        );

        if (ret != MV_OK)
        {
            m_last_error = "Failed to set exposure time.";
            return false;
        }

        // 设置增益：15 dB。
        ret = MV_CC_SetFloatValue(
            m_handle,
            "Gain",
            15.0F
        );

        if (ret != MV_OK)
        {
            m_last_error = "Failed to set gain.";
            return false;
        }

        return true;
    }

    // ========================================================
    // 1.5 图像采集
    // ========================================================

    bool start_grabbing()
    {
        m_last_error.clear();

        if (!m_device_opened)
        {
            m_last_error =
                "Open the camera before starting acquisition.";

            return false;
        }

        if (m_grabbing)
        {
            return true;
        }

        int ret = MV_CC_StartGrabbing(m_handle);

        if (ret != MV_OK)
        {
            m_last_error = "Failed to start image acquisition.";
            return false;
        }

        m_grabbing = true;

        return true;
    }

    bool get_frame(cv::Mat& image)
    {
        m_last_error.clear();

        if (!m_grabbing)
        {
            m_last_error =
                "Start image acquisition before getting frames.";

            return false;
        }

        MV_FRAME_OUT frame_info{};

        int ret = MV_CC_GetImageBuffer(
            m_handle,
            &frame_info,
            1000
        );

        if (ret != MV_OK)
        {
            m_last_error = "Failed to get image buffer.";
            return false;
        }

        // 转换图像。
        bool convert_success = false;

        try
        {
            convert_success = convert_to_bgr(frame_info, image);
        }
        catch (const cv::Exception&)
        {
            // 即使 OpenCV 转换抛出异常，也继续释放 MVS 图像缓冲区。
            convert_success = false;
        }

        // MVS 图像缓冲区必须归还给 SDK。
        int free_ret = MV_CC_FreeImageBuffer(
            m_handle,
            &frame_info
        );

        if (!convert_success)
        {
            image.release();
            m_last_error = "Failed to convert image to BGR format.";
            return false;
        }

        if (free_ret != MV_OK)
        {
            image.release();
            m_last_error = "Failed to release image buffer.";
            return false;
        }

        return true;
    }

    bool convert_to_bgr(
        MV_FRAME_OUT& frame_info,
        cv::Mat& image
    )
    {
        if (frame_info.pBufAddr == nullptr)
        {
            return false;
        }

        int width = static_cast<int>(
            frame_info.stFrameInfo.nWidth
        );

        int height = static_cast<int>(
            frame_info.stFrameInfo.nHeight
        );

        if (width <= 0 || height <= 0)
        {
            return false;
        }

        // raw_image 只引用 MVS 提供的原始图像缓冲区，
        // 不拥有这块内存，因此不能在释放 MVS 缓冲区后继续使用。
        cv::Mat raw_image(
            height,
            width,
            CV_8UC1,
            frame_info.pBufAddr
        );

        // Bayer 转 BGR。
        cv::cvtColor(
            raw_image,
            image,
            cv::COLOR_BayerBG2BGR
        );

        return !image.empty();
    }

    // ========================================================
    // 1.6 停止采集和关闭设备
    // ========================================================

    bool stop_grabbing()
    {
        if (!m_grabbing)
        {
            return true;
        }

        m_last_error.clear();

        int ret = MV_CC_StopGrabbing(m_handle);

        if (ret != MV_OK)
        {
            m_last_error = "Failed to stop image acquisition.";
            return false;
        }

        m_grabbing = false;

        return true;
    }

    bool close()
    {
        m_last_error.clear();

        if (m_handle == nullptr)
        {
            m_device_opened = false;
            m_grabbing = false;
            return true;
        }

        bool success = true;
        std::string first_error;

        // 第一步：尝试停止采集。
        if (m_grabbing)
        {
            if (!stop_grabbing())
            {
                success = false;
                first_error = m_last_error;
            }
        }

        // 第二步：尝试关闭设备。
        if (m_device_opened)
        {
            int ret = MV_CC_CloseDevice(m_handle);

            if (ret != MV_OK)
            {
                // 关闭设备失败时，不能销毁仍然关联设备的句柄。
                if (success)
                {
                    first_error = "Failed to close camera device.";
                }

                m_last_error = first_error;
                return false;
            }

            m_device_opened = false;
            m_grabbing = false;
        }

        // 第三步：销毁句柄。
        int ret = MV_CC_DestroyHandle(m_handle);

        if (ret != MV_OK)
        {
            if (success)
            {
                first_error = "Failed to destroy camera handle.";
            }

            m_last_error = first_error;
            return false;
        }

        m_handle = nullptr;
        m_device_opened = false;
        m_grabbing = false;

        if (!success)
        {
            m_last_error = first_error;
            return false;
        }

        return true;
    }

    // ========================================================
    // 1.7 状态和错误信息
    // ========================================================

    bool is_opened() const
    {
        return m_device_opened;
    }

    std::string last_error() const
    {
        return m_last_error;
    }

private:
    // ========================================================
    // 1.8 启动失败后的回滚
    // ========================================================

    bool rollback_start()
    {
        // 保存真正导致启动失败的错误。
        std::string original_error = m_last_error;

        // 尝试释放设备和 SDK 资源。
        finalize();

        // 清理过程中可能产生其他错误，
        // 这里恢复最初的启动失败原因。
        m_last_error = original_error;

        return false;
    }

    // ========================================================
    // 1.9 内部成员
    // ========================================================

    MV_CC_DEVICE_INFO_LIST m_device_list{};
    void* m_handle;

    static int m_sdk_reference_count;

    bool m_sdk_acquired;
    bool m_device_opened;
    bool m_grabbing;

    std::string m_last_error;
};


// 静态成员必须在类外定义并初始化。
int Camera::Impl::m_sdk_reference_count = 0;


// ============================================================
// 2. Camera：对外公开的接口
// ============================================================

Camera::Camera()
    : m_impl(std::make_unique<Impl>())
{
}

Camera::~Camera() = default;

bool Camera::start(int device_id)
{
    return m_impl->start(device_id);
}

bool Camera::initialize()
{
    return m_impl->initialize();
}

bool Camera::finalize()
{
    return m_impl->finalize();
}

bool Camera::enumerate_devices()
{
    return m_impl->enumerate_devices();
}

int Camera::device_count() const
{
    return m_impl->device_count();
}

bool Camera::create_handle(int device_id)
{
    return m_impl->create_handle(device_id);
}

bool Camera::open()
{
    return m_impl->open();
}

bool Camera::set_camera_parameters()
{
    return m_impl->set_camera_parameters();
}

bool Camera::start_grabbing()
{
    return m_impl->start_grabbing();
}

bool Camera::get_frame(cv::Mat& image)
{
    return m_impl->get_frame(image);
}

bool Camera::stop_grabbing()
{
    return m_impl->stop_grabbing();
}

bool Camera::close()
{
    return m_impl->close();
}

bool Camera::is_opened() const
{
    return m_impl->is_opened();
}

std::string Camera::last_error() const
{
    return m_impl->last_error();
}