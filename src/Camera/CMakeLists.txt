
# 创建 Camera 动态库
add_library(Camera SHARED
    src/Camera.cpp
)

# 设置公开头文件搜索路径
target_include_directories(Camera
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Camera 的公开接口使用 cv::Mat，因此 OpenCV 是公开依赖
target_link_libraries(Camera
    PUBLIC
        ${OpenCV_LIBS}
)

# MVS SDK 头文件只在 Camera.cpp 中使用
target_include_directories(Camera
    PRIVATE
        /opt/MVS/include
)

# 链接 MVS 动态库
target_link_libraries(Camera
    PRIVATE
        /opt/MVS_Runtime/MVS/lib/64/libMvCameraControl.so
)

# 让构建出来的动态库在运行时优先查找自身目录
set_target_properties(Camera PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
