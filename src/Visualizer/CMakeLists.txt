# 创建 Visualizer 静态库。
add_library(Visualizer STATIC
    src/Visualizer.cpp
)

# 公开头文件目录。
# 其他链接 Visualizer 的模块可以直接 #include "Visualizer.h"。
target_include_directories(Visualizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Visualizer 对外接口使用 cv::Mat，因此 OpenCV 是公开依赖。
target_link_libraries(Visualizer
    PUBLIC
        ${OpenCV_LIBS}
)

# 使用 C++17。
target_compile_features(Visualizer
    PUBLIC
        cxx_std_17
)