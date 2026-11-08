
# 创建 PnP 独立静态库。
add_library(pnp_solver
    src/PnP.cpp
)

# 将当前目录作为公开头文件搜索路径。
# 其他模块链接 pnp_solver 后，就可以直接 #include "PnP.h"。
target_include_directories(pnp_solver
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# PnP 模块依赖 OpenCV。
target_include_directories(pnp_solver
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(pnp_solver
    PUBLIC
        ${OpenCV_LIBS}
)

# 使用 C++17，支持当前项目使用的标准库功能。
target_compile_features(pnp_solver
    PUBLIC
        cxx_std_17
)
