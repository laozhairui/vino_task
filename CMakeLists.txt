cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(Task8 LANGUAGES CXX)

find_package(OpenCV REQUIRED)


add_subdirectory(src/YoloVino)
add_subdirectory(src/Camera)

add_executable(${PROJECT_NAME} app/main.cpp)
target_link_libraries(${PROJECT_NAME}
 PRIVATE
  YoloVino_LIB
  Camera
  )
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
