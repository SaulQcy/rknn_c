project(cv_resize)
cmake_minimum_required(VERSION 3.6)
set(
    RV1106_COMPILER_PATH
    /home/saul/code/RV06_03_Linux_SDK/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
)
set(
    RV1106_SYSROOT
    /home/saul/code/RV06_03_Linux_SDK/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/arm-rockchip830-linux-uclibcgnueabihf/sysroot
)
set(
    RGA_PATH
    /home/saul/code/librga
)

set(
    CMAKE_C_COMPILER
    ${RV1106_COMPILER_PATH}gcc
)
set(
    CMAKE_CXX_COMPILER
    ${RV1106_COMPILER_PATH}g++
)


file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.cc" "src/*.c" "include/*.h" "include/*.hpp")

set(OpenCV_DIR "/home/saul/code/opencv-mobile-4.10.0-luckfox-pico/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(
    cv_resize
    ${PROJECT_SOURCES}
    ${RGA_PATH}/samples/utils/utils.cpp
    ${RGA_PATH}/samples/utils/allocator/dma_alloc.cpp
)

include_directories(
    ./include
    ${3RD_PARTY_PATH}/stb
    ${RGA_PATH}/include
    ${RGA_PATH}/samples/utils
    ${RGA_PATH}/samples/utils/allocator/include
    ${RV1106_SYSROOT}/usr/include
)

target_link_libraries(
    cv_resize
    # ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so
    ${RGA_PATH}/libs/Linux/gcc-uclib-armhf/librga.so
    ${OpenCV_LIBS}
)

