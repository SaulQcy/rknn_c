cmake_minimum_required(VERSION 3.6)
project(pfld_demo)

# set absolute path, i.e., compilers and RKNN-API
set(
    RV1106_COMPILER_PATH
    /home/saul/code/RV06_03_Linux_SDK/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
)
set(
    RKNN_API_PATH
    /home/saul/code/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
)
set(
    3RD_PARTY_PATH
    /home/saul/code/rknn-toolkit2/rknpu2/examples/3rdparty
)

# set compilers
set(
    CMAKE_C_COMPILER
    ${RV1106_COMPILER_PATH}gcc
)
set(
    CMAKE_CXX_COMPILER
    ${RV1106_COMPILER_PATH}g++
)

# add compiled files
# file(GLOB PROJECT_SOURCES "*.cpp" "*.cc" "*.c" "*.h" "*.hpp")
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.cc" "src/*.c" "include/*.h" "include/*.hpp")

add_executable(
    pfld_demo
    ${PROJECT_SOURCES}
)

#include / link
include_directories(
    ${RKNN_API_PATH}/include
    ${3RD_PARTY_PATH}/stb
    ./include
    ${3RD_PARTY_PATH}/opencv/opencv-linux-armhf/include
)

link_directories(${3RD_PARTY_PATH}/opencv/opencv-linux-armhf/lib)

target_link_libraries(
    pfld_demo
    ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so
    ${OpenCV_LIBS}
)
