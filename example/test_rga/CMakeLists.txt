cmake_minimum_required(VERSION 3.6)
project(test_rga)

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

set(
        RGA_PATH
        ${3RD_PARTY_PATH}/rga
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
        test_rga
        ${PROJECT_SOURCES}
)

#include / link
include_directories(
        ${RKNN_API_PATH}/include
        ${3RD_PARTY_PATH}/stb
        ./include
        ${RGA_PATH}/include
)


target_link_libraries(
        test_rga
        ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so
        ${RGA_PATH}/libs/Linux/gcc-uclib-armhf/librga.so
)