project(mbnet_cv_singal)
cmake_minimum_required(VERSION 3.6)

set(
    RKNN_API_PATH
    /home/saul/code/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
)
set(
    OpenCV_DIR 
    /home/saul/code/opencv-mobile-4.10.0-luckfox-pico/lib/cmake/opencv4
)
set(
    3RD_PARTY_PATH
    /home/saul/code/rknn-toolkit2/rknpu2/examples/3rdparty
)

set(CHIP "RV1106" CACHE STRING "Target chip: RV1106 or RK3588")

message(STATUS "Target CHIP = ${CHIP}")

if(CHIP STREQUAL "RV1106")
    message(STATUS "Configure for RV1106")
    # Setting the compilers
    set(
        CMAKE_C_COMPILER
        /home/saul/code/RV06_03_Linux_SDK/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc
    )
    set(
        CMAKE_CXX_COMPILER
        /home/saul/code/RV06_03_Linux_SDK/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++
    )

    # setting include
    include_directories(/home/saul/code/RV06_03_Linux_SDK/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/arm-rockchip830-linux-uclibcgnueabihf/sysroot/usr/include)
elseif(CHIP STREQUAL "RK3588")
    message(FATAL_ERROR "Unsupported CHIP: ${CHIP}")
else()
    message(FATAL_ERROR "Unsupported CHIP: ${CHIP}")
endif()


file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.cc" "src/*.c" "include/*.h" "include/*.hpp")

find_package(OpenCV REQUIRED)

add_executable(
    mbnet_cv_singal
    ${PROJECT_SOURCES}
)

#include / link
include_directories(
    ${RKNN_API_PATH}/include
    ./include
    ${OpenCV_INCLUDE_DIRS}
    ${3RD_PARTY_PATH}/stb
)

target_link_libraries(
    mbnet_cv_singal
    ${OpenCV_LIBS}
    ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so
)


