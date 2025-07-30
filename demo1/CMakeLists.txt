cmake_minimum_required(VERSION 3.6)
project(demo1)

# Set the gcc and g++ compilers.
# set(
#     CMAKE_C_COMPILER
#     /home/saul/RockChip/RV06_03_Linux_SDK-main/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc
# )
set(
    CMAKE_CXX_COMPILER
    /home/saul/RockChip/RV06_03_Linux_SDK-main/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++
)


# set(
#     GLOB
#     PROJECT_SOURCES
#     "*.cc"
#     "*.h"
# )
file(GLOB PROJECT_SOURCES "*.cc" "*.h")
add_executable(
    demo1
    ${PROJECT_SOURCES}
)

# Set the 
set(
    RKNN_API_PATH
    /home/saul/code/rknn_c/rknpu2/runtime/Linux/librknn_api
)
set(
    RKNN_RT_LIB
    ${RKNN_API_PATH}//armhf-uclibc/librknnmrt.so
)

# include rknn lib file, e.g., rknn_api.h
include_directories(${RKNN_API_PATH}/include)
include_directories(
    /home/saul/code/rknn_c/rknpu2/examples/3rdparty
)



# target the lib rknn mnt .so file
target_link_libraries(demo1
  ${RKNN_RT_LIB}
)