project(hello_gpio)
cmake_minimum_required(VERSION 3.6)

set(
    CMAKE_C_COMPILER
    /home/saul/code/RV06_03_Linux_SDK/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc
)

add_executable(
    hello_gpio
    ./src/main.c
)

