project(a.out)
cmake_minimum_required(VERSION 3.6)

set(CMAKE_C_COMPILER $ENV{RV1106_C_COMPILER})

include_directories(/home/saul/code/RV1106_rootfs/usr/include)

link_directories(/home/saul/code/RV1106_rootfs/usr/lib)

add_executable(
    a.out
    src/main.c
)

target_link_libraries(
    a.out
    gpiod
)