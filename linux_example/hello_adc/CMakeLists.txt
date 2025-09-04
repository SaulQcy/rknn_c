project(hello_adc)
cmake_minimum_required(VERSION 3.6)

set(CMAKE_C_COMPILER $ENV{RV1106_C_COMPILER})
set(CMAKE_CXX_COMPILER $ENV{RV1106_CXX_COMPILER})

add_executable(
    hello_adc
    src/main.cc
)