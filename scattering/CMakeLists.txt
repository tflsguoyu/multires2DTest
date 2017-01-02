cmake_minimum_required(VERSION 2.8.12)

project(scatter_linux)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/home/tflsguoyu/Documents/include/")

add_executable(scatter_linux scattering.cpp)
target_link_libraries(scatter_linux ${OpenCV_LIBS})