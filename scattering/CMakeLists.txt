cmake_minimum_required(VERSION 2.8.12)

project(scattering)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(scattering scattering.cpp)
target_link_libraries(scattering ${OpenCV_LIBS})