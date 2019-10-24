cmake_minimum_required(VERSION 3.15)
project(TemporalRvl)

include_directories("${PROJECT_SOURCE_DIR}/src")

find_path(SYSTEM_INCLUDE_DIR zlib.h)
include_directories(${SYSTEM_INCLUDE_DIR})

find_package(OpenCV REQUIRED)

add_subdirectory(src)