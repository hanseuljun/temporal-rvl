cmake_minimum_required(VERSION 3.15)
project(TemporalRvl)

include_directories("${PROJECT_SOURCE_DIR}/src")

find_path(VCPKG_INCLUDE_DIR NAMES zlib.h PATH_SUFFIXES zlib)
include_directories(${VCPKG_INCLUDE_DIR})

find_package(OpenCV REQUIRED)

add_subdirectory(src)