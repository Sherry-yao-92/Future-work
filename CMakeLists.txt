cmake_minimum_required(VERSION 3.8)
project(findcontour_time_10000)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 包含 OpenCV 的頭文件目錄
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可執行文件
add_executable(findcontour_time_10000 findcontour_time_10000.cpp)

# 鏈接 OpenCV 庫
target_link_libraries(findcontour_time_10000 PRIVATE ${OpenCV_LIBS})

# 查找 TBB 包
find_package(TBB REQUIRED)
if(TBB_FOUND)
    message(STATUS "Found TBB")
    target_link_libraries(findcontour_time_10000 PRIVATE TBB::tbb)
else()
    message(FATAL_ERROR "TBB not found")
endif()

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    target_link_libraries(findcontour_time_10000 PRIVATE OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found")
endif()
