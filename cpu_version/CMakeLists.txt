# Specify the minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(cpu_watershed)

# Find the OpenCV package
set(OpenCV_DIR "/nfs/ug/homes-5/l/like23/ECE_1747/opencv/build")
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(cpu_watershed main.cpp)

# Link the OpenCV libraries
target_link_libraries(cpu_watershed ${OpenCV_LIBS})
