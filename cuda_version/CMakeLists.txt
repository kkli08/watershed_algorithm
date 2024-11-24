# Specify the minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name and specify that CUDA is required
project(CUDAWatershed LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the path to your OpenCV build directory
set(OpenCV_DIR "/nfs/ug/homes-5/l/like23/ECE_1747/opencv/build") # Update this path accordingly

# Find the OpenCV package with CUDA support
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudafilters cudaimgproc)

# Find the CUDA package
find_package(CUDA REQUIRED)

# Include OpenCV and CUDA include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Add the executable and specify the source files
add_executable(CUDAWatershed main.cpp)

# Link the OpenCV and CUDA libraries
target_link_libraries(CUDAWatershed ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# Enable CUDA for the project
enable_language(CUDA)

# Set the CUDA architecture (adjust according to your GPU)
# For example, 86 corresponds to NVIDIA Ampere architecture (e.g., RTX 30 series)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Optional: Set compiler flags for CUDA
# This can help with optimization; adjust as needed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
