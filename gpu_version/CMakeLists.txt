# Specify the minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name and specify that CUDA is required
project(OpenCVExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

# Find the OpenCV package with CUDA support
set(OpenCV_DIR "/nfs/ug/homes-5/l/like23/ECE_1747/opencv/build")

# To supress the warning about CMP0146
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
# Set the CMP0104 policy to NEW
# if(POLICY CMP0104)
#     cmake_policy(SET CMP0104 NEW)
# endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudafilters cudaimgproc)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable and specify the source files
add_executable(OpenCVExample main.cpp cuda_watershed.cu)

# Set the properties for the CUDA source file
set_source_files_properties(cuda_watershed.cu PROPERTIES LANGUAGE CUDA)

# Link the OpenCV libraries
target_link_libraries(OpenCVExample ${OpenCV_LIBS})

# Enable C++11 (or higher) standard
set_property(TARGET OpenCVExample PROPERTY CXX_STANDARD 11)
set_property(TARGET OpenCVExample PROPERTY CUDA_STANDARD 11)

# Optionally, set the CUDA architecture (adjust according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 86)
