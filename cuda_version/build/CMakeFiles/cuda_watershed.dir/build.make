# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/build

# Include any dependencies generated for this target.
include CMakeFiles/cuda_watershed.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cuda_watershed.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_watershed.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_watershed.dir/flags.make

CMakeFiles/cuda_watershed.dir/main.cpp.o: CMakeFiles/cuda_watershed.dir/flags.make
CMakeFiles/cuda_watershed.dir/main.cpp.o: /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/main.cpp
CMakeFiles/cuda_watershed.dir/main.cpp.o: CMakeFiles/cuda_watershed.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cuda_watershed.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cuda_watershed.dir/main.cpp.o -MF CMakeFiles/cuda_watershed.dir/main.cpp.o.d -o CMakeFiles/cuda_watershed.dir/main.cpp.o -c /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/main.cpp

CMakeFiles/cuda_watershed.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuda_watershed.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/main.cpp > CMakeFiles/cuda_watershed.dir/main.cpp.i

CMakeFiles/cuda_watershed.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuda_watershed.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/main.cpp -o CMakeFiles/cuda_watershed.dir/main.cpp.s

# Object files for target cuda_watershed
cuda_watershed_OBJECTS = \
"CMakeFiles/cuda_watershed.dir/main.cpp.o"

# External object files for target cuda_watershed
cuda_watershed_EXTERNAL_OBJECTS =

cuda_watershed: CMakeFiles/cuda_watershed.dir/main.cpp.o
cuda_watershed: CMakeFiles/cuda_watershed.dir/build.make
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_highgui.so.4.10.0
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_cudaimgproc.so.4.10.0
cuda_watershed: /usr/lib/x86_64-linux-gnu/libcudart_static.a
cuda_watershed: /usr/lib/x86_64-linux-gnu/librt.a
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_videoio.so.4.10.0
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_imgcodecs.so.4.10.0
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_cudafilters.so.4.10.0
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_imgproc.so.4.10.0
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_cudaarithm.so.4.10.0
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_core.so.4.10.0
cuda_watershed: /homes/l/like23/ECE_1747/opencv/build/lib/libopencv_cudev.so.4.10.0
cuda_watershed: CMakeFiles/cuda_watershed.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cuda_watershed"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_watershed.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_watershed.dir/build: cuda_watershed
.PHONY : CMakeFiles/cuda_watershed.dir/build

CMakeFiles/cuda_watershed.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_watershed.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_watershed.dir/clean

CMakeFiles/cuda_watershed.dir/depend:
	cd /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/build /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/build /homes/l/like23/ECE_1747/project/watershed_algorithm/cuda_version/build/CMakeFiles/cuda_watershed.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda_watershed.dir/depend

