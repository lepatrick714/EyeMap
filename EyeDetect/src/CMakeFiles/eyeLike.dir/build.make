# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/calvin/gitclones/HackPoly-2016/EyeDetect/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/calvin/gitclones/HackPoly-2016/EyeDetect/src

# Include any dependencies generated for this target.
include CMakeFiles/eyeLike.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/eyeLike.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/eyeLike.dir/flags.make

CMakeFiles/eyeLike.dir/main.o: CMakeFiles/eyeLike.dir/flags.make
CMakeFiles/eyeLike.dir/main.o: main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/eyeLike.dir/main.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eyeLike.dir/main.o -c /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/main.cpp

CMakeFiles/eyeLike.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eyeLike.dir/main.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/main.cpp > CMakeFiles/eyeLike.dir/main.i

CMakeFiles/eyeLike.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eyeLike.dir/main.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/main.cpp -o CMakeFiles/eyeLike.dir/main.s

CMakeFiles/eyeLike.dir/main.o.requires:
.PHONY : CMakeFiles/eyeLike.dir/main.o.requires

CMakeFiles/eyeLike.dir/main.o.provides: CMakeFiles/eyeLike.dir/main.o.requires
	$(MAKE) -f CMakeFiles/eyeLike.dir/build.make CMakeFiles/eyeLike.dir/main.o.provides.build
.PHONY : CMakeFiles/eyeLike.dir/main.o.provides

CMakeFiles/eyeLike.dir/main.o.provides.build: CMakeFiles/eyeLike.dir/main.o

CMakeFiles/eyeLike.dir/findEyeCenter.o: CMakeFiles/eyeLike.dir/flags.make
CMakeFiles/eyeLike.dir/findEyeCenter.o: findEyeCenter.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/eyeLike.dir/findEyeCenter.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eyeLike.dir/findEyeCenter.o -c /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/findEyeCenter.cpp

CMakeFiles/eyeLike.dir/findEyeCenter.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eyeLike.dir/findEyeCenter.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/findEyeCenter.cpp > CMakeFiles/eyeLike.dir/findEyeCenter.i

CMakeFiles/eyeLike.dir/findEyeCenter.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eyeLike.dir/findEyeCenter.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/findEyeCenter.cpp -o CMakeFiles/eyeLike.dir/findEyeCenter.s

CMakeFiles/eyeLike.dir/findEyeCenter.o.requires:
.PHONY : CMakeFiles/eyeLike.dir/findEyeCenter.o.requires

CMakeFiles/eyeLike.dir/findEyeCenter.o.provides: CMakeFiles/eyeLike.dir/findEyeCenter.o.requires
	$(MAKE) -f CMakeFiles/eyeLike.dir/build.make CMakeFiles/eyeLike.dir/findEyeCenter.o.provides.build
.PHONY : CMakeFiles/eyeLike.dir/findEyeCenter.o.provides

CMakeFiles/eyeLike.dir/findEyeCenter.o.provides.build: CMakeFiles/eyeLike.dir/findEyeCenter.o

CMakeFiles/eyeLike.dir/findEyeCorner.o: CMakeFiles/eyeLike.dir/flags.make
CMakeFiles/eyeLike.dir/findEyeCorner.o: findEyeCorner.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/eyeLike.dir/findEyeCorner.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eyeLike.dir/findEyeCorner.o -c /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/findEyeCorner.cpp

CMakeFiles/eyeLike.dir/findEyeCorner.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eyeLike.dir/findEyeCorner.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/findEyeCorner.cpp > CMakeFiles/eyeLike.dir/findEyeCorner.i

CMakeFiles/eyeLike.dir/findEyeCorner.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eyeLike.dir/findEyeCorner.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/findEyeCorner.cpp -o CMakeFiles/eyeLike.dir/findEyeCorner.s

CMakeFiles/eyeLike.dir/findEyeCorner.o.requires:
.PHONY : CMakeFiles/eyeLike.dir/findEyeCorner.o.requires

CMakeFiles/eyeLike.dir/findEyeCorner.o.provides: CMakeFiles/eyeLike.dir/findEyeCorner.o.requires
	$(MAKE) -f CMakeFiles/eyeLike.dir/build.make CMakeFiles/eyeLike.dir/findEyeCorner.o.provides.build
.PHONY : CMakeFiles/eyeLike.dir/findEyeCorner.o.provides

CMakeFiles/eyeLike.dir/findEyeCorner.o.provides.build: CMakeFiles/eyeLike.dir/findEyeCorner.o

CMakeFiles/eyeLike.dir/helpers.o: CMakeFiles/eyeLike.dir/flags.make
CMakeFiles/eyeLike.dir/helpers.o: helpers.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/eyeLike.dir/helpers.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/eyeLike.dir/helpers.o -c /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/helpers.cpp

CMakeFiles/eyeLike.dir/helpers.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eyeLike.dir/helpers.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/helpers.cpp > CMakeFiles/eyeLike.dir/helpers.i

CMakeFiles/eyeLike.dir/helpers.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eyeLike.dir/helpers.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/helpers.cpp -o CMakeFiles/eyeLike.dir/helpers.s

CMakeFiles/eyeLike.dir/helpers.o.requires:
.PHONY : CMakeFiles/eyeLike.dir/helpers.o.requires

CMakeFiles/eyeLike.dir/helpers.o.provides: CMakeFiles/eyeLike.dir/helpers.o.requires
	$(MAKE) -f CMakeFiles/eyeLike.dir/build.make CMakeFiles/eyeLike.dir/helpers.o.provides.build
.PHONY : CMakeFiles/eyeLike.dir/helpers.o.provides

CMakeFiles/eyeLike.dir/helpers.o.provides.build: CMakeFiles/eyeLike.dir/helpers.o

# Object files for target eyeLike
eyeLike_OBJECTS = \
"CMakeFiles/eyeLike.dir/main.o" \
"CMakeFiles/eyeLike.dir/findEyeCenter.o" \
"CMakeFiles/eyeLike.dir/findEyeCorner.o" \
"CMakeFiles/eyeLike.dir/helpers.o"

# External object files for target eyeLike
eyeLike_EXTERNAL_OBJECTS =

eyeLike: CMakeFiles/eyeLike.dir/main.o
eyeLike: CMakeFiles/eyeLike.dir/findEyeCenter.o
eyeLike: CMakeFiles/eyeLike.dir/findEyeCorner.o
eyeLike: CMakeFiles/eyeLike.dir/helpers.o
eyeLike: CMakeFiles/eyeLike.dir/build.make
eyeLike: //lib/libopencv_videostab.so.3.0.0
eyeLike: //lib/libopencv_videoio.so.3.0.0
eyeLike: //lib/libopencv_video.so.3.0.0
eyeLike: //lib/libopencv_superres.so.3.0.0
eyeLike: //lib/libopencv_stitching.so.3.0.0
eyeLike: //lib/libopencv_shape.so.3.0.0
eyeLike: //lib/libopencv_photo.so.3.0.0
eyeLike: //lib/libopencv_objdetect.so.3.0.0
eyeLike: //lib/libopencv_ml.so.3.0.0
eyeLike: //lib/libopencv_imgproc.so.3.0.0
eyeLike: //lib/libopencv_imgcodecs.so.3.0.0
eyeLike: //lib/libopencv_highgui.so.3.0.0
eyeLike: //lib/libopencv_hal.a
eyeLike: //lib/libopencv_flann.so.3.0.0
eyeLike: //lib/libopencv_features2d.so.3.0.0
eyeLike: //lib/libopencv_core.so.3.0.0
eyeLike: //lib/libopencv_calib3d.so.3.0.0
eyeLike: //lib/libopencv_features2d.so.3.0.0
eyeLike: //lib/libopencv_ml.so.3.0.0
eyeLike: //lib/libopencv_highgui.so.3.0.0
eyeLike: //lib/libopencv_videoio.so.3.0.0
eyeLike: //lib/libopencv_imgcodecs.so.3.0.0
eyeLike: //lib/libopencv_flann.so.3.0.0
eyeLike: //lib/libopencv_video.so.3.0.0
eyeLike: //lib/libopencv_imgproc.so.3.0.0
eyeLike: //lib/libopencv_core.so.3.0.0
eyeLike: //lib/libopencv_hal.a
eyeLike: /share/OpenCV/3rdparty/lib/libippicv.a
eyeLike: CMakeFiles/eyeLike.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable eyeLike"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/eyeLike.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/eyeLike.dir/build: eyeLike
.PHONY : CMakeFiles/eyeLike.dir/build

CMakeFiles/eyeLike.dir/requires: CMakeFiles/eyeLike.dir/main.o.requires
CMakeFiles/eyeLike.dir/requires: CMakeFiles/eyeLike.dir/findEyeCenter.o.requires
CMakeFiles/eyeLike.dir/requires: CMakeFiles/eyeLike.dir/findEyeCorner.o.requires
CMakeFiles/eyeLike.dir/requires: CMakeFiles/eyeLike.dir/helpers.o.requires
.PHONY : CMakeFiles/eyeLike.dir/requires

CMakeFiles/eyeLike.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/eyeLike.dir/cmake_clean.cmake
.PHONY : CMakeFiles/eyeLike.dir/clean

CMakeFiles/eyeLike.dir/depend:
	cd /home/calvin/gitclones/HackPoly-2016/EyeDetect/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/calvin/gitclones/HackPoly-2016/EyeDetect/src /home/calvin/gitclones/HackPoly-2016/EyeDetect/src /home/calvin/gitclones/HackPoly-2016/EyeDetect/src /home/calvin/gitclones/HackPoly-2016/EyeDetect/src /home/calvin/gitclones/HackPoly-2016/EyeDetect/src/CMakeFiles/eyeLike.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/eyeLike.dir/depend

