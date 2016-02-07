cmake_minimum_required(VERSION 2.0)
PROJECT( EyeDetect_Project )
FIND_PACKAGE( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(src)
