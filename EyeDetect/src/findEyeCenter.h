#ifndef EYE_CENTER_H
#define EYE_CENTER_H

#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow);
std::vector < cv::Point > findEyeContours(cv::Mat face, cv::Rect Eye, std::string contourWin);

#endif
