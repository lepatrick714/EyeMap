#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "helpers.h"

#include "findEyeCorner.h"

cv::Mat *leftCornerKernel;
cv::Mat *rightCornerKernel;

// not constant because stupid opencv type signatures
float kEyeCornerKernel[4][6] = {
  {-1,-1,-1, 1, 1, 1},
  {-1,-1,-1,-1, 1, 1},
  {-1,-1,-1,-1, 0, 3},
  { 1, 1, 1, 1, 1, 1},
};

void createCornerKernels() {
  rightCornerKernel = new cv::Mat(4,6,CV_32F,kEyeCornerKernel);
  leftCornerKernel = new cv::Mat(4,6,CV_32F);
  // flip horizontally
  cv::flip(*rightCornerKernel, *leftCornerKernel, 1);
}

void releaseCornerKernels() {
  delete leftCornerKernel;
  delete rightCornerKernel;
}

// TODO implement these
cv::Mat eyeCornerMap(const cv::Mat &region, bool left, bool left2) {
  cv::Mat cornerMap;

  cv::Size sizeRegion = region.size();
  cv::Range colRange(sizeRegion.width / 4, sizeRegion.width * 3 / 4);
  cv::Range rowRange(sizeRegion.height / 4, sizeRegion.height * 3 / 4);

  cv::Mat miRegion(region, rowRange, colRange);

  cv::filter2D(miRegion, cornerMap, CV_32F,
               (left && !left2) || (!left && !left2) ? *leftCornerKernel : *rightCornerKernel);

  return cornerMap;
}

cv::Point2f findEyeCorner(cv::Mat region, bool left, bool left2) {
  // cv::Mat cornerMap = eyeCornerMap(region, left, left2);
  cv::namedWindow("eyeCorner",CV_WINDOW_NORMAL);
  imshow("eyeCorner",region);

//  cv::Point maxP;
//  cv::minMaxLoc(cornerMap,NULL,NULL,NULL,&maxP);

  cv::Point2f maxP2;

  maxP2.x = region.cols / 2;
  maxP2.y = region.rows / 2;
   //maxP2 = findSubpixelEyeCorner(cornerMap, maxP);
   // GFTT
//  std::vector<cv::Point2f> corners;
//  cv::goodFeaturesToTrack(region, corners, 500, 0.005, 20);
//  for (int i = 0; i < corners.size(); ++i) {
//    cv::circle(region, corners[i], 2, 200);
//  }
//  imshow("Corners",region);

  return maxP2;
}
