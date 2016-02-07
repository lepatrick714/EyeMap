#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdlib>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

using namespace cv;
/** Constants **/


/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat gazeMat;
cv::Point lastPoint;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
int xres, yres;
Point2f old;
float yLine; //ymax, xmax, offx, offy;
// bool calibrate, calibratetop, calibrateright;
std::vector<float> xs, ys;

/**
 * @function main
 */
int main( int argc, const char** argv ) {
//    printf("Please input if you are wearing glasses\n");
//    std::string input;
//    std::getline(std::cin, input);
//    
//    if(input == "Yes")
//        face_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//    else
  face_cascade_name = "haarcascade_frontalface_alt.xml";
    
    xres = 1280;
    yres = 800;
    gazeMat = Mat(yres, xres, CV_64F, cvScalar(255));
    
    old.y = yres - 1;
    
  CvCapture* capture;
  cv::Mat frame;
    Mat comp = imread("result.png", 1);
  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };

  // vector <
  //  cv::namedWindow("Gaze Map", CV_WINDOW_NORMAL);
  // cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  // cv::moveWindow(main_window_name, 400, 100);
  // cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  // cv::moveWindow(face_window_name, 10, 100);
  // cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
  // cv::moveWindow("Right Eye", 10, 600);
  // cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
  // cv::moveWindow("Left Eye", 10, 800);
  // cv::namedWindow("aa",CV_WINDOW_NORMAL);
  // cv::moveWindow("aa", 10, 800);
  // cv::namedWindow("aaa",CV_WINDOW_NORMAL);
  // cv::moveWindow("aaa", 10, 800);

  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155), cv::Size(23, 15),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

   // Read the video stream
  capture = cvCaptureFromCAM( -1 );
//  calibrate = true;
//    calibratetop = false;
//    calibrateright = false;
//  double tstart = 0.0;
//  double t = 0.0;
    
  if( capture ) {
      // tstart = (double)getTickCount();
    while( true ) {
      frame = cv::cvarrToMat(cvQueryFrame( capture ));
//        if(calibrate){
//            putText(frame, "Look at Mid", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
//            
//            printf("Calibrate Mid\n");
//            t = 1000*((double)getTickCount() - tstart) / getTickFrequency();
//            calibrate = t < 3000;
//            if(!calibrate){
//                float aveY = 0;
//                for(int i = 0; i < ys.size(); i++)
//                    aveY+=ys[i];
//                aveY = aveY / ys.size();
//                
//                offy = -1 * aveY;
//                
//                float aveX = 0;
//                for(int i = 0; i < xs.size(); i++)
//                    aveX+=xs[i];
//                aveX = aveX / xs.size();
//                
//                offx = -1 * aveX;
//                calibratetop = true;
//                ys.clear();
//                xs.clear();
//            }
//        }
//        
//        else if(calibratetop){
//            printf("Calibrate Top\n");
//            t = 1000*((double)getTickCount() - tstart) / getTickFrequency();
//            calibratetop = t < 6000;
//            if(!calibratetop){
//                float aveY = 0;
//                for(int i = 0; i < ys.size(); i++)
//                    aveY+=(ys[i]+offy);
//                aveY = aveY / ys.size();
//                
//                ymax = aveY;
//                
//                calibrateright = true;
//                ys.clear();
//                xs.clear();
//                printf("Done Calibrating\n");
//                printf("offx = %f, offy = %f, xmax = %f, ymax = %f\n", offx, offy, xmax, ymax);
//            }
//        }
//        
//        else if(calibrateright){
//            printf("Calibrate Right\n");
//            t = 1000*((double)getTickCount() - tstart) / getTickFrequency();
//            calibrateright = t < 9000;
//            if(!calibrateright){
//                float aveX = 0;
//                for(int i = 0; i < xs.size(); i++)
//                    aveX+=(xs[i]+offx);
//                aveX = aveX / xs.size();
//                
//                xmax = aveX;
//                
//                ys.clear();
//                xs.clear();
//                printf("Done Calibrating\n");
//                printf("offx = %f, offy = %f, xmax = %f, ymax = %f\n", offx, offy, xmax, ymax);
//            }
//        }
        
        
      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(debugImage);
        yLine = frame.rows/2;
      // printf("%i\n", yLine);
      // Apply the classifier to the frame
      if( !frame.empty() ) {
        detectAndDisplay( frame );
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }
        
      imshow("Gaze Map", gazeMat);
      imshow(main_window_name, debugImage);
      imshow("Comparison", comp);
        
      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }

    }
      
  }
    
    printf("%i, %i\n", xres, gazeMat.cols);
//
//    gazeMat.at<float>(50, 100) = 0;
//    gazeMat.at<float>(51, 100) = 0;
//    gazeMat.at<float>(52, 100) = 0;
//    gazeMat.at<float>(53, 100) = 0;
//    gazeMat.at<float>(54, 100) = 0;
//    gazeMat.at<float>(55, 100) = 0;
//    gazeMat.at<float>(56, 100) = 0;
//    printf("Channels: %i\n", gazeMat.channels());
  imwrite("gaze-map.png", gazeMat);
  releaseCornerKernels();

  return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
  cv::Mat faceROI = frame_gray(face);
  cv::Mat debugFace = faceROI;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 255, 255 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);


  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);


  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  // get corner regions
  cv::Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  cv::Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  cv::Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  cv::Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);

  //-- Find Eye Corners
    cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
    cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
    cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
    cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
    
    float yDiff = -1*(face.y + leftRightCorner.y - yLine);
    //printf("%f\n", yDiff);
    
    float correction = 0.0f;
    
    if(yDiff > 100){
        correction = (yDiff - 100) / (13.0f);
    }
    
    rightLeftCorner.y -= correction;
    rightRightCorner.y -= correction;
    leftRightCorner.y -= correction;
    leftLeftCorner.y -= correction;
    
    circle(faceROI, leftRightCorner, 3, 200);
    circle(faceROI, leftLeftCorner, 3, 200);
    circle(faceROI, rightLeftCorner, 3, 200);
    circle(faceROI, rightRightCorner, 3, 200);
    
    cv::Point2f centerLeft(leftLeftCorner.x - (leftLeftCorner.x - leftRightCorner.x) / 2, leftLeftCorner.y +(leftLeftCorner.y - leftRightCorner.y) / 2);
    circle(debugFace, centerLeft, 3, 1234);
    
    cv::Point2f centerRight(rightLeftCorner.x - (rightLeftCorner.x - rightRightCorner.x) / 2, rightLeftCorner.y +(rightLeftCorner.y - rightRightCorner.y) / 2);
    circle(debugFace, centerRight, 3, 1234);
    
    cv::Point2f centerFace(face.x + face.width / 2, face.y + leftRightCorner.y);
    cv::Point2f centerCam(frame_gray.cols/2, frame_gray.rows/2);
    cv::Point2f diff(centerFace.x - centerCam.x, centerFace.y - centerCam.y);
    //printf("%f, %f\n", centerFace.x - centerCam.x, centerFace.y - centerCam.y);
    
    float dist = log2f(2240000 / (face.width * face.height)) * 0.5f;
//    printf("Distance %f\n", dist);
    
    int div = 6;
//    printf("Center Left X: %f\n", centerLeft.x);
    float angle = (int)((leftPupil.x - centerLeft.x) / (centerLeft.x - leftLeftCorner.x) * 100);
    float angle2 = (int)((rightPupil.x - centerRight.x) / (rightRightCorner.x - centerRight.x) * 100);
    angle = (angle + angle2) / 2;
    
    
//    printf("Angle: %f\n", angle);
//    printf("Diff: %f\n", diff.x);
//    printf("Pupil X: %d\n", leftPupil.x);
//    printf("Corner Left X: %f\n", leftLeftCorner.x);
//    printf("Difference Pupil-Corner: %f\n", leftLeftCorner.x - leftPupil.x);
    //float angle = (rightPupil.x - centerRight.x) / (rightRightCorner.x - centerRight.x) * 90;
    //printf("%f\n", angle);
    double distInFromCenter = 12.0f * dist * tan(angle * 3.14159265f / 180.0f);
//    printf("Distance from your center: %f\n", distInFromCenter);
    double inHorizontal = diff.x / 640.0;
//    printf("Hoizontal Inches: %f\n", inHorizontal);
    
    if(abs(distInFromCenter + inHorizontal) < 6.39f){
        double ratio = (distInFromCenter + inHorizontal) / 6.39f;
        int pixelX = xres/2 + ratio * xres/2;
        printf("%d\n", pixelX);
        int sign = 0;
        (rand() % 2 == 0)? sign = 1: sign = -1;
        int y = old.y + sign * (rand() % 50);
        if(y < 0) y = -y;
        else if(y >= yres) y = yres - (y - yres);
        Point2f p(pixelX, y);
        line(gazeMat, old, p, Scalar(0, 0, 0));
        old = p;
        printf("%d, %i\n", pixelX, y);
        circle(gazeMat, p, 10, 200);
    }
    
    
    //70,000 at 2.5 feet
    //140,000 at 2 ft.
    //2,240,000
    
//    if(calibrate || calibratetop || calibrateright){
//        float rightYDiff = -(rightPupil.y - centerRight.y);
//        float rightXDiff = rightPupil.x - centerRight.x;
//        xs.push_back(rightXDiff);
//        ys.push_back(rightYDiff);
//        imshow(face_window_name, faceROI);
//        return;
//    }
    
//    float rightYDiff = -(rightPupil.y - centerRight.y);
//    float rightXDiff = rightPupil.x - centerRight.x + offx;
//    
//    float rightYRatio = rightYDiff / ymax;
//    float pixelY = yres/2 + rightYRatio * yres/2;
//    
//    
//    float rightXRatio = rightXDiff / xmax;
//    float pixelX = xres/2 + rightXRatio * xres/2;
    

    // printf("%f, %f\n", pixelX, pixelY);
    
    imshow(face_window_name, faceROI);
}

cv::Mat findSkin (cv::Mat &frame) {
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
//    uchar *Or = output.ptr<uchar>(y);
    cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      cv::Vec3b ycrcb = Mr[x];
//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = cv::Vec3b(0,0,0);
      }
    }
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) {
  std::vector<cv::Rect> faces;
  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];

  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
  //  findSkin(debugImage);

  for( int i = 0; i < faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], 1234);
  }
  //-- Show what you got
  if (faces.size() > 0) {
    findEyes(frame_gray, faces[0]);
  }
}
