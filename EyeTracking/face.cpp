#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

using namespace cv;

int main()
{
  cout <<"Please enter path" << endl;
  string fpath;
  getline(cin,fpath);
  Mat m;
  m = imread(fpath,CV_LOAD_IMAGE_COLOR);
  namedWindow("FACE",WINDOW_AUTOSIZE);



  String face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
  String eyes_cascade_name = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";
  CascadeClassifier face_cascade;
  CascadeClassifier eyes_cascade;

  if( !face_cascade.load( face_cascade_name ) )
  {
    printf("--(!)Error loading face cascade\n"); return -1;
  };
  if( !eyes_cascade.load( eyes_cascade_name ) )
  {
     printf("--(!)Error loading eyes cascade\n"); return -1;
  };

  Mat m_grey;
  cvtColor(m,m_grey,COLOR_RGB2GRAY);
  vector<Rect> faces;
  equalizeHist(m_grey,m_grey);

  face_cascade.detectMultiScale(m_grey,faces,1.1,3,0|CASCADE_SCALE_IMAGE,Size(30,30));


    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( m, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = m_grey( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            cout << eye_center.x << " " << eye_center.y << endl;
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( m, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
    imshow( "FACE", m );
    waitKey(0);

  return 0;
}
