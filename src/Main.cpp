/*
 * Main.cpp
 *
 *  Created on: 10/10/2012
 *      Author: su
 */

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>

using namespace cv;

int main( int argc, char** argv ) {
    Mat image;
    image = imread( argv[1], 1 );

    Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");

    if( argc != 2 || !image.data ) {
      printf( "No image data \n" );
      return -1;
    }

    namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
    imshow( "Display Image", image );

    waitKey(0);

    return 0;
}
