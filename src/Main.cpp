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
    vector<KeyPoint> keypoints;

    if (argc != 2) {
      printf( "Uso:\n./%s nome_da_imagem\n", argv[1]);
      return -1;
    }

    image = imread( argv[1], 1 );
    if (!image.data) {
        printf("Erro ao abrir arquivo %s", argv[1]);
        return -1;
    }

    Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");

    /* TODO Em algum momento, faremos a extração dos keypoints: */
    detector->detect(image, keypoints);

   return 0;
}

