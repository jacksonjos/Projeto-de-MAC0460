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
    Mat features;
    vector<KeyPoint> keypoints;
    Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

    /* TODO entender os argumentos */
    int dictionarySize = 1000;
    TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
    int retries = 1;
    int flags = KMEANS_PP_CENTERS;

    BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
    BOWImgDescriptorExtractor bowDE(extractor, matcher);

    if (argc != 2) {
        printf( "Uso:\n./%s nome_da_imagem\n", argv[0]);
        return -1;
    }

    image = imread( argv[1], 1 );
    if (!image.data) {
        printf("Erro ao abrir arquivo %s\n", argv[1]);
        return -1;
    }

    /* TODO Para cada imagem de treinamento */
    detector->detect(image, keypoints);
    extractor->compute(image, keypoints, features);
    bowTrainer.add(features);

    /* Criação do dicionário */
    Mat dictionary = bowTrainer.cluster();
    bowDE.setVocabulary(dictionary);

    Mat trainingData(0, dictionarySize, CV_32FC1);
    Mat labels(0, 1, CV_32FC1);

    /* TODO Para cada imagem de treinamento */
    detector->detect(image, keypoints);
    Mat bowDescriptor;
    bowDE.compute(image, keypoints, bowDescriptor);
    trainingData.push_back(bowDescriptor);
    /* float label=atof(entryPath.filename().c_str());
labels.push_back(label); */

    /* Treinamento do classificador */
    NormalBayesClassifier classifier;
    classifier.train(trainingData, labels);

    Mat evalData(0, dictionarySize, CV_32FC1);
    Mat groundTruth(0, 1, CV_32FC1);
    /* TODO Para cada imagem de validação */
    detector->detect(image, keypoints);
    bowDE.compute(image, keypoints, bowDescriptor);
    evalData.push_back(bowDescriptor);
    /*label=atof(entryPath.filename().c_str());
groundTruth.push_back(label); */

    /* Avaliação */
    Mat results;
    classifier.predict(evalData, &results);
    double errorRate = (double) countNonZero(groundTruth - results) / evalData.rows;

    return 0;
}


