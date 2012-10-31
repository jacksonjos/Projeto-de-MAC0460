/*
 * Main.cpp
 *
 *  Created on: 10/10/2012
 *      Author: su
 */

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

//location of the training data
#define TRAINING_DATA_DIR "imagens/treinamento"
//location of the evaluation data
#define EVAL_DATA_DIR "imagens/teste"

//See article on BoW model for details
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");

//See article on BoW model for details
int dictionarySize = 1000;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

//See article on BoW model for details
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
//See article on BoW model for details
BOWImgDescriptorExtractor bowDE(extractor, matcher);

/**
 * \brief Recursively traverses a folder hierarchy. Extracts features from the training images and adds them to the bowTrainer.
 */
void extractTrainingVocabulary(const path& basepath) {
    for (directory_iterator iter = directory_iterator(basepath); iter
            != directory_iterator(); iter++) {
        directory_entry entry = *iter;

        if (is_directory(entry.path())) {

            cout << "Processing directory " << entry.path().string() << endl;
            extractTrainingVocabulary(entry.path());

        } else {

            path entryPath = entry.path();
            if (entryPath.extension() == ".jpg") {

                cout << "Processing file " << entryPath.string() << endl;
                Mat img = imread(entryPath.string());
                if (!img.empty()) {
                    vector<KeyPoint> keypoints;
                    detector->detect(img, keypoints);
                    if (keypoints.empty()) {
                        cerr << "Warning: Could not find key points in image: "
                                << entryPath.string() << endl;
                    } else {
                        Mat features;
                        extractor->compute(img, keypoints, features);
                        bowTrainer.add(features);
                    }
                } else {
                    cerr << "Warning: Could not read image: "
                            << entryPath.string() << endl;
                }

            }
        }
    }
}

/**
 * \brief Recursively traverses a folder hierarchy. Creates a BoW descriptor for each image encountered.
 */
void extractBOWDescriptor(const path& basepath, Mat& descriptors, Mat& labels) {
    for (directory_iterator iter = directory_iterator(basepath); iter
            != directory_iterator(); iter++) {
        directory_entry entry = *iter;
        if (is_directory(entry.path())) {
            cout << "Processing directory " << entry.path().string() << endl;
            extractBOWDescriptor(entry.path(), descriptors, labels);
        } else {
            path entryPath = entry.path();
            if (entryPath.extension() == ".jpg") {
                cout << "Processing file " << entryPath.string() << endl;
                Mat img = imread(entryPath.string());
                if (!img.empty()) {
                    vector<KeyPoint> keypoints;
                    detector->detect(img, keypoints);
                    if (keypoints.empty()) {
                        cerr << "Warning: Could not find key points in image: "
                                << entryPath.string() << endl;
                    } else {
                        Mat bowDescriptor;
                        bowDE.compute(img, keypoints, bowDescriptor);
                        descriptors.push_back(bowDescriptor);
                        float label=atof(entryPath.filename().c_str());
                        labels.push_back(label);
                    }
                } else {
                    cerr << "Warning: Could not read image: "
                            << entryPath.string() << endl;
                }
            }
        }
    }
}

int main(int argc, char ** argv) {

    cout<<"Creating dictionary..."<<endl;
    extractTrainingVocabulary(path(TRAINING_DATA_DIR));
    vector<Mat> descriptors = bowTrainer.getDescriptors();
    int count=0;
    for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
    {
        count+=iter->rows;
    }
    cout<<"Clustering "<<count<<" features"<<endl;
    Mat dictionary = bowTrainer.cluster();
    bowDE.setVocabulary(dictionary);
    cout<<"Processing training data..."<<endl;
    Mat trainingData(0, dictionarySize, CV_32FC1);
    Mat labels(0, 1, CV_32FC1);
    extractBOWDescriptor(path(TRAINING_DATA_DIR), trainingData, labels);

    NormalBayesClassifier classifier;
    cout<<"Training classifier..."<<endl;

    classifier.train(trainingData, labels);

    cout<<"Processing evaluation data..."<<endl;
    Mat evalData(0, dictionarySize, CV_32FC1);
    Mat groundTruth(0, 1, CV_32FC1);
    extractBOWDescriptor(path(EVAL_DATA_DIR), evalData, groundTruth);

    cout<<"Evaluating classifier..."<<endl;
    Mat results;
    classifier.predict(evalData, &results);

    double errorRate = (double) countNonZero(groundTruth - results) / evalData.rows;
            ;
    cout << "Error rate: " << errorRate << endl;

}

int main2( int argc, char** argv ) {
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
