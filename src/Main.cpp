/* MAC0460 - Aprendizagem Computacional: Modelos, Agoritmos e Aplicacoes
 * Professora Nina Hirata
 *
 *
 * Projeto da graduacao: Detector de monumentos
 *
 * Jackson José de Souza - 6796969
 * Suzana de Siqueira Santos - 6909971
 *
 ******************************************************************************/


/* ============================== BIBLIOTECAS =============================== */

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <boost/filesystem.hpp>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <string>

using namespace std;
using namespace boost::filesystem;
using namespace cv;


/* ================================= MACROS ================================= */

/* Diretorio com as imagens de treinamento */
#define TRAINING_DATA_DIR "imagens/treinamento"
/* Diretorio com as imagens de teste */
#define EVAL_DATA_DIR "imagens/teste"


/* ============================ VARIAVEIS GLOBAIS =========================== */

vector<String> training_data; /* Vetor com o nome das imagens de treinamento */
vector<String> test_data; /* Vetor com o nome das iamgens de teste */
map<String, int> classes;
/* Parametros do dicionario do modelo BOW */
int dictionarySize = 1000;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

/* TODO: Transfomar em variaveis locais */
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor;
Ptr<FeatureDetector> detector;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
Ptr<BOWImgDescriptorExtractor> bowDE;

Ptr<Mat> trainingData;
Ptr<Mat> labels;
Ptr<Mat> evalData;
Ptr<Mat> groundTruth;

/**
 * Inicializa vetor com os arquivos
 */
void extractTrainingVocabulary(const path& basepath) {
    for (directory_iterator iter = directory_iterator(basepath); iter
            != directory_iterator(); iter++) {
        directory_entry entry = *iter;
        vector <string> classe;
        if (is_directory(entry.path())) {
            boost::split(classe, entry.path().string(),  boost::is_any_of("/"));
            if (classes.find(classe[2]) == classes.end()) {
                classes[classe[2]] = classes.size();
                cout << classe[2] << " -> " << classes[classe[2]] << endl;
            }
            cout << "Processing directory " << entry.path().string() << endl;
            extractTrainingVocabulary(entry.path());
        } else {

            path entryPath = entry.path();
            if (entryPath.extension() == ".png") {

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
            if (entryPath.extension() == ".png") {
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
                        bowDE->compute(img, keypoints, bowDescriptor);
                        descriptors.push_back(bowDescriptor);
                        vector <string> classe;
                        boost::split(classe, entryPath.string(),  boost::is_any_of("/"));
                        float label= classes[classe[2]];
                        cout << "\nIMAGEM " << entryPath.string() << " é da classe "  << classe[2] << " - " << label << endl;

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

void createDictionary(string d, string e) {
    extractor = DescriptorExtractor::create(e);
    detector = FeatureDetector::create(d);
    cout<<"Creating dictionary..."<<endl;
    bowDE = new BOWImgDescriptorExtractor(extractor, matcher);
    extractTrainingVocabulary(path(TRAINING_DATA_DIR));
    vector<Mat> descriptors = bowTrainer.getDescriptors();
    int count=0;
    for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
    {
        count+=iter->rows;
    }
    cout<<"Clustering "<<count<<" features"<<endl;
    Mat dictionary = bowTrainer.cluster();
    bowDE->setVocabulary(dictionary);
}

void processTrainingData () {
    cout<<"Processing training data..."<<endl;
    trainingData = new Mat(0, dictionarySize, CV_32FC1);
    labels = new Mat(0, 1, CV_32FC1);
    extractBOWDescriptor(path(TRAINING_DATA_DIR), *trainingData, *labels);
}

void processEvaluationData() {
    cout<<"Processing evaluation data..."<<endl;
    evalData = new Mat(0, dictionarySize, CV_32FC1);
    groundTruth = new Mat(0, 1, CV_32FC1);
    extractBOWDescriptor(path(EVAL_DATA_DIR), *evalData, *groundTruth);
}

/*
void train (char name[]) {
    if (strcmp(name, "NormalBayesClassifier") == 0) {
        CvNormalBayesClassifier classifier;
        classifier.train(*trainingData, *labels);
    }
    else if (strcmp(name, "KNearest") == 0) {
        CvKNearest classifier;
        classifier.train(*trainingData, *labels);
    }
    else if (strcmp(name, "SVM")) {
        CvSVM classifier;
        classifier.train(*trainingData, *labels);
    }
    else if (strcmp(name, "DTree")) {
        CvDTree classifier;
        classifier.train(*trainingData, *labels);
    }
    else if (strcmp(name, "Boost")) {
        CvBoost classifier;
        classifier.train(*trainingData, *labels);
    }
    else if (strcmp(name, "GradientBoostedTrees")) {

    }
    else if (strcmp("Random Trees")) {

    }
    else if ("ExpectationMaximization") {

    }
    else if ("NeuralNetworks") {

    }

}
*/

int main(int argc, char ** argv) {
    char extractors[][6] = {"SIFT", "SURF", "ORB", "BRIEF"};
    char detectors[][11] = {"FAST", "STAR", "SIFT", "SURF", "ORB", "MSER", "GFTT", "HARRIS", "Dense", "SimpleBlob"};

    createDictionary(detectors[3], extractors[1]);
    processTrainingData();

    CvNormalBayesClassifier classifier;
    cout<<"Training classifier..."<<endl;
    classifier.train(*trainingData, *labels);

    processEvaluationData();
    cout<<"Evaluating classifier..."<<endl;
    Mat results;
    classifier.predict(*evalData, &results);

    double errorRate = (double) countNonZero(*groundTruth - results) / evalData->rows;

   cout << "Error rate: " << errorRate << endl;
}
