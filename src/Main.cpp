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
#include <cmath>
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

#define NDETECTORS 4
#define NEXTRACTORS 2
#define NCLASSIFIERS 1

/* ============================ VARIAVEIS GLOBAIS =========================== */

int confusionMatrix[6][6];                          //GAMBIIIIIIII

vector<String> training_data; /* Vetor com o nome das imagens de treinamento */
vector<String> test_data; /* Vetor com o nome das iamgens de teste */
map<String, int> classes;
/* Parametros do dicionario do modelo BOW */
int dictionarySize = 300;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
Mat results;

/* TODO: Transfomar em variaveis locais */
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor;
Ptr<FeatureDetector> detector;
Ptr<BOWKMeansTrainer> bowTrainer;
Ptr<BOWImgDescriptorExtractor> bowDE;

Ptr<Mat> trainingData;
Ptr<Mat> labels;
Ptr<Mat> evalData;
Ptr<Mat> groundTruth;

CvNormalBayesClassifier NBclassifier;
CvKNearest KNNclassifier;
CvSVM SVMclassifier;
CvDTree DTclassifier;

void createAndFillConfusionMatrix (Mat realValues, Mat supposedRealValues) {            //Tirar testes e printfs
    int i, j;

    for (i = 0; i < 6; i++) //colocar o valor verdadeiro
        for (j = 0; j < 6; j++)
            confusionMatrix[i][j] = 0;

    for (i = 0;(i < realValues.rows) && (i < supposedRealValues.rows); i++) {
        confusionMatrix[(int)supposedRealValues.at<float>(i,0)-1][(int)realValues.at<float>(i,0)-1]++;
        //printf("realValue %.0f supposedRealValue %.0f\n", realValues.at<float>(i,0), supposedRealValues.at<float>(i,0));
    }

    printf("Matriz de confusao:\n \t1\t2\t3\t4\t5\t6\n\n");          //TESTE
    for (i = 0; i < 6; i++) {
    	printf("%d", i+1);
        for (j = 0; j < 6; j++)
            printf("\t%d", confusionMatrix[i][j]);
    	printf("\n");
    }
}

// Calcula intervalo de confiança para erro "errorRate" com intervalo de confiança 
void confidenceInterval(double errorRate, int n) {
	double moduloDoIntervalo, Zn = 1.96;
	double menorValor, maiorValor;
	if ((errorRate - moduloDoIntervalo) < 0)
		 menorValor = 0;
	else
		menorValor = errorRate - moduloDoIntervalo;
		
	if ((errorRate + moduloDoIntervalo) > 1)
		 maiorValor = 1;
	else
		maiorValor = errorRate + moduloDoIntervalo;
		
	moduloDoIntervalo = Zn*sqrt(errorRate*(1-errorRate)/n);
	
	printf("O intervalo de confiança é: ]%f, %f[\n", menorValor, maiorValor);
}


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
            //cout << "Processing directory " << entry.path().string() << endl;
            extractTrainingVocabulary(entry.path());
        } else {

            path entryPath = entry.path();
            if (entryPath.extension() == ".png") {

                //cout << "Processing file " << entryPath.string() << endl;
                Mat img = imread(entryPath.string());
                if (!img.empty()) {
                    vector<KeyPoint> keypoints;
                    detector->detect(img, keypoints);
                    if (keypoints.empty()) {
                        cerr << "Erro ao extrair keypoints da imagem: "
                                << entryPath.string() << endl;
                    } else {
                        Mat features;
                        extractor->compute(img, keypoints, features);
                        bowTrainer->add(features);
                    }
                } else {
                    cerr << "Erro ao abrir a imagem: "
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
            //cout << "Processing directory " << entry.path().string() << endl;
            extractBOWDescriptor(entry.path(), descriptors, labels);
        } else {
            path entryPath = entry.path();
            if (entryPath.extension() == ".png") {
              //  cout << "Processing file " << entryPath.string() << endl;
                Mat img = imread(entryPath.string());
                if (!img.empty()) {
                    vector<KeyPoint> keypoints;
                    detector->detect(img, keypoints);
                    if (keypoints.empty()) {
                        cerr << "Erro ao extrair keypoints da imagem: "
                                << entryPath.string() << endl;
                    } else {
                        Mat bowDescriptor;
                        bowDE->compute(img, keypoints, bowDescriptor);
                        descriptors.push_back(bowDescriptor);
                        vector <string> classe;
                        boost::split(classe, entryPath.string(),  boost::is_any_of("/"));
                        float label= classes[classe[2]];
                        //cout << "\nIMAGEM " << entryPath.string() << " é da classe "  << classe[2] << " - " << label << endl;

                        labels.push_back(label);
                    }
                } else {
                    cerr << "Erro ao abrir a imagem: "
                            << entryPath.string() << endl;
                }
            }
        }
    }
}

void createDictionary(string d, string e) {
    bowTrainer =  new BOWKMeansTrainer(dictionarySize, tc, retries, flags);
    extractor = DescriptorExtractor::create(e);
    detector = FeatureDetector::create(d);
    if (extractor ==  NULL )
        cerr << "Erro ao criar extrator de descritor: " << e << endl;
    if (detector == NULL)
        cerr << "Erro ao criar detector de feature: " << d << endl;
    cout<<"Criando dicionario..."<<endl;
    bowDE = new BOWImgDescriptorExtractor(extractor, matcher);
    extractTrainingVocabulary(path(TRAINING_DATA_DIR));
    vector<Mat> descriptors = bowTrainer->getDescriptors();
    int count=0;
    for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
    {
        count+=iter->rows;
    }
    cout<<"\"Clusterizando\" "<<count<<" features"<<endl;
    Mat dictionary = bowTrainer->cluster();
    bowDE->setVocabulary(dictionary);
}

void processTrainingData () {
    cout<<"Processando conjunto de treinamento..."<<endl;
    trainingData = new Mat(0, dictionarySize, CV_32FC1);
    labels = new Mat(0, 1, CV_32FC1);
    extractBOWDescriptor(path(TRAINING_DATA_DIR), *trainingData, *labels);
}

void processEvaluationData() {
    cout<<"Processando comjunto de teste..."<<endl;
    evalData = new Mat(0, dictionarySize, CV_32FC1);
    groundTruth = new Mat(0, 1, CV_32FC1);
    extractBOWDescriptor(path(EVAL_DATA_DIR), *evalData, *groundTruth);
}


void train (char name[]) {
    if (strcmp(name, "NormalBayesClassifier") == 0)
        NBclassifier.train(*trainingData, *labels);
    else if (strcmp(name, "KNearest") == 0)
        KNNclassifier.train(*trainingData, *labels);
    else if (strcmp(name, "SVM") == 0)
        SVMclassifier.train(*trainingData, *labels);
    else if (strcmp(name, "DTree") == 0)
        DTclassifier.train(*trainingData, CV_ROW_SAMPLE, *labels);

    /*
    else if (strcmp(name, "Boost") == 0) {
        CvBoost classifier;
        classifier.train(*trainingData, CV_ROW_SAMPLE, *labels);
    }
    else if (strcmp(name, "GradientBoostedTrees") == 0) {
        CvGBTrees classifier;
        classifier.train(*trainingData,  CV_ROW_SAMPLE, *labels);
    }
    else if (strcmp(name, "RandomTrees") == 0) {
        CvRTrees classifier;
        classifier.train(*trainingData,  CV_ROW_SAMPLE, *labels);
    }

    else if (strcmp(name, "NeuralNetworks") == 0) {
        CvANN_MLP classifier;
        classifier.train(*trainingData, *labels);
    }*/

}

void predict (char name[]) {
    if (strcmp(name, "NormalBayesClassifier") == 0)
        NBclassifier.predict(*evalData, &results);

    else if (strcmp(name, "KNearest") == 0)
        KNNclassifier.find_nearest(*evalData, 1, &results);

    else if (strcmp(name, "SVM") == 0) {
        int i;
        results = *(new Mat(evalData->rows, 1, CV_32FC1));
        for (i = 0; i < evalData->rows; i++)
            results.at<float>(i, 0) = SVMclassifier.predict(evalData->row(i));
    }

    else if (strcmp(name, "DTree") == 0) {
        int i;
        results = *(new Mat(evalData->rows, 1, CV_32FC1));
        for (i = 0; i < evalData->rows; i++)
            results.at<float>(i, 0) = DTclassifier.predict(evalData->row(i))->value;
    }
    /*
    else if (strcmp(name, "Boost") == 0) {
        CvBoost classifier;
        classifier.predict(*evalData, &results);
    }
    else if (strcmp(name, "GradientBoostedTrees") == 0) {
        CvGBTrees classifier;
        classifier.predict(*trainingData, &results);
    }
    else if (strcmp(name, "RandomTrees") == 0) {
        CvRTrees classifier;
        classifier.predict(*trainingData, &results);
    }

    else if (strcmp(name, "NeuralNetworks") == 0) {
        CvANN_MLP classifier;
        classifier.train(*trainingData, *labels);
    }*/

}

int main(int argc, char ** argv) {
    char extractors[][6] = {"SIFT", "SURF"}; // "BRIEF", "ORB"
    char detectors[][11] = {"FAST", "STAR", "SIFT", "SURF"};
    //"ORB", "MSER", "GFTT", "HARRIS", "Dense", "SimpleBlob"};

    char classifiers[][22] = {"DTree", "SVM", "NormalBayesClassifier", "KNearest"};

    int i, j, k;

    for (i = 0; i < NDETECTORS; i++) {
        for (j = 0; j < NEXTRACTORS; j++) {
            createDictionary(detectors[i], extractors[j]);
            processTrainingData();
            processEvaluationData();
            for (k = 0; k < NCLASSIFIERS; k++) {
                printf("\n_______ %s - %s - %s _______\n", classifiers[k], detectors[i], extractors[j]);
                cout<<"Treinando classificador..."<<endl;
                train(classifiers[k]);
                cout<<"Avaliando classificador..."<<endl;
                predict(classifiers[k]);
                double errorRate = (double) countNonZero(*groundTruth - results) / evalData->rows;

                cout << "Taxa de erro: " << errorRate << endl;
				confidenceInterval(errorRate, results.rows);
                createAndFillConfusionMatrix(*groundTruth, results);
            }

        }
    }
}
