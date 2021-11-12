//
//  main.cpp
//  TEST_APP
//
//  Created by Marc on 1/20/21.
//

// THINGS CURRENTLY TO DO: 
// POLYMORPHIC IMPLEMENTATION OF REGRESSION CLASSES
// EXTEND SGD/MBGD SUPPORT FOR DYN. SIZED ANN 
// ADD LEAKYRELU, ELU, SELU TO ANN

// HYPOTHESIS TESTING CLASS 
// GAUSS MARKOV CHECKER CLASS

#include <iostream>
#include <ctime>
#include <vector>
#include "MLPP/UniLinReg/UniLinReg.hpp"
#include "MLPP/LinReg/LinReg.hpp"
#include "MLPP/LogReg/LogReg.hpp"
#include "MLPP/CLogLogReg/CLogLogReg.hpp"
#include "MLPP/ExpReg/ExpReg.hpp"
#include "MLPP/ProbitReg/ProbitReg.hpp"
#include "MLPP/SoftmaxReg/SoftmaxReg.hpp"
#include "MLPP/TanhReg/TanhReg.hpp"
#include "MLPP/MLP/MLP.hpp"
#include "MLPP/SoftmaxNet/SoftmaxNet.hpp"
#include "MLPP/AutoEncoder/AutoEncoder.hpp"
#include "MLPP/ANN/ANN.hpp"
#include "MLPP/MANN/MANN.hpp"
#include "MLPP/MultinomialNB/MultinomialNB.hpp"
#include "MLPP/BernoulliNB/BernoulliNB.hpp"
#include "MLPP/GaussianNB/GaussianNB.hpp"
#include "MLPP/KMeans/KMeans.hpp"
#include "MLPP/kNN/kNN.hpp"
#include "MLPP/PCA/PCA.hpp"
#include "MLPP/OutlierFinder/OutlierFinder.hpp"
#include "MLPP/Stat/Stat.hpp"
#include "MLPP/LinAlg/LinAlg.hpp"
#include "MLPP/Activation/Activation.hpp"
#include "MLPP/Cost/Cost.hpp"
#include "MLPP/Data/Data.hpp"
#include "MLPP/Convolutions/Convolutions.hpp"
#include "MLPP/SVC/SVC.hpp"
#include "MLPP/NumericalAnalysis/NumericalAnalysis.hpp"


using namespace MLPP;


double f(double x){
    return x*x*x + 2*x - 2;
}

double f_mv(std::vector<double> x){
    return x[0] * x[0] + x[1] * x[1] + x[1] + 5; 
    // Where x,y=x[0],x[1], this function is defined as:
    // f(x,y) = x^2 + y^2 + y + 5
}

int main() {

    // // OBJECTS
    Stat stat;
    LinAlg alg;
    // Activation avn;
    // Cost cost;
    // Data data; 
    // Convolutions conv; 

    // DATA SETS
    // std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8,9,10}, {3,5,9,12,15,18,21,24,27,30}};
    // std::vector<double> outputSet = {2,4,6,8,10,12,14,16,18,20};

    // std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8}, {0,0,0,0,1,1,1,1}};
    // std::vector<double> outputSet = {0,0,0,0,1,1,1,1};

    // std::vector<std::vector<double>> inputSet = {{4,3,0,-3,-4}, {0,0,0,1,1}};
    // std::vector<double> outputSet = {1,1,0,-1,-1};

    // std::vector<std::vector<double>> inputSet = {{0,1,2,3,4}};
    // std::vector<double> outputSet = {1,2,4,8,16};

    //std::vector<std::vector<double>> inputSet = {{32, 0, 7}, {2, 28, 17}, {0, 9, 23}}; 

    // std::vector<std::vector<double>> inputSet = {{1,1,0,0,1}, {0,0,1,1,1}, {0,1,1,0,1}};
    // std::vector<double> outputSet = {0,1,0,1,1};

    // std::vector<std::vector<double>> inputSet = {{0,0,1,1}, {0,1,0,1}};
    // std::vector<double> outputSet = {0,1,1,0};

    // // STATISTICS
    // std::vector<double> x = {1,2,3,4,5,6,7,8,9,10};
    // std::vector<double> y = {10,9,8,7,6,5,4,3,2,1};
    // std::vector<double> w = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

    // std::cout << "Arithmetic Mean: " << stat.mean(x) << std::endl;
    // std::cout << "Median: " << stat.median(x) << std::endl;
    // alg.printVector(stat.mode(x));
    // std::cout << "Range: " << stat.range(x) << std::endl;
    // std::cout << "Midrange: " << stat.midrange(x) << std::endl;
    // std::cout << "Absolute Average Deviation: " << stat.absAvgDeviation(x) << std::endl;
    // std::cout << "Standard Deviation: " << stat.standardDeviation(x) << std::endl;
    // std::cout << "Variance: " << stat.variance(x) << std::endl;
    // std::cout << "Covariance: " << stat.covariance(x, y) << std::endl;
    // std::cout << "Correlation: " << stat.correlation(x, y) << std::endl;
    // std::cout << "R^2: " << stat.R2(x, y) << std::endl;
    // // Returns 1 - (1/k^2)
    // std::cout << "Chebyshev Inequality: " << stat.chebyshevIneq(2) << std::endl;
    // std::cout << "Weighted Mean: " << stat.weightedMean(x, w) << std::endl;
    // std::cout << "Geometric Mean: " << stat.geometricMean(x) << std::endl;
    // std::cout << "Harmonic Mean: " << stat.harmonicMean(x) << std::endl;
    // std::cout << "Root Mean Square (Quadratic mean): " << stat.RMS(x) << std::endl;
    // std::cout << "Power Mean (p = 5): " << stat.powerMean(x, 5) << std::endl;
    // std::cout << "Lehmer Mean (p = 5): " << stat.lehmerMean(x, 5) << std::endl;
    // std::cout << "Weighted Lehmer Mean (p = 5): " << stat.weightedLehmerMean(x, w, 5) << std::endl;
    // std::cout << "Contraharmonic Mean: " << stat.contraharmonicMean(x) << std::endl;
    // std::cout << "Hernonian Mean: " << stat.heronianMean(1, 10) << std::endl;
    // std::cout << "Heinz Mean (x = 1): " << stat.heinzMean(1, 10, 1) << std::endl;
    // std::cout << "Neuman-Sandor Mean: " << stat.neumanSandorMean(1, 10) << std::endl;
    // std::cout << "Stolarsky Mean (p = 5): " << stat.stolarskyMean(1, 10, 5) << std::endl;
    // std::cout << "Identric Mean: " << stat.identricMean(1, 10) << std::endl;
    // std::cout << "Logarithmic Mean: " << stat.logMean(1, 10) << std::endl;
    // std::cout << "Absolute Average Deviation: " << stat.absAvgDeviation(x) << std::endl;

    // // LINEAR ALGEBRA
    // std::vector<std::vector<double>> A = {
    //     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    //     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    // };
    // std::vector<double> a = {4, 3, 1, 3}; 
    // std::vector<double> b = {3, 5, 6, 1};

    // alg.printMatrix(alg.matmult(alg.transpose(A), A)); 
    // std::cout << std::endl;
    // std::cout << alg.dot(a, b) << std::endl;
    // std::cout << std::endl;
    // alg.printMatrix(alg.hadamard_product(A, A));
    // std::cout << std::endl;
    // alg.printMatrix(alg.identity(10));

    // // UNIVARIATE LINEAR REGRESSION 
    // // Univariate, simple linear regression case where k = 1
    // std::vector<double> inputSet; 
    // std::vector<double> outputSet; 
    // // Analytical solution used for calculating the parameters. 
    // data.setData("/Users/marcmelikyan/Desktop/Data/FiresAndCrime.csv", inputSet, outputSet);
    // UniLinReg model(inputSet, outputSet);
    // alg.printVector(model.modelSetTest(inputSet));

    // MULIVARIATE LINEAR REGRESSION
    // std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8,9,10}, {3,5,9,12,15,18,21,24,27,30}};
    // std::vector<double> outputSet = {2,4,6,8,10,12,14,16,18,20};
    // LinReg model(alg.transpose(inputSet), outputSet); // Can use Lasso, Ridge, ElasticNet Reg
    // model.normalEquation(); 
    // model.gradientDescent(0.001, 30000, 1);
    // model.SGD(0.001, 30000, 1);
    // model.MBGD(0.001, 10000, 2, 1);
    // alg.printVector(model.modelSetTest((alg.transpose(inputSet))));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // // LOGISTIC REGRESSION
    // std::vector<std::vector<double>> inputSet; 
    // std::vector<double> outputSet; 
    // data.setData(30, "/Users/marcmelikyan/Desktop/Data/BreastCancer.csv", inputSet, outputSet);
    // LogReg model(inputSet, outputSet); 
    // model.SGD(0.001, 100000, 0);
    // model.MLE(0.1, 10000, 0);
    // alg.printVector(model.modelSetTest(inputSet));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // // PROBIT REGRESSION
    // std::vector<std::vector<double>> inputSet;
    // std::vector<double> outputSet;
    // data.setData(30, "/Users/marcmelikyan/Desktop/Data/BreastCancer.csv", inputSet, outputSet);
    // ProbitReg model(inputSet, outputSet); 
    // model.SGD(0.001, 10000, 1);
    // alg.printVector(model.modelSetTest(inputSet));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // // CLOGLOG REGRESSION
    // std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8}, {0,0,0,0,1,1,1,1}};
    // std::vector<double> outputSet = {0,0,0,0,1,1,1,1};
    // CLogLogReg model(alg.transpose(inputSet), outputSet); 
    // model.SGD(0.1, 10000, 0);
    // alg.printVector(model.modelSetTest(alg.transpose(inputSet)));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // // EXPREG REGRESSION
    // std::vector<std::vector<double>> inputSet = {{0,1,2,3,4}};
    // std::vector<double> outputSet = {1,2,4,8,16};
    // ExpReg model(alg.transpose(inputSet), outputSet); 
    // model.SGD(0.001, 10000, 0);
    // alg.printVector(model.modelSetTest(alg.transpose(inputSet)));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // // TANH REGRESSION
    // std::vector<std::vector<double>> inputSet = {{4,3,0,-3,-4}, {0,0,0,1,1}};
    // std::vector<double> outputSet = {1,1,0,-1,-1};
    // TanhReg model(alg.transpose(inputSet), outputSet); 
    // model.SGD(0.1, 10000, 0);
    // alg.printVector(model.modelSetTest(alg.transpose(inputSet)));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // // SOFTMAX REGRESSION
    // std::vector<std::vector<double>> inputSet; 
    // std::vector<double> tempOutputSet; 
    // data.setData(4, "/Users/marcmelikyan/Desktop/Data/Iris.csv", inputSet, tempOutputSet);
    // std::vector<std::vector<double>> outputSet = data.oneHotRep(tempOutputSet, 3);

    // // SUPPORT VECTOR CLASSIFICATION
    // std::vector<std::vector<double>> inputSet; 
    // std::vector<double> outputSet; 
    // data.setData(30, "/Users/marcmelikyan/Desktop/Data/BreastCancerSVM.csv", inputSet, outputSet);
    // SVC model(inputSet, outputSet, 1); 
    // model.SGD(0.00001, 100000, 1);
    // alg.printVector(model.modelSetTest(inputSet));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // SoftmaxReg model(inputSet, outputSet); 
    // model.SGD(0.001, 20000, 0);
    // alg.printMatrix(model.modelSetTest(inputSet));

    // // MLP
    // std::vector<std::vector<double>> inputSet = {{0,0,1,1}, {0,1,0,1}};
    // std::vector<double> outputSet = {0,1,1,0};
    // MLP model(alg.transpose(inputSet), outputSet, 2); 
    // model.gradientDescent(0.1, 10000, 0);
    // alg.printVector(model.modelSetTest(alg.transpose(inputSet)));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // // SOFTMAX NETWORK
    // std::vector<std::vector<double>> inputSet; 
    // std::vector<double> tempOutputSet; 
    // data.setData(13, "/Users/marcmelikyan/Desktop/Data/Wine.csv", inputSet, tempOutputSet);
    // std::vector<std::vector<double>> outputSet = data.oneHotRep(tempOutputSet, 3);

    // SoftmaxNet model(inputSet, outputSet, 5); 
    // model.SGD(0.1, 500000, 0);
    // alg.printMatrix(model.modelSetTest(inputSet));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // // AUTOENCODER
    // std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8,9,10}, {3,5,9,12,15,18,21,24,27,30}};
    // AutoEncoder model(alg.transpose(inputSet), 5); 
    // model.SGD(0.001, 300000, 0);
    // alg.printMatrix(model.modelSetTest(alg.transpose(inputSet)));
    // std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    // DYNAMICALLY SIZED ANN
    // Possible Weight Init Methods: Default, Uniform, HeNormal, HeUniform, XavierNormal, XavierUniform
    // Possible Activations: Linear, Sigmoid, Swish, Softplus, Softsign, CLogLog, Ar{Sinh, Cosh, Tanh, Csch, Sech, Coth},  GaussianCDF, GELU, UnitStep
    // Possible Loss Functions: MSE, RMSE, MBE, LogLoss, CrossEntropy, HingeLoss
    // std::vector<std::vector<double>> inputSet = {{0,0,1,1}, {0,1,0,1}};
    // std::vector<double> outputSet = {0,1,1,0};
    // ANN ann(alg.transpose(inputSet), outputSet);
    // ann.addLayer(10, "RELU", "Default", "Ridge", 0.0001);
    // ann.addLayer(10, "Sigmoid", "Default");
    // ann.addOutputLayer("Sigmoid", "LogLoss", "XavierNormal");
    // ann.gradientDescent(0.1, 80000, 0);
    // alg.printVector(ann.modelSetTest(alg.transpose(inputSet)));
    // std::cout << "ACCURACY: " << 100 * ann.score() << "%" << std::endl;

    // std::vector<std::vector<double>> inputSet = {{0,0,1,1}, {0,1,0,1}};
    // std::vector<double> outputSet = {0,1,1,0};
    // ANN ann(alg.transpose(inputSet), outputSet);
    // ann.addLayer(10, "Sigmoid");
    // ann.addLayer(10, "Sigmoid");
    // ann.addLayer(10, "Sigmoid");
    // ann.addLayer(10, "Sigmoid");
    // ann.addOutputLayer("Sigmoid", "LogLoss");
    // ann.gradientDescent(0.1, 80000, 0);
    // alg.printVector(ann.modelSetTest(alg.transpose(inputSet)));
    // std::cout << "ACCURACY: " << 100 * ann.score() << "%" << std::endl;

    // // DYNAMICALLY SIZED MANN (Multidimensional Output ANN)
    // std::vector<std::vector<double>> inputSet = {{1,2,3},{2,4,6},{3,6,9},{4,8,12}};
    // std::vector<std::vector<double>> outputSet = {{1,5}, {2,10}, {3,15}, {4,20}};

    // MANN mann(inputSet, outputSet);
    // mann.addOutputLayer("Linear", "MSE");
    // mann.gradientDescent(0.001, 80000, 0);
    // alg.printMatrix(mann.modelSetTest(inputSet));
    // std::cout << "ACCURACY: " << 100 * mann.score() << "%" << std::endl;

    // std::vector<std::vector<double>> inputSet;
    // std::vector<double> tempOutputSet;
    // data.setData(4, "/Users/marcmelikyan/Desktop/Data/Iris.csv", inputSet, tempOutputSet);
    // std::vector<std::vector<double>> outputSet = data.oneHotRep(tempOutputSet, 3);

    // std::vector<std::vector<double>> inputSet;
    // std::vector<double> tempOutputSet;
    // data.setData(784, "mini_mnist.csv", inputSet, tempOutputSet);
    // std::vector<std::vector<double>> outputSet = data.oneHotRep(tempOutputSet, 10);

    // MANN mann(inputSet, outputSet);
    // mann.addLayer(2, "RELU");
    // mann.addLayer(2, "RELU");
    // mann.addOutputLayer("Softmax", "CrossEntropy");
    // mann.gradientDescent(0.001, 80000, 1);
    // alg.printMatrix(mann.modelSetTest(inputSet));
    // std::cout << "ACCURACY: " << 100 * mann.score() << "%" << std::endl;

    // // NAIVE BAYES
    // std::vector<std::vector<double>> inputSet = {{1,1,1,1,1}, {0,0,1,1,1}, {0,0,1,0,1}};
    // std::vector<double> outputSet = {0,1,0,1,1};

    // MultinomialNB MNB(alg.transpose(inputSet), outputSet, 2);
    // alg.printVector(MNB.modelSetTest(alg.transpose(inputSet)));

    // BernoulliNB BNB(alg.transpose(inputSet), outputSet);
    // alg.printVector(BNB.modelSetTest(alg.transpose(inputSet)));

    // GaussianNB GNB(alg.transpose(inputSet), outputSet, 2);
    // alg.printVector(GNB.modelSetTest(alg.transpose(inputSet)));

    // // KMeans
    // std::vector<std::vector<double>> inputSet = {{32, 0, 7}, {2, 28, 17}, {0, 9, 23}}; 
    // KMeans kmeans(inputSet, 3, "KMeans++");
    // kmeans.train(3, 1);
    // std::cout << std::endl;
    // alg.printMatrix(kmeans.modelSetTest(inputSet)); // Returns the assigned centroids to each of the respective training examples
    // std::cout << std::endl;
    // alg.printVector(kmeans.silhouette_scores());

    // // kNN 
    // std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8}, {0,0,0,0,1,1,1,1}};
    // std::vector<double> outputSet = {0,0,0,0,1,1,1,1};
    // kNN knn(alg.transpose(inputSet), outputSet, 8);
    // alg.printVector(knn.modelSetTest(alg.transpose(inputSet)));
    // std::cout << "ACCURACY: " << 100 * knn.score() << "%" << std::endl;


    // // CONVOLUTION, POOLING, ETC.. 
    // std::vector<std::vector<double>> input = {
    //     {1,1,1,1,0,0,0,0},
    //     {1,1,1,1,0,0,0,0},
    //     {1,1,1,1,0,0,0,0},
    //     {1,1,1,1,0,0,0,0},
    //     {1,1,1,1,0,0,0,0},
    //     {1,1,1,1,0,0,0,0},
    //     {1,1,1,1,0,0,0,0},
    //     {1,1,1,1,0,0,0,0}
    // };

    // alg.printMatrix(conv.convolve(input, conv.getPrewittVertical(), 1)); // Can use padding
    // alg.printMatrix(conv.pool(input, 4, 4, "Max")); // Can use Max, Min, or Average pooling. 

    // std::vector<std::vector<std::vector<double>>> tensorSet; 
    // tensorSet.push_back(input);
    // tensorSet.push_back(input);
    // alg.printVector(conv.globalPool(tensorSet, "Average")); // Can use Max, Min, or Average global pooling. 

    // // PCA, SVD, eigenvalues & eigenvectors
    // std::vector<std::vector<double>> inputSet = {{1,1}, {1,1}};
    // auto [Eigenvectors, Eigenvalues] = alg.eig(inputSet); 
    // std::cout << "Eigenvectors:" << std::endl; 
    // alg.printMatrix(Eigenvectors);
    // std::cout << std::endl;
    // std::cout << "Eigenvalues:" << std::endl; 
    // alg.printMatrix(Eigenvalues);

    // auto [U, S, Vt] = alg.SVD(inputSet);

    // // PCA done using Jacobi's method to approximate eigenvalues and eigenvectors.
    // PCA dr(inputSet, 1); // 1 dimensional representation. 
    // std::cout << std::endl;
    // std::cout << "Dimensionally reduced representation:" << std::endl;
    // alg.printMatrix(dr.principalComponents());
    // std::cout << "SCORE: " << dr.score() << std::endl; 


    // // NLP/DATA
    // std::string verbText = "I am appearing and thinking, as well as conducting.";
    // std::cout << "Stemming Example:" << std::endl;
    // std::cout << data.stemming(verbText) << std::endl;
    // std::cout << std::endl;

    // std::vector<std::string> sentences = {"He is a good boy", "She is a good girl", "The boy and girl are good"};
    // std::cout << "Bag of Words Example:" << std::endl;
    // alg.printMatrix(data.BOW(sentences, "Default"));
    // std::cout << std::endl;
    // std::cout << "TFIDF Example:" << std::endl;
    // alg.printMatrix(data.TFIDF(sentences));
    // std::cout << std::endl;

    // std::cout << "Tokenization:" << std::endl;
    // alg.printVector(data.tokenize(verbText));
    // std::cout << std::endl;

    // std::cout << "Word2Vec:" << std::endl;
    // std::string textArchive = {"He is a good boy. She is a good girl. The boy and girl are good."};
    // std::vector<std::string> corpus = data.splitSentences(textArchive);
    // auto [wordEmbeddings, wordList] = data.word2Vec(corpus, "CBOW", 2, 2, 0.1, 10000); // Can use either CBOW or Skip-n-gram.
    // alg.printMatrix(wordEmbeddings);
    // std::cout << std::endl;

    // std::vector<std::vector<double>> inputSet = {{1,2},{2,3},{3,4},{4,5},{5,6}};
    // std::cout << "Feature Scaling Example:" << std::endl;
    // alg.printMatrix(data.featureScaling(inputSet));
    // std::cout << std::endl;

    // std::cout << "Mean Centering Example:" << std::endl;
    // alg.printMatrix(data.meanCentering(inputSet));
    // std::cout << std::endl;

    // std::cout << "Mean Normalization Example:" << std::endl;
    // alg.printMatrix(data.meanNormalization(inputSet));
    // std::cout << std::endl;

    // // Outlier Finder
    // std::vector<double> inputSet = {1,2,3,4,5,6,7,8,9,23554332523523};
    // OutlierFinder outlierFinder(2); // Any datapoint outside of 2 stds from the mean is marked as an outlier. 
    // alg.printVector(outlierFinder.modelTest(inputSet));

    // // Testing new Functions
    // double z_s = 0.001;
    // std::cout << avn.logit(z_s) << std::endl;
    // std::cout << avn.logit(z_s, 1) << std::endl;

    // std::vector<double> z_v = {0.001};
    // alg.printVector(avn.logit(z_v));
    // alg.printVector(avn.logit(z_v, 1));

    // std::vector<std::vector<double>> Z_m = {{0.001}};
    // alg.printMatrix(avn.logit(Z_m));
    // alg.printMatrix(avn.logit(Z_m, 1));

    // std::cout << alg.trace({{1,2}, {3,4}}) << std::endl;
    // alg.printMatrix(alg.pinverse({{1,2}, {3,4}}));
    // alg.printMatrix(alg.diag({1,2,3,4,5}));
    // alg.printMatrix(alg.kronecker_product({{1,2,3,4,5}}, {{6,7,8,9,10}}));
    // alg.printMatrix(alg.matrixPower({{5,5},{5,5}}, 2));
    // alg.printVector(alg.solve({{1,1}, {1.5, 4.0}}, {2200, 5050}));

    // std::vector<std::vector<double>> matrixOfCubes = {{1,2,64,27}};
    // std::vector<double> vectorOfCubes = {1,2,64,27};
    // alg.printMatrix(alg.cbrt(matrixOfCubes));
    // alg.printVector(alg.cbrt(vectorOfCubes));
    // std::cout << alg.max({{1,2,3,4,5}, {6,5,3,4,1}, {9,9,9,9,9}}) << std::endl;
    // std::cout << alg.min({{1,2,3,4,5}, {6,5,3,4,1}, {9,9,9,9,9}}) << std::endl;

    // std::vector<double> chicken; 
    // data.getImage("../../Data/apple.jpeg", chicken);
    // alg.printVector(chicken);

    // // TESTING QR DECOMP. EXAMPLE VIA WIKIPEDIA. SEE https://en.wikipedia.org/wiki/QR_decomposition.

    // std::vector<std::vector<double>> P = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
    // alg.printMatrix(P);

    // alg.printMatrix(alg.gramSchmidtProcess(P));

    // auto [Q, R] = alg.QRD(P); // It works! 
    
    //  alg.printMatrix(Q);

    //  alg.printMatrix(R); 

    // // Checking positive-definiteness checker. For Cholesky Decomp. 
    // std::vector<std::vector<double>> A = 
    // {
    //     {1,-1,-1,-1},                        
    //     {-1,2,2,2},
    //     {-1,2,3,1},
    //     {-1,2,1,4}
    // };

    // std::cout << std::boolalpha << alg.positiveDefiniteChecker(A) << std::endl;
    // auto [L, Lt] = alg.chol(A); // WORKS !!!!
    // alg.printMatrix(L);
    // alg.printMatrix(Lt);

    // Checks for numerical analysis class.
    NumericalAnalysis numAn;

    std::cout << numAn.numDiff(&f, 1) << std::endl;
    std::cout << numAn.newtonRaphsonMethod(&f, 1, 1000) << std::endl;

    std::cout << numAn.numDiff(&f_mv, {1, 1}, 1) << std::endl; // Derivative w.r.t. x.

    alg.printVector(numAn.jacobian(&f_mv, {1, 1}));

    return 0;
}

