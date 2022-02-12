//
//  ANN.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#ifndef ANN_hpp
#define ANN_hpp

#include "HiddenLayer/HiddenLayer.hpp"
#include "OutputLayer/OutputLayer.hpp"

#include <vector>
#include <tuple>
#include <string>

namespace  MLPP{

class ANN{
        public:
        ANN(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet);
        ~ANN();
        std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
        double modelTest(std::vector<double> x);
        void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
        void SGD(double learning_rate, int max_epoch, bool UI = 1);
        void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
        void Momentum(double learning_rate, int max_epoch, int mini_batch_size, double gamma, bool NAG, bool UI = 1);
        void Adagrad(double learning_rate, int max_epoch, int mini_batch_size, double e, bool UI = 1);
        void Adadelta(double learning_rate, int max_epoch, int mini_batch_size, double b1, double e, bool UI = 1);
        void Adam(double learning_rate, int max_epoch, int mini_batch_size, double b1, double b2, double e, bool UI = 1);
        void Adamax(double learning_rate, int max_epoch, int mini_batch_size, double b1, double b2, double e, bool UI = 1);
        void Nadam(double learning_rate, int max_epoch, int mini_batch_size, double b1, double b2, double e, bool UI = 1);
        void AMSGrad(double learning_rate, int max_epoch, int mini_batch_size, double b1, double b2, double e, bool UI = 1);
        double score(); 
        void save(std::string fileName); 

        void setLearningRateScheduler(std::string type, double decayConstant);
        void setLearningRateScheduler(std::string type, double decayConstant, double dropRate);

        void addLayer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", double lambda = 0.5, double alpha = 0.5); 
        void addOutputLayer(std::string activation, std::string loss, std::string weightInit = "Default", std::string reg = "None", double lambda = 0.5, double alpha = 0.5); 
        
        private:
            double applyLearningRateScheduler(double learningRate, double decayConstant, double epoch, double dropRate);

            double Cost(std::vector<double> y_hat, std::vector<double> y);

            void forwardPass();
            void updateParameters(std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations, std::vector<double> outputLayerUpdation, double learning_rate);
            std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<double>> computeGradients(std::vector<double> y_hat, std::vector<double> outputSet);

            void UI(int epoch, double cost_prev, std::vector<double> y_hat, std::vector<double> outputSet);


            std::vector<std::vector<double>> inputSet;
            std::vector<double> outputSet;
            std::vector<double> y_hat;

            std::vector<HiddenLayer> network;
            OutputLayer *outputLayer;

            int n;
            int k;

            std::string lrScheduler;
            double decayConstant;
            double dropRate;
    };
}

#endif /* ANN_hpp */