//
//  WGAN.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#ifndef WGAN_hpp
#define WGAN_hpp

#include "HiddenLayer/HiddenLayer.hpp"
#include "OutputLayer/OutputLayer.hpp"

#include <vector>
#include <tuple>
#include <string>

namespace  MLPP{

class WGAN{
        public:
        WGAN(double k, std::vector<std::vector<double>> outputSet);
        ~WGAN();
        std::vector<std::vector<double>> generateExample(int n);
        void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
        double score(); 
        void save(std::string fileName);

        void addLayer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", double lambda = 0.5, double alpha = 0.5); 
        void addOutputLayer(std::string weightInit = "Default", std::string reg = "None", double lambda = 0.5, double alpha = 0.5); 
        
        private:
            std::vector<std::vector<double>> modelSetTestGenerator(std::vector<std::vector<double>> X); // Evaluator for the generator of the WGAN.
            std::vector<double> modelSetTestDiscriminator(std::vector<std::vector<double>> X); // Evaluator for the discriminator of the WGAN.

            double Cost(std::vector<double> y_hat, std::vector<double> y);

            void forwardPass();
            void updateDiscriminatorParameters(std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations, std::vector<double> outputLayerUpdation, double learning_rate);
            void updateGeneratorParameters(std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations, double learning_rate);
            std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<double>> computeDiscriminatorGradients(std::vector<double> y_hat, std::vector<double> outputSet);
            std::vector<std::vector<std::vector<double>>> computeGeneratorGradients(std::vector<double> y_hat, std::vector<double> outputSet);

            void UI(int epoch, double cost_prev, std::vector<double> y_hat, std::vector<double> outputSet);

            std::vector<std::vector<double>> outputSet;
            std::vector<double> y_hat;

            std::vector<HiddenLayer> network;
            OutputLayer *outputLayer;

            int n;
            int k;
    };
}

#endif /* WGAN_hpp */