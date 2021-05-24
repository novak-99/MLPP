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
#include <string>

namespace  MLPP{

class ANN{
        public:
        ANN(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet);
        ~ANN();
        std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
        double modelTest(std::vector<double> x);
        void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
        double score(); 
        void save(std::string fileName);

        void addLayer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", double lambda = 0.5, double alpha = 0.5); 
        void addOutputLayer(std::string activation, std::string loss, std::string weightInit = "Default", std::string reg = "None", double lambda = 0.5, double alpha = 0.5); 
        
        private:
            double Cost(std::vector<double> y_hat, std::vector<double> y);
            void forwardPass();

            std::vector<std::vector<double>> inputSet;
            std::vector<double> outputSet;
            std::vector<double> y_hat;

            std::vector<HiddenLayer> network;
            OutputLayer *outputLayer;

            int n;
            int k;
    };
}

#endif /* ANN_hpp */