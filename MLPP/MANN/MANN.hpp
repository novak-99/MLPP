//
//  MANN.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#ifndef MANN_hpp
#define MANN_hpp

#include "HiddenLayer/HiddenLayer.hpp"
#include "MultiOutputLayer/MultiOutputLayer.hpp"

#include <vector>
#include <string>

namespace  MLPP{

class MANN{
        public:
        MANN(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet);
        ~MANN();
        std::vector<std::vector<double>> modelSetTest(std::vector<std::vector<double>> X);
        std::vector<double> modelTest(std::vector<double> x);
        void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
        double score(); 
        void save(std::string fileName);

        void addLayer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", double lambda = 0.5, double alpha = 0.5); 
        void addOutputLayer(std::string activation, std::string loss, std::string weightInit = "Default", std::string reg = "None", double lambda = 0.5, double alpha = 0.5); 
        
        private:
            double Cost(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);
            void forwardPass();

            std::vector<std::vector<double>> inputSet;
            std::vector<std::vector<double>> outputSet;
            std::vector<std::vector<double>> y_hat;

            std::vector<HiddenLayer> network;
            MultiOutputLayer *outputLayer;

            int n;
            int k;
            int n_output;
    };
}

#endif /* MANN_hpp */