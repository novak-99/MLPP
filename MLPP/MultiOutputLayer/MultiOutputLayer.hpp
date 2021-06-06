//
//  MultiOutputLayer.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#ifndef MultiOutputLayer_hpp
#define MultiOutputLayer_hpp

#include "Activation/Activation.hpp"
#include "Cost/Cost.hpp"

#include <vector>
#include <map>
#include <string>

namespace  MLPP {
    class MultiOutputLayer{
        public:
            MultiOutputLayer(int n_output, int n_hidden, std::string activation, std::string cost, std::vector<std::vector<double>> input, std::string weightInit, std::string reg, double lambda, double alpha);
        
            int n_output;
            int n_hidden;
            std::string activation;
            std::string cost;

            std::vector<std::vector<double>> input;   

            std::vector<std::vector<double>> weights;
            std::vector<double> bias;
        
            std::vector<std::vector<double>> z;
            std::vector<std::vector<double>> a;

            std::map<std::string, std::vector<std::vector<double>> (Activation::*)(std::vector<std::vector<double>>, bool)> activation_map;
            std::map<std::string, std::vector<double> (Activation::*)(std::vector<double>, bool)> activationTest_map;
            std::map<std::string, double (Cost::*)(std::vector<std::vector<double>>, std::vector<std::vector<double>>)> cost_map;
            std::map<std::string, std::vector<std::vector<double>> (Cost::*)(std::vector<std::vector<double>>, std::vector<std::vector<double>>)> costDeriv_map;

            std::vector<double> z_test;
            std::vector<double> a_test; 

            std::vector<std::vector<double>> delta;

            // Regularization Params
            std::string reg;
            double lambda; /* Regularization Parameter */
            double alpha; /* This is the controlling param for Elastic Net*/
            
            std::string weightInit;

            void forwardPass();
            void Test(std::vector<double> x);
    };
}

#endif /* MultiOutputLayer_hpp */

