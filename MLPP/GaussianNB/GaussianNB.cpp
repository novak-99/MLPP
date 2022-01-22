//
//  GaussianNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "GaussianNB.hpp"
#include "Stat/Stat.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Utilities/Utilities.hpp"

#include <iostream>
#include <algorithm>
#include <random>

namespace MLPP{
    GaussianNB::GaussianNB(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int class_num)
    : inputSet(inputSet), outputSet(outputSet), class_num(class_num)
    {
        y_hat.resize(outputSet.size());
        Evaluate();
        LinAlg alg;
    }

    std::vector<double> GaussianNB::modelSetTest(std::vector<std::vector<double>> X){
        std::vector<double> y_hat;
        for(int i = 0; i < X.size(); i++){
            y_hat.push_back(modelTest(X[i]));
        }
        return y_hat;
    }
    
    double GaussianNB::modelTest(std::vector<double> x){
        Stat stat;
        LinAlg alg;

        double score[class_num];
        double y_hat_i = 1;
        for(int i = class_num - 1; i >= 0; i--){
            y_hat_i += std::log(priors[i] * (1 / sqrt(2 * M_PI * sigma[i] * sigma[i])) * exp(-(x[i] * mu[i]) * (x[i] * mu[i]) / (2 * sigma[i] * sigma[i])));
            score[i] = exp(y_hat_i);
        }
        return std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(double)));
    }

    double GaussianNB::score(){
        Utilities util;
        return util.performance(y_hat, outputSet);
    }

    void GaussianNB::Evaluate(){
        Stat stat;
        LinAlg alg;

        // Computing mu_k_y and sigma_k_y
        mu.resize(class_num);
        sigma.resize(class_num);
        for(int i = class_num - 1; i >= 0; i--){
            std::vector<double> set; 
            for(int j = 0; j < inputSet.size(); j++){
                for(int k = 0; k < inputSet[j].size(); k++){
                    if(outputSet[j] == i){
                        set.push_back(inputSet[j][k]);
                    }
                }
            }
            mu[i] = stat.mean(set);
            sigma[i] = stat.standardDeviation(set);
        }

        // Priors
        priors.resize(class_num);
        for(int i = 0; i < outputSet.size(); i++){
            priors[int(outputSet[i])]++;
        }
        priors = alg.scalarMultiply( double(1)/double(outputSet.size()), priors);

        for(int i = 0; i < outputSet.size(); i++){
            double score[class_num];
            double y_hat_i = 1;
            for(int j = class_num - 1; j >= 0; j--){
                for(int k = 0; k < inputSet[i].size(); k++){
                    y_hat_i += std::log(priors[j] * (1 / sqrt(2 * M_PI * sigma[j] * sigma[j])) * exp(-(inputSet[i][k] * mu[j]) * (inputSet[i][k] * mu[j]) / (2 * sigma[j] * sigma[j])));
                }
                score[j] = exp(y_hat_i);
                std::cout << score[j] << std::endl;
            }
            y_hat[i] = std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(double)));
            std::cout << std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(double))) << std::endl;
        }
    }
}