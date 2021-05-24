//
//  Reg.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include <iostream>
#include <random>
#include "Reg.hpp"

namespace MLPP{

    double Reg::regTerm(std::vector<double> weights, double lambda, double alpha, std::string reg){
        if(reg == "Ridge"){
            double reg = 0;
            for(int i = 0; i < weights.size(); i++){
                reg += weights[i] * weights[i];
            }
            return reg * lambda / 2;
        }
        else if(reg == "Lasso"){
            double reg = 0;
            for(int i = 0; i < weights.size(); i++){
                reg += abs(weights[i]);
            }
            return reg * lambda;
        }
        else if(reg == "ElasticNet"){
            double reg = 0;
            for(int i = 0; i < weights.size(); i++){
                reg += alpha * abs(weights[i]); // Lasso Reg
                reg += ((1 - alpha) / 2) * weights[i] * weights[i]; // Ridge Reg
            }
            return reg * lambda;
        }
        return 0;
    }

    double Reg::regTerm(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg){
        if(reg == "Ridge"){
            double reg = 0;
            for(int i = 0; i < weights.size(); i++){
                for(int j = 0; j < weights[i].size(); j++){
                    reg += weights[i][j] * weights[i][j];
                }
            }
            return reg * lambda / 2;
        }
        else if(reg == "Lasso"){
            double reg = 0;
            for(int i = 0; i < weights.size(); i++){
                for(int j = 0; j < weights[i].size(); j++){
                    reg += abs(weights[i][j]);
                }
            }
            return reg * lambda;
        }
        else if(reg == "ElasticNet"){
            double reg = 0;
            for(int i = 0; i < weights.size(); i++){
                for(int j = 0; j < weights[i].size(); j++){
                    reg += alpha * abs(weights[i][j]); // Lasso Reg
                    reg += ((1 - alpha) / 2) * weights[i][j] * weights[i][j]; // Ridge Reg
                }
            }
            return reg * lambda;
        }
        return 0;
    }

    std::vector<double> Reg::regWeights(std::vector<double> weights, double lambda, double alpha, std::string reg){
        for(int i = 0; i < weights.size(); i++){
            weights[i] -= regDerivTerm(weights, lambda, alpha, reg, i);
        }
        return weights;
    }

    std::vector<std::vector<double>> Reg::regWeights(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg){
        for(int i = 0; i < weights.size(); i++){
            for(int j = 0; j < weights[i].size(); j++){
                weights[i][j] -= regDerivTerm(weights, lambda, alpha, reg, i, j);
            }
        }
        return weights;
    }

    double Reg::regDerivTerm(std::vector<double> weights, double lambda, double alpha, std::string reg, int j){
        if(reg == "Ridge"){
            return lambda * weights[j];
        }
        else if(reg == "Lasso"){
            return lambda * sign(weights[j]);
        }
        else if(reg == "ElasticNet"){
            return alpha * lambda * sign(weights[j]) + (1 - alpha) * lambda * weights[j];
        }
        else {
            return 0;
        }
    }

    double Reg::regDerivTerm(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg, int i, int j){
        if(reg == "Ridge"){
            return lambda * weights[i][j];
        }
        else if(reg == "Lasso"){
            return lambda * sign(weights[i][j]);
        }
        else if(reg == "ElasticNet"){
            return alpha * lambda * sign(weights[i][j]) + (1 - alpha) * lambda * weights[i][j];
        }
        else {
            return 0;
        }
    }

    int Reg::sign(double weight){
        if(weight < 0){
            return -1;
        }
        else if(weight == 0){
            return 0;
        }
        else{
            return 1;
        }
    }
}
