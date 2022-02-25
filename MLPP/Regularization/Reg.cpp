//
//  Reg.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include <iostream>
#include <random>
#include "Reg.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Activation/Activation.hpp"

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
        LinAlg alg;
        if(reg == "WeightClipping"){ return regDerivTerm(weights, lambda, alpha, reg); }
        return alg.subtraction(weights, regDerivTerm(weights, lambda, alpha, reg));
        // for(int i = 0; i < weights.size(); i++){
        //     weights[i] -= regDerivTerm(weights, lambda, alpha, reg, i);
        // }
        // return weights;
    }

    std::vector<std::vector<double>> Reg::regWeights(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg){
        LinAlg alg;
        if(reg == "WeightClipping"){ return regDerivTerm(weights, lambda, alpha, reg); }
        return alg.subtraction(weights, regDerivTerm(weights, lambda, alpha, reg));
        // for(int i = 0; i < weights.size(); i++){
        //     for(int j = 0; j < weights[i].size(); j++){
        //         weights[i][j] -= regDerivTerm(weights, lambda, alpha, reg, i, j);
        //     }
        // }
        // return weights;
    }

    std::vector<double> Reg::regDerivTerm(std::vector<double> weights, double lambda, double alpha, std::string reg){
        std::vector<double> regDeriv; 
        regDeriv.resize(weights.size());

        for(int i = 0; i < regDeriv.size(); i++){
            regDeriv[i] = regDerivTerm(weights, lambda, alpha, reg, i);
        }
        return regDeriv;
    }

    std::vector<std::vector<double>> Reg::regDerivTerm(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg){
        std::vector<std::vector<double>> regDeriv; 
        regDeriv.resize(weights.size());
        for(int i = 0; i < regDeriv.size(); i++){
            regDeriv[i].resize(weights[0].size());
        }

        for(int i = 0; i < regDeriv.size(); i++){
            for(int j = 0; j < regDeriv[i].size(); j++){
                regDeriv[i][j] = regDerivTerm(weights, lambda, alpha, reg, i, j);
            }
        }
        return regDeriv;
    }

    double Reg::regDerivTerm(std::vector<double> weights, double lambda, double alpha, std::string reg, int j){
        Activation act;
        if(reg == "Ridge"){
            return lambda * weights[j];
        }
        else if(reg == "Lasso"){
            return lambda * act.sign(weights[j]);
        }
        else if(reg == "ElasticNet"){
            return alpha * lambda * act.sign(weights[j]) + (1 - alpha) * lambda * weights[j];
        }
        else if(reg == "WeightClipping"){ // Preparation for Wasserstein GANs. 
            // We assume lambda is the lower clipping threshold, while alpha is the higher clipping threshold. 
            // alpha > lambda. 
            if(weights[j] > alpha){
                return alpha;
            }
            else if(weights[j] < lambda){
                return lambda;
            }
            else{
                return weights[j];
            }
        }
        else {
            return 0;
        }
    }

    double Reg::regDerivTerm(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg, int i, int j){
        Activation act;
        if(reg == "Ridge"){
            return lambda * weights[i][j];
        }
        else if(reg == "Lasso"){
            return lambda * act.sign(weights[i][j]);
        }
        else if(reg == "ElasticNet"){
            return alpha * lambda * act.sign(weights[i][j]) + (1 - alpha) * lambda * weights[i][j];
        }
        else if(reg == "WeightClipping"){ // Preparation for Wasserstein GANs.
            // We assume lambda is the lower clipping threshold, while alpha is the higher clipping threshold. 
            // alpha > lambda. 
            if(weights[i][j] > alpha){
                return alpha;
            }
            else if(weights[i][j] < lambda){
               return lambda;
            }
            else{
                return weights[i][j];
            }
        }
        else {
            return 0;
        }
    }
}
