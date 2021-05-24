//
//  CLogLogReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "CLogLogReg.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Regularization/Reg.hpp"
#include "Utilities/Utilities.hpp"
#include "Cost/Cost.hpp"

#include <iostream>
#include <random>

namespace MLPP{
    CLogLogReg::CLogLogReg(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, std::string reg, double lambda, double alpha)
    : inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha)
    {
        y_hat.resize(n);
        weights = Utilities::weightInitialization(k);
        bias = Utilities::biasInitialization();
    }

    std::vector<double> CLogLogReg::modelSetTest(std::vector<std::vector<double>> X){
        return Evaluate(X);
    }

    double CLogLogReg::modelTest(std::vector<double> x){
        return Evaluate(x);
    }

    void CLogLogReg::gradientDescent(double learning_rate, int max_epoch, bool UI){
        Reg regularization;
        LinAlg alg;
        Activation avn;
        double cost_prev = 0;
        int epoch = 1;
        forwardPass();
        
        while(true){
            cost_prev = Cost(y_hat, outputSet);
                
            std::vector<double> error = alg.subtraction(y_hat, outputSet);


            // Calculating the weight gradients
            weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate/n, alg.mat_vec_mult(alg.transpose(inputSet), alg.hadamard_product(error, avn.cloglog(z, 1)))));
            weights = regularization.regWeights(weights, lambda, alpha, reg);

            // Calculating the bias gradients
            bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(z, 1))) / n;
            
            forwardPass();
                
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
                Utilities::UI(weights, bias); 
            }
            epoch++;
            
            if(epoch > max_epoch) { break; }
        }
    }

    void CLogLogReg::MLE(double learning_rate, int max_epoch, bool UI){
        Reg regularization; 
        Activation avn;
        LinAlg alg;
        double cost_prev = 0;
        int epoch = 1;
        forwardPass();
        
        while(true){
            cost_prev = Cost(y_hat, outputSet);

            std::vector<double> error = alg.subtraction(y_hat, outputSet);            

            weights = alg.addition(weights, alg.scalarMultiply(learning_rate/n, alg.mat_vec_mult(alg.transpose(inputSet), alg.hadamard_product(error, avn.cloglog(z, 1)))));
            weights = regularization.regWeights(weights, lambda, alpha, reg);

            // Calculating the bias gradients
            bias += learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(z, 1))) / n;
            forwardPass();
                
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
                Utilities::UI(weights, bias); 
            }
            epoch++;
            
            if(epoch > max_epoch) { break; }
                
        }
    }

    void CLogLogReg::SGD(double learning_rate, int max_epoch, bool UI){
        Reg regularization; 
        double cost_prev = 0;
        int epoch = 1;
        forwardPass();
        
        while(true){
            std::random_device rd;
            std::default_random_engine generator(rd()); 
            std::uniform_int_distribution<int> distribution(0, int(n - 1));
            int outputIndex = distribution(generator);

            double y_hat = Evaluate(inputSet[outputIndex]);
            double z = propagate(inputSet[outputIndex]);

            cost_prev = Cost({y_hat}, {outputSet[outputIndex]});
                
            for(int i = 0; i < k; i++){
                    
                // Calculating the weight gradients
                double w_gradient = (y_hat - outputSet[outputIndex]) * exp(z-exp(z)) * inputSet[outputIndex][i];
                    

                // Weight updation
                weights[i] -= learning_rate * w_gradient;
            }
            weights = regularization.regWeights(weights, lambda, alpha, reg);
            
            
            // Calculating the bias gradients
            double b_gradient = (y_hat - outputSet[outputIndex]) * exp(z-exp(z));
            
            // Bias updation
            bias -= learning_rate * b_gradient;
            y_hat = Evaluate({inputSet[outputIndex]});
                
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost({y_hat}, {outputSet[outputIndex]}));
                Utilities::UI(weights, bias); 
            }
            epoch++;
                
           if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void CLogLogReg::MBGD(double learning_rate, int max_epoch, int miniBatch_size, bool UI){
        Reg regularization;
        Activation avn;
        LinAlg alg;
        double cost_prev = 0;
        int epoch = 1;

        int n_miniBatch = n/miniBatch_size;
        
        std::vector<std::vector<std::vector<double>>> inputMiniBatches; 
        std::vector<std::vector<double>> outputMiniBatches; 
        // Creating the mini-batches
        for(int i = 0; i < n_miniBatch; i++){
            std::vector<std::vector<double>> currentInputSet; 
            std::vector<double> currentOutputSet; 
            std::vector<double> currentPreActivationSet; 
            for(int j = 0; j < n/n_miniBatch; j++){
                currentInputSet.push_back(inputSet[n/n_miniBatch * i + j]);
                currentOutputSet.push_back(outputSet[n/n_miniBatch * i + j]);
            }
            inputMiniBatches.push_back(currentInputSet);
            outputMiniBatches.push_back(currentOutputSet);
        }

        if(double(n)/double(n_miniBatch) - int(n/n_miniBatch) != 0){
            for(int i = 0; i < n - n/n_miniBatch * n_miniBatch; i++){
                inputMiniBatches[n_miniBatch - 1].push_back(inputSet[n/n_miniBatch * n_miniBatch + i]);
                outputMiniBatches[n_miniBatch - 1].push_back(outputSet[n/n_miniBatch * n_miniBatch + i]);
            }
        }
        
        while(true){
            for(int i = 0; i < n_miniBatch; i++){
                std::vector<double> y_hat = Evaluate(inputMiniBatches[i]);
                std::vector<double> z = propagate(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                std::vector<double> error = alg.subtraction(y_hat, outputMiniBatches[i]);

                // Calculating the weight gradients
                weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate/n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), alg.hadamard_product(error, avn.cloglog(z, 1)))));
                weights = regularization.regWeights(weights, lambda, alpha, reg);

                // Calculating the bias gradients
                bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(z, 1))) / n;
            
                forwardPass();

                y_hat = Evaluate(inputMiniBatches[i]);
                    
                if(UI) { 
                    Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputMiniBatches[i]));
                    Utilities::UI(weights, bias); 
                }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass(); 
    }

    double CLogLogReg::score(){
        Utilities util;
        return util.performance(y_hat, outputSet);
    }

    double CLogLogReg::Cost(std::vector <double> y_hat, std::vector<double> y){
        Reg regularization;
        class Cost cost; 
        return cost.MSE(y_hat, y) + regularization.regTerm(weights, lambda, alpha, reg);
    }

    std::vector<double> CLogLogReg::Evaluate(std::vector<std::vector<double>> X){
        LinAlg alg;
        Activation avn;
        return avn.cloglog(alg.scalarAdd(bias, alg.mat_vec_mult(X, weights))); 
    }
    
    std::vector<double>CLogLogReg::propagate(std::vector<std::vector<double>> X){
        LinAlg alg;
        return alg.scalarAdd(bias, alg.mat_vec_mult(X, weights)); 
    }

    double CLogLogReg::Evaluate(std::vector<double> x){
        LinAlg alg;
        Activation avn;
        return avn.cloglog(alg.dot(weights, x) + bias);
    }

    double CLogLogReg::propagate(std::vector<double> x){
        LinAlg alg;
        return alg.dot(weights, x) + bias;
    }

    // cloglog ( wTx + b )
    void CLogLogReg::forwardPass(){
        LinAlg alg;
        Activation avn;
        
        z = propagate(inputSet);
        y_hat = avn.cloglog(z);
    }
}