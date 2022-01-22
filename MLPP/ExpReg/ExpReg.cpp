//
//  ExpReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "ExpReg.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Stat/Stat.hpp"
#include "Regularization/Reg.hpp"
#include "Utilities/Utilities.hpp"
#include "Cost/Cost.hpp"

#include <iostream>
#include <random>

namespace MLPP{
    ExpReg::ExpReg(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, std::string reg, double lambda, double alpha)
    : inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha)
    {
        y_hat.resize(n);
        weights = Utilities::weightInitialization(k);
        initial = Utilities::weightInitialization(k);
        bias = Utilities::biasInitialization();
    }

    std::vector<double> ExpReg::modelSetTest(std::vector<std::vector<double>> X){
        return Evaluate(X);
    }

    double ExpReg::modelTest(std::vector<double> x){
        return Evaluate(x);
    }

    void ExpReg::gradientDescent(double learning_rate, int max_epoch, bool UI){
        LinAlg alg;
        Reg regularization;
        double cost_prev = 0;
        int epoch = 1;
        forwardPass();
        
        while(true){
            cost_prev = Cost(y_hat, outputSet);

            std::vector<double> error = alg.subtraction(y_hat, outputSet);

            for(int i = 0; i < k; i++){
            
                // Calculating the weight gradient
                double sum = 0;
                for(int j = 0; j < n; j++){
                    sum += error[j] * inputSet[j][i] * std::pow(weights[i], inputSet[j][i] - 1);
                }
                double w_gradient = sum / n;
                    
                // Calculating the initial gradient
                double sum2 = 0;
                for(int j = 0; j < n; j++){
                    sum2 += error[j] * std::pow(weights[i], inputSet[j][i]);
                }


                double i_gradient = sum2 / n;
                
                // Weight/initial updation
                weights[i] -= learning_rate * w_gradient;
                initial[i] -= learning_rate * i_gradient;
                    
            }
            weights = regularization.regWeights(weights, lambda, alpha, reg);
                
            // Calculating the bias gradient
            double sum = 0;
            for(int j = 0; j < n; j++){
                sum += (y_hat[j] - outputSet[j]);
            }
            double b_gradient = sum / n;
                
            // bias updation
            bias -= learning_rate * b_gradient;
            forwardPass();
            
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
                Utilities::UI(weights, bias); 
            }
            epoch++;
                
            if(epoch > max_epoch) { break; }
                
        }
    }

    void ExpReg::SGD(double learning_rate, int max_epoch, bool UI){
        Reg regularization;
        double cost_prev = 0;
        int epoch = 1;
        
        while(true){
            std::random_device rd;
            std::default_random_engine generator(rd());
            std::uniform_int_distribution<int> distribution(0, int(n - 1));
            int outputIndex = distribution(generator);

            double y_hat = Evaluate(inputSet[outputIndex]);
            cost_prev = Cost({y_hat}, {outputSet[outputIndex]});

                
            for(int i = 0; i < k; i++){
                    
                // Calculating the weight gradients
                
                double w_gradient = (y_hat - outputSet[outputIndex]) * inputSet[outputIndex][i] * std::pow(weights[i], inputSet[outputIndex][i] - 1);
                double i_gradient = (y_hat - outputSet[outputIndex]) * std::pow(weights[i], inputSet[outputIndex][i]);

                // Weight/initial updation
                weights[i] -= learning_rate * w_gradient;
                initial[i] -= learning_rate * i_gradient;
            }
            weights = regularization.regWeights(weights, lambda, alpha, reg);
            
            // Calculating the bias gradients
            double b_gradient = (y_hat - outputSet[outputIndex]);
            
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

    void ExpReg::MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI){
        LinAlg alg;
        Reg regularization;
        double cost_prev = 0;
        int epoch = 1;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        
        while(true){
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = Evaluate(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);
                std::vector<double> error = alg.subtraction(y_hat, outputMiniBatches[i]);

                for(int j = 0; j < k; j++){
                    // Calculating the weight gradient
                    double sum = 0;
                    for(int k = 0; k < outputMiniBatches[i].size(); k++){
                        sum += error[k] * inputMiniBatches[i][k][j] * std::pow(weights[j], inputMiniBatches[i][k][j] - 1);
                    }
                    double w_gradient = sum / outputMiniBatches[i].size();
                        
                    // Calculating the initial gradient
                    double sum2 = 0;
                    for(int k = 0; k < outputMiniBatches[i].size(); k++){
                        sum2 += error[k] * std::pow(weights[j], inputMiniBatches[i][k][j]);
                    }


                    double i_gradient = sum2 / outputMiniBatches[i].size();
                    
                    // Weight/initial updation
                    weights[j] -= learning_rate * w_gradient;
                    initial[j] -= learning_rate * i_gradient;
                }   
                weights = regularization.regWeights(weights, lambda, alpha, reg);
                    
                // Calculating the bias gradient
                double sum = 0;
                for(int j = 0; j < outputMiniBatches[i].size(); j++){
                    sum += (y_hat[j] - outputMiniBatches[i][j]);
                }
                double b_gradient = sum / outputMiniBatches[i].size();
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

    double ExpReg::score(){
        Utilities util;
        return util.performance(y_hat, outputSet);
    }

    void ExpReg::save(std::string fileName){
         Utilities util;
         util.saveParameters(fileName, weights, initial, bias);
     }

    double ExpReg::Cost(std::vector <double> y_hat, std::vector<double> y){
        Reg regularization;
        class Cost cost; 
        return cost.MSE(y_hat, y) + regularization.regTerm(weights, lambda, alpha, reg);
    }

    std::vector<double> ExpReg::Evaluate(std::vector<std::vector<double>> X){
        std::vector<double> y_hat;
        y_hat.resize(X.size());
        for(int i = 0; i < X.size(); i++){
            y_hat[i] = 0;
            for(int j = 0; j < X[i].size(); j++){
                y_hat[i] += initial[j] * std::pow(weights[j], X[i][j]);
            }
            y_hat[i] += bias;
        }
        return y_hat;
    }

    double ExpReg::Evaluate(std::vector<double> x){
        double y_hat = 0;
        for(int i = 0; i < x.size(); i++){
            y_hat += initial[i] * std::pow(weights[i], x[i]);
        }
        
        return y_hat + bias;
    }

    // a * w^x + b
    void ExpReg::forwardPass(){
        y_hat = Evaluate(inputSet); 
    }
}