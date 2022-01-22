//
//  SVC.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "SVC.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Regularization/Reg.hpp"
#include "Utilities/Utilities.hpp"
#include "Cost/Cost.hpp"

#include <iostream>
#include <random>

namespace MLPP{
    SVC::SVC(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, double C)
    : inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), C(C)
    {
        y_hat.resize(n);
        weights = Utilities::weightInitialization(k);
        bias = Utilities::biasInitialization();
    }

    std::vector<double> SVC::modelSetTest(std::vector<std::vector<double>> X){
        return Evaluate(X);
    }

    double SVC::modelTest(std::vector<double> x){
        return Evaluate(x);
    }

    void SVC::gradientDescent(double learning_rate, int max_epoch, bool UI){
        class Cost cost;
        Activation avn;
        LinAlg alg;
        Reg regularization;
        double cost_prev = 0;
        int epoch = 1;
        forwardPass();
        
        while(true){
            cost_prev = Cost(y_hat, outputSet, weights, C);

            weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate/n, alg.mat_vec_mult(alg.transpose(inputSet), cost.HingeLossDeriv(z, outputSet, C))));
            weights = regularization.regWeights(weights, learning_rate/n, 0, "Ridge");

            // Calculating the bias gradients
            bias += learning_rate * alg.sum_elements(cost.HingeLossDeriv(y_hat, outputSet, C)) / n;
            
            forwardPass();
                
            // UI PORTION
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet, weights, C));
                Utilities::UI(weights, bias); 
            }
            epoch++;
            
            if(epoch > max_epoch) { break; }

        }
    }

    void SVC::SGD(double learning_rate, int max_epoch, bool UI){
        class Cost cost;
        Activation avn;
        LinAlg alg;
        Reg regularization;
        
        double cost_prev = 0;
        int epoch = 1;
        
        while(true){
            std::random_device rd;
            std::default_random_engine generator(rd()); 
            std::uniform_int_distribution<int> distribution(0, int(n - 1));
            int outputIndex = distribution(generator);

            double y_hat = Evaluate(inputSet[outputIndex]);
            double z = propagate(inputSet[outputIndex]);
            cost_prev = Cost({z}, {outputSet[outputIndex]}, weights, C);

            double costDeriv = cost.HingeLossDeriv(std::vector<double>({z}), std::vector<double>({outputSet[outputIndex]}), C)[0]; // Explicit conversion to avoid ambiguity with overloaded function. Error occured on Ubuntu.

            // Weight Updation
            weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate * costDeriv, inputSet[outputIndex]));
            weights = regularization.regWeights(weights, learning_rate, 0, "Ridge");
            
            // Bias updation
            bias -= learning_rate * costDeriv;

            y_hat = Evaluate({inputSet[outputIndex]});
                
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost({z}, {outputSet[outputIndex]}, weights, C));
                Utilities::UI(weights, bias); 
            }
            epoch++;
            
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void SVC::MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI){
        class Cost cost; 
        Activation avn;
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
                std::vector<double> z = propagate(inputMiniBatches[i]);
                cost_prev = Cost(z, outputMiniBatches[i], weights, C);

                // Calculating the weight gradients
                weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate/n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), cost.HingeLossDeriv(z, outputMiniBatches[i], C))));
                weights = regularization.regWeights(weights, learning_rate/n, 0, "Ridge");
                

                // Calculating the bias gradients
                bias -= learning_rate * alg.sum_elements(cost.HingeLossDeriv(y_hat, outputMiniBatches[i], C)) / n;
            
                forwardPass();

                y_hat = Evaluate(inputMiniBatches[i]);
                    
                if(UI) { 
                    Utilities::CostInfo(epoch, cost_prev, Cost(z, outputMiniBatches[i], weights, C));
                    Utilities::UI(weights, bias); 
                }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass(); 
    }

    double SVC::score(){
        Utilities util;
        return util.performance(y_hat, outputSet);
    }

     void SVC::save(std::string fileName){
         Utilities util;
         util.saveParameters(fileName, weights, bias);
     }

    double SVC::Cost(std::vector <double> z, std::vector<double> y, std::vector<double> weights, double C){
        class Cost cost; 
        return cost.HingeLoss(z, y, weights, C);    
    }

    std::vector<double> SVC::Evaluate(std::vector<std::vector<double>> X){
        LinAlg alg;
        Activation avn;
        return avn.sign(alg.scalarAdd(bias, alg.mat_vec_mult(X, weights))); 
    }
    
    std::vector<double>SVC::propagate(std::vector<std::vector<double>> X){
        LinAlg alg;
        Activation avn;
        return alg.scalarAdd(bias, alg.mat_vec_mult(X, weights)); 
    }

    double SVC::Evaluate(std::vector<double> x){
        LinAlg alg;
        Activation avn;
        return avn.sign(alg.dot(weights, x) + bias);
    }

    double SVC::propagate(std::vector<double> x){
        LinAlg alg;
        Activation avn;
        return alg.dot(weights, x) + bias;
    }

    // sign ( wTx + b )
    void SVC::forwardPass(){
        LinAlg alg;
        Activation avn;
        
        z = propagate(inputSet);
        y_hat = avn.sign(z);
    }
}