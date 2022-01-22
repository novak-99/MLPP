//
//  DualSVC.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "DualSVC.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Regularization/Reg.hpp"
#include "Utilities/Utilities.hpp"
#include "Cost/Cost.hpp"

#include <iostream>
#include <random>

namespace MLPP{
    DualSVC::DualSVC(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, double C, std::string kernel)
    : inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), C(C), kernel(kernel)
    {
        y_hat.resize(n);
        bias = Utilities::biasInitialization();
        alpha = Utilities::weightInitialization(n); // One alpha for all training examples, as per the lagrangian multipliers.
        K = kernelFunction(inputSet, inputSet, kernel); // For now this is unused. When non-linear kernels are added, the K will be manipulated.
    }

    std::vector<double> DualSVC::modelSetTest(std::vector<std::vector<double>> X){
        return Evaluate(X);
    }

    double DualSVC::modelTest(std::vector<double> x){
        return Evaluate(x);
    }

    void DualSVC::gradientDescent(double learning_rate, int max_epoch, bool UI){
        class Cost cost;
        Activation avn;
        LinAlg alg;
        Reg regularization;
        double cost_prev = 0;
        int epoch = 1;
        forwardPass();
        
        while(true){
            cost_prev = Cost(alpha, inputSet, outputSet);

            alpha = alg.subtraction(alpha, alg.scalarMultiply(learning_rate, cost.dualFormSVMDeriv(alpha, inputSet, outputSet)));

            alphaProjection();

            // Calculating the bias 
            double biasGradient = 0; 
            for(int i = 0; i < alpha.size(); i++){
                double sum = 0;
                if(alpha[i] < C && alpha[i] > 0){
                    for(int j = 0; j < alpha.size(); j++){
                        if(alpha[j] > 0){  
                            sum += alpha[j] * outputSet[j] * alg.dot(inputSet[j], inputSet[i]); // TO DO: DON'T forget to add non-linear kernelizations. 
                        }
                    }
                }
                biasGradient = (1 - outputSet[i] * sum) / outputSet[i];
                break;
            }
            bias -= biasGradient * learning_rate;
            
            forwardPass();
                
            // UI PORTION
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost(alpha, inputSet, outputSet));
                Utilities::UI(alpha, bias);
                std::cout << score() << std::endl; // TO DO: DELETE THIS. 
            }
            epoch++;
            
            if(epoch > max_epoch) { break; }

        }
    }

    // void DualSVC::SGD(double learning_rate, int max_epoch, bool UI){
    //     class Cost cost;
    //     Activation avn;
    //     LinAlg alg;
    //     Reg regularization;
        
    //     double cost_prev = 0;
    //     int epoch = 1;
        
    //     while(true){
    //         std::random_device rd;
    //         std::default_random_engine generator(rd()); 
    //         std::uniform_int_distribution<int> distribution(0, int(n - 1));
    //         int outputIndex = distribution(generator);

    //         cost_prev = Cost(alpha, inputSet[outputIndex], outputSet[outputIndex]);
            
    //         // Bias updation
    //         bias -= learning_rate * costDeriv;

    //         y_hat = Evaluate({inputSet[outputIndex]});
                
    //         if(UI) { 
    //             Utilities::CostInfo(epoch, cost_prev, Cost(alpha));
    //             Utilities::UI(weights, bias); 
    //         }
    //         epoch++;
            
    //         if(epoch > max_epoch) { break; }
    //     }
    //     forwardPass();
    // }

    // void DualSVC::MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI){
    //     class Cost cost; 
    //     Activation avn;
    //     LinAlg alg;
    //     Reg regularization;
    //     double cost_prev = 0;
    //     int epoch = 1;
        
    //     // Creating the mini-batches
    //     int n_mini_batch = n/mini_batch_size;
    //     auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

    //     while(true){
    //         for(int i = 0; i < n_mini_batch; i++){
    //             std::vector<double> y_hat = Evaluate(inputMiniBatches[i]);
    //             std::vector<double> z = propagate(inputMiniBatches[i]);
    //             cost_prev = Cost(z, outputMiniBatches[i], weights, C);

    //             // Calculating the weight gradients
    //             weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate/n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), cost.HingeLossDeriv(z, outputMiniBatches[i], C))));
    //             weights = regularization.regWeights(weights, learning_rate/n, 0, "Ridge");
                

    //             // Calculating the bias gradients
    //             bias -= learning_rate * alg.sum_elements(cost.HingeLossDeriv(y_hat, outputMiniBatches[i], C)) / n;
            
    //             forwardPass();

    //             y_hat = Evaluate(inputMiniBatches[i]);
                    
    //             if(UI) { 
    //                 Utilities::CostInfo(epoch, cost_prev, Cost(z, outputMiniBatches[i], weights, C));
    //                 Utilities::UI(weights, bias); 
    //             }
    //         }
    //         epoch++;
    //         if(epoch > max_epoch) { break; }
    //     }
    //     forwardPass(); 
    // }

    double DualSVC::score(){
        Utilities util;
        return util.performance(y_hat, outputSet);
    }

     void DualSVC::save(std::string fileName){
         Utilities util;
         util.saveParameters(fileName, alpha, bias);
     }

    double DualSVC::Cost(std::vector<double> alpha, std::vector<std::vector<double>> X, std::vector<double> y){
        class Cost cost; 
        return cost.dualFormSVM(alpha, X, y);    
    }

    std::vector<double> DualSVC::Evaluate(std::vector<std::vector<double>> X){
        Activation avn;
        return avn.sign(propagate(X)); 
    }
    
    std::vector<double> DualSVC::propagate(std::vector<std::vector<double>> X){
        LinAlg alg; 
        std::vector<double> z; 
        for(int i = 0; i < X.size(); i++){
            double sum = 0;
            for(int j = 0; j < alpha.size(); j++){
                if(alpha[j] != 0){
                    sum += alpha[j] * outputSet[j] * alg.dot(inputSet[j], X[i]); // TO DO: DON'T forget to add non-linear kernelizations. 
                }
            }
            sum += bias; 
            z.push_back(sum);
        }
        return z; 
    }

    double DualSVC::Evaluate(std::vector<double> x){
        Activation avn;
        return avn.sign(propagate(x));
    }

    double DualSVC::propagate(std::vector<double> x){
        LinAlg alg;
        double z = 0;
        for(int j = 0; j < alpha.size(); j++){
            if(alpha[j] != 0){
                z += alpha[j] * outputSet[j] * alg.dot(inputSet[j], x); // TO DO: DON'T forget to add non-linear kernelizations. 
            }
        }
        z += bias; 
        return z; 
    }

    void DualSVC::forwardPass(){
        LinAlg alg;
        Activation avn;
        
        z = propagate(inputSet);
        y_hat = avn.sign(z);
    }

    void DualSVC::alphaProjection(){
        for(int i = 0; i < alpha.size(); i++){
            if(alpha[i] > C){
                alpha[i] = C;
            }
            else if(alpha[i] < 0){
                alpha[i] = 0;
            }
        }
    }

    double DualSVC::kernelFunction(std::vector<double> u, std::vector<double> v, std::string kernel){
        LinAlg alg;
        if(kernel == "Linear"){
            return alg.dot(u, v);
        } // warning: non-void function does not return a value in all control paths [-Wreturn-type]
    }

    std::vector<std::vector<double>> DualSVC::kernelFunction(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B, std::string kernel){
        LinAlg alg;
        if(kernel == "Linear"){
            return alg.matmult(inputSet, alg.transpose(inputSet));
        } // warning: non-void function does not return a value in all control paths [-Wreturn-type]
    }
}