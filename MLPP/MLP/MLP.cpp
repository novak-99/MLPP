//
//  MLP.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "MLP.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Regularization/Reg.hpp"
#include "Utilities/Utilities.hpp"
#include "Cost/Cost.hpp"

#include <iostream>
#include <random>

namespace MLPP {
    MLP::MLP(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int n_hidden, std::string reg, double lambda, double alpha)
    : inputSet(inputSet), outputSet(outputSet), n_hidden(n_hidden), n(inputSet.size()), k(inputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha)
    {
        Activation avn;
        y_hat.resize(n);

        weights1 = Utilities::weightInitialization(k, n_hidden);
        weights2 = Utilities::weightInitialization(n_hidden);
        bias1 = Utilities::biasInitialization(n_hidden);
        bias2 = Utilities::biasInitialization();
    }

    std::vector<double> MLP::modelSetTest(std::vector<std::vector<double>> X){
        return Evaluate(X);
    }

    double MLP::modelTest(std::vector<double> x){
        return Evaluate(x);
    }

    void MLP::gradientDescent(double learning_rate, int max_epoch, bool UI){
        Activation avn;
        LinAlg alg;
        Reg regularization;
        double cost_prev = 0;
        int epoch = 1;
        forwardPass();
        
        while(true){
            cost_prev = Cost(y_hat, outputSet);

            // Calculating the errors
            std::vector<double> error = alg.subtraction(y_hat, outputSet);
                    
            // Calculating the weight/bias gradients for layer 2

            std::vector<double> D2_1 = alg.mat_vec_mult(alg.transpose(a2), error);

            // weights and bias updation for layer 2
            weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate/n, D2_1));
            weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

            bias2 -= learning_rate * alg.sum_elements(error) / n;

            // Calculating the weight/bias for layer 1

            std::vector<std::vector<double>> D1_1;
            D1_1.resize(n);

            D1_1 = alg.outerProduct(error, weights2);

            std::vector<std::vector<double>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

            std::vector<std::vector<double>> D1_3 = alg.matmult(alg.transpose(inputSet), D1_2);


            // weight an bias updation for layer 1
            weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate/n, D1_3));
            weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

            bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate/n, D1_2));
    
            forwardPass();
                
            // UI PORTION
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
                std::cout << "Layer 1:" << std::endl;
                Utilities::UI(weights1, bias1); 
                std::cout << "Layer 2:" << std::endl;
                Utilities::UI(weights2, bias2);
            }
            epoch++;
                
            if(epoch > max_epoch) { break; }
        }

    }

    void MLP::SGD(double learning_rate, int max_epoch, bool UI){
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
            auto [z2, a2] = propagate(inputSet[outputIndex]);
            cost_prev = Cost({y_hat}, {outputSet[outputIndex]});
            double error = y_hat - outputSet[outputIndex];

            // Weight updation for layer 2
            std::vector<double> D2_1 = alg.scalarMultiply(error, a2);
            weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, D2_1));
            weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

            // Bias updation for layer 2
            bias2 -= learning_rate * error;

            // Weight updation for layer 1
            std::vector<double> D1_1 = alg.scalarMultiply(error, weights2);
            std::vector<double> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));
            std::vector<std::vector<double>> D1_3 = alg.outerProduct(inputSet[outputIndex], D1_2);

            weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate, D1_3));
            weights1 = regularization.regWeights(weights1, lambda, alpha, reg);
            // Bias updation for layer 1

            bias1 = alg.subtraction(bias1, alg.scalarMultiply(learning_rate, D1_2));

            y_hat = Evaluate(inputSet[outputIndex]);
            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost({y_hat}, {outputSet[outputIndex]}));
                std::cout << "Layer 1:" << std::endl;
                Utilities::UI(weights1, bias1); 
                std::cout << "Layer 2:" << std::endl;
                Utilities::UI(weights2, bias2);
            }
            epoch++;
            
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void MLP::MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI){
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
                auto [z2, a2] = propagate(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                // Calculating the errors
                std::vector<double> error = alg.subtraction(y_hat, outputMiniBatches[i]);
                        
                // Calculating the weight/bias gradients for layer 2

                std::vector<double> D2_1 = alg.mat_vec_mult(alg.transpose(a2), error);

                // weights and bias updation for layser 2
                weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate/outputMiniBatches[i].size(), D2_1));
                weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

                // Calculating the bias gradients for layer 2
                double b_gradient = alg.sum_elements(error);
                
                // Bias Updation for layer 2
                bias2 -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size();

                //Calculating the weight/bias for layer 1

                std::vector<std::vector<double>> D1_1 = alg.outerProduct(error, weights2);

                std::vector<std::vector<double>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

                std::vector<std::vector<double>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);


                // weight an bias updation for layer 1
                weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate/outputMiniBatches[i].size(), D1_3));
                weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

                bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate/outputMiniBatches[i].size(), D1_2));

                y_hat = Evaluate(inputMiniBatches[i]);
                    
                if(UI) { 
                    Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputMiniBatches[i]));
                    std::cout << "Layer 1:" << std::endl;
                    Utilities::UI(weights1, bias1); 
                    std::cout << "Layer 2:" << std::endl;
                    Utilities::UI(weights2, bias2);
                }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass(); 
    }

    double MLP::score(){
        Utilities util;
        return util.performance(y_hat, outputSet);
    }

    void MLP::save(std::string fileName){
         Utilities util;
         util.saveParameters(fileName, weights1, bias1, 0, 1);
         util.saveParameters(fileName, weights2, bias2, 1, 2);
     }

    double MLP::Cost(std::vector <double> y_hat, std::vector<double> y){
        Reg regularization;
        class Cost cost; 
        return cost.LogLoss(y_hat, y) + regularization.regTerm(weights2, lambda, alpha, reg) + regularization.regTerm(weights1, lambda, alpha, reg);
    }

    std::vector<double> MLP::Evaluate(std::vector<std::vector<double>> X){
        LinAlg alg;
        Activation avn;
        std::vector<std::vector<double>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
        std::vector<std::vector<double>> a2 = avn.sigmoid(z2);
        return avn.sigmoid(alg.scalarAdd(bias2, alg.mat_vec_mult(a2, weights2))); 
    }

    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> MLP::propagate(std::vector<std::vector<double>> X){
        LinAlg alg;
        Activation avn;
        std::vector<std::vector<double>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
        std::vector<std::vector<double>> a2 = avn.sigmoid(z2);
        return {z2, a2};
    }

    double MLP::Evaluate(std::vector<double> x){
        LinAlg alg;
        Activation avn;
        std::vector<double> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1); 
        std::vector<double> a2 = avn.sigmoid(z2);
        return avn.sigmoid(alg.dot(weights2, a2) + bias2);
    }

    std::tuple<std::vector<double>, std::vector<double>> MLP::propagate(std::vector<double> x){
        LinAlg alg;
        Activation avn;
        std::vector<double> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1); 
        std::vector<double> a2 = avn.sigmoid(z2);
        return {z2, a2};
    }

    void MLP::forwardPass(){
        LinAlg alg;
        Activation avn;
        z2 = alg.mat_vec_add(alg.matmult(inputSet, weights1), bias1);
        a2 = avn.sigmoid(z2);
        y_hat = avn.sigmoid(alg.scalarAdd(bias2, alg.mat_vec_mult(a2, weights2))); 
    }
}
