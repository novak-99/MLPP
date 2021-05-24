//
//  ANN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "ANN.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Regularization/Reg.hpp"
#include "Utilities/Utilities.hpp"
#include "Cost/Cost.hpp"

#include <iostream>

namespace MLPP {
    ANN::ANN(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet)
    : inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size())
    {

    }

    ANN::~ANN(){
        delete outputLayer;
    }

    std::vector<double> ANN::modelSetTest(std::vector<std::vector<double>> X){
        network[0].input = X;
        network[0].forwardPass();

        for(int i = 1; i < network.size(); i++){
            network[i].input = network[i - 1].a;
            network[i].forwardPass();
        }
        outputLayer->input = network[network.size() - 1].a;
        outputLayer->forwardPass();
        return outputLayer->a;
    }

    double ANN::modelTest(std::vector<double> x){

        network[0].Test(x);
        for(int i = 1; i < network.size(); i++){
            network[i].Test(network[i - 1].a_test);
        }
        outputLayer->Test(network[network.size() - 1].a_test);
        return outputLayer->a_test;
    }

    void ANN::gradientDescent(double learning_rate, int max_epoch, bool UI){
        class Cost cost; 
        LinAlg alg;
        Activation avn;
        Reg regularization;

        double cost_prev = 0;
        int epoch = 1;
        forwardPass();

        while(true){
            cost_prev = Cost(y_hat, outputSet);
 
            auto costDeriv = outputLayer->costDeriv_map[outputLayer->cost];
            auto outputAvn = outputLayer->activation_map[outputLayer->activation];
            outputLayer->delta = alg.hadamard_product((cost.*costDeriv)(y_hat, outputSet), (avn.*outputAvn)(outputLayer->z, 1));
            std::vector<double> outputWGrad = alg.mat_vec_mult(alg.transpose(outputLayer->input), outputLayer->delta);

            outputLayer->weights = alg.subtraction(outputLayer->weights, alg.scalarMultiply(learning_rate/n, outputWGrad));
            outputLayer->weights = regularization.regWeights(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg);
            outputLayer->bias -= learning_rate * alg.sum_elements(outputLayer->delta) / n;

            auto hiddenLayerAvn = network[network.size() - 1].activation_map[network[network.size() - 1].activation];
            network[network.size() - 1].delta = alg.hadamard_product(alg.vecmult(outputLayer->delta, outputLayer->weights), (avn.*hiddenLayerAvn)(network[network.size() - 1].z, 1));
            std::vector<std::vector<double>> hiddenLayerWGrad = alg.matmult(alg.transpose(network[network.size() - 1].input), network[network.size() - 1].delta);
            
            network[network.size() - 1].weights = alg.subtraction(network[network.size() - 1].weights, alg.scalarMultiply(learning_rate/n, hiddenLayerWGrad));
            network[network.size() - 1].weights = regularization.regWeights(network[network.size() - 1].weights, network[network.size() - 1].lambda, network[network.size() - 1].alpha, network[network.size() - 1].reg);
            network[network.size() - 1].bias = alg.subtractMatrixRows(network[network.size() - 1].bias, alg.scalarMultiply(learning_rate/n, network[network.size() - 1].delta));

            for(int i = network.size() - 2; i >= 0; i--){
                auto hiddenLayerAvn = network[i].activation_map[network[i].activation];
                network[i].delta = alg.hadamard_product(alg.matmult(network[i + 1].delta, network[i + 1].weights), (avn.*hiddenLayerAvn)(network[i].z, 1));
                std::vector<std::vector<double>> hiddenLayerWGrad = alg.matmult(alg.transpose(network[i].input), network[i].delta);
                network[i].weights = alg.subtraction(network[i].weights, alg.scalarMultiply(learning_rate/n, hiddenLayerWGrad));
                network[i].weights = regularization.regWeights(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg);
                network[i].bias = alg.subtractMatrixRows(network[i].bias, alg.scalarMultiply(learning_rate/n, network[i].delta));
            }
            
            forwardPass();

            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
                std::cout << "Layer " << network.size() + 1 << ": " << std::endl;
                Utilities::UI(outputLayer->weights, outputLayer->bias); 
                std::cout << "Layer " << network.size() << ": " << std::endl;
                Utilities::UI(network[network.size() - 1].weights, network[network.size() - 1].bias); 
                for(int i = network.size() - 2; i >= 0; i--){
                    std::cout << "Layer " << i + 1 << ": " << std::endl;
                    Utilities::UI(network[i].weights, network[i].bias); 
                }
            }

            epoch++;
            if(epoch > max_epoch) { break; }
        }
    }

    double ANN::score(){
        Utilities util;
        forwardPass();
        return util.performance(y_hat, outputSet);
    }

    void ANN::save(std::string fileName){
        Utilities util;
        util.saveParameters(fileName, network[0].weights, network[0].bias, 0, 1);
        for(int i = 1; i < network.size(); i++){
            util.saveParameters(fileName, network[i].weights, network[i].bias, 1, i + 1); 
        }
        util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, 1, network.size() + 1);
     }

    void ANN::addLayer(int n_hidden, std::string activation, std::string weightInit, std::string reg, double lambda, double alpha){
        if(network.empty()){
            network.push_back(HiddenLayer(n_hidden, activation, inputSet, weightInit, reg, lambda, alpha));
            network[0].forwardPass();
        }
        else{
            network.push_back(HiddenLayer(n_hidden, activation, network[network.size() - 1].a, weightInit, reg, lambda, alpha));
            network[network.size() - 1].forwardPass();
        }
    }
    
    void ANN::addOutputLayer(std::string activation, std::string loss, std::string weightInit, std::string reg, double lambda, double alpha){
        outputLayer = new OutputLayer(network[0].n_hidden, outputSet.size(), activation, loss, network[network.size() - 1].a, weightInit, reg, lambda, alpha);
    }

    double ANN::Cost(std::vector<double> y_hat, std::vector<double> y){
        Reg regularization;
        class Cost cost;
        double totalRegTerm = 0;

        auto cost_function = outputLayer->cost_map[outputLayer->cost];
        for(int i = 0; i < network.size() - 1; i++){
            totalRegTerm += regularization.regTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg);
        }
        return (cost.*cost_function)(y_hat, y) + totalRegTerm + regularization.regTerm(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg);
    }

    void ANN::forwardPass(){
        network[0].input = inputSet;
        network[0].forwardPass();

        for(int i = 1; i < network.size(); i++){
            network[i].input = network[i - 1].a;
            network[i].forwardPass();
        }
        outputLayer->input = network[network.size() - 1].a;
        outputLayer->forwardPass();
        y_hat = outputLayer->a;
    }
}