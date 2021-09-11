//
//  MANN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "MANN.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Regularization/Reg.hpp"
#include "Utilities/Utilities.hpp"
#include "Cost/Cost.hpp"

#include <iostream>

namespace MLPP {
    MANN::MANN(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet)
    : inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), n_output(outputSet[0].size())
    {

    }

    MANN::~MANN(){
        delete outputLayer;
    }

    std::vector<std::vector<double>> MANN::modelSetTest(std::vector<std::vector<double>> X){
        if(!network.empty()){
            network[0].input = X;
            network[0].forwardPass();

            for(int i = 1; i < network.size(); i++){
                network[i].input = network[i - 1].a;
                network[i].forwardPass();
            }
            outputLayer->input = network[network.size() - 1].a;
        }
        else {
            outputLayer->input = X;
        }
        outputLayer->forwardPass();
        return outputLayer->a;
    }

    std::vector<double> MANN::modelTest(std::vector<double> x){
        if(!network.empty()){
            network[0].Test(x);
            for(int i = 1; i < network.size(); i++){
                network[i].Test(network[i - 1].a_test);
            }
            outputLayer->Test(network[network.size() - 1].a_test);
        }
        else{
            outputLayer->Test(x);
        }
        return outputLayer->a_test;
    }

    void MANN::gradientDescent(double learning_rate, int max_epoch, bool UI){
        class Cost cost; 
        Activation avn;
        LinAlg alg;
        Reg regularization;

        double cost_prev = 0;
        int epoch = 1;
        forwardPass();

        while(true){
            cost_prev = Cost(y_hat, outputSet);
 
            if(outputLayer->activation == "Softmax"){
                outputLayer->delta = alg.subtraction(y_hat, outputSet);
            }
            else{
                auto costDeriv = outputLayer->costDeriv_map[outputLayer->cost];
                auto outputAvn = outputLayer->activation_map[outputLayer->activation];
                outputLayer->delta = alg.hadamard_product((cost.*costDeriv)(y_hat, outputSet), (avn.*outputAvn)(outputLayer->z, 1));
            }

            std::vector<std::vector<double>> outputWGrad = alg.matmult(alg.transpose(outputLayer->input), outputLayer->delta);

            outputLayer->weights = alg.subtraction(outputLayer->weights, alg.scalarMultiply(learning_rate/n, outputWGrad));
            outputLayer->weights = regularization.regWeights(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg);
            outputLayer->bias = alg.subtractMatrixRows(outputLayer->bias, alg.scalarMultiply(learning_rate/n, outputLayer->delta));

            if(!network.empty()){
                auto hiddenLayerAvn = network[network.size() - 1].activation_map[network[network.size() - 1].activation];
                network[network.size() - 1].delta = alg.hadamard_product(alg.matmult(outputLayer->delta, alg.transpose(outputLayer->weights)), (avn.*hiddenLayerAvn)(network[network.size() - 1].z, 1));
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
            }
            
            forwardPass();

            if(UI) { 
                Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
                std::cout << "Layer " << network.size() + 1 << ": " << std::endl;
                Utilities::UI(outputLayer->weights, outputLayer->bias); 
                if(!network.empty()){
                    std::cout << "Layer " << network.size() << ": " << std::endl; 
                    for(int i = network.size() - 1; i >= 0; i--){
                        std::cout << "Layer " << i + 1 << ": " << std::endl;
                        Utilities::UI(network[i].weights, network[i].bias); 
                    }
                }
            }

            epoch++;
            if(epoch > max_epoch) { break; }
        }
    }

    double MANN::score(){
        Utilities util;
        forwardPass();
        return util.performance(y_hat, outputSet);
    }

    void MANN::save(std::string fileName){
        Utilities util;
        if(!network.empty()){
            util.saveParameters(fileName, network[0].weights, network[0].bias, 0, 1);
            for(int i = 1; i < network.size(); i++){
                util.saveParameters(fileName, network[i].weights, network[i].bias, 1, i + 1); 
            }
            util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, 1, network.size() + 1);
        }
        else{
            util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, 0, network.size() + 1);
        }
     }

    void MANN::addLayer(int n_hidden, std::string activation, std::string weightInit, std::string reg, double lambda, double alpha){
        if(network.empty()){
            network.push_back(HiddenLayer(n_hidden, activation, inputSet, weightInit, reg, lambda, alpha));
            network[0].forwardPass();
        }
        else{
            network.push_back(HiddenLayer(n_hidden, activation, network[network.size() - 1].a, weightInit, reg, lambda, alpha));
            network[network.size() - 1].forwardPass();
        }
    }
    
    void MANN::addOutputLayer(std::string activation, std::string loss, std::string weightInit, std::string reg, double lambda, double alpha){
        if(!network.empty()){
            outputLayer = new MultiOutputLayer(n_output, network[0].n_hidden, activation, loss, network[network.size() - 1].a, weightInit, reg, lambda, alpha);
        }
        else{
            outputLayer = new MultiOutputLayer(n_output, k, activation, loss, inputSet, weightInit, reg, lambda, alpha);
        }
    }

    double MANN::Cost(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y){
        Reg regularization;
        class Cost cost;
        double totalRegTerm = 0;

        auto cost_function = outputLayer->cost_map[outputLayer->cost];
        if(!network.empty()){
            for(int i = 0; i < network.size() - 1; i++){
                totalRegTerm += regularization.regTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg);
            }
        }
        return (cost.*cost_function)(y_hat, y) + totalRegTerm + regularization.regTerm(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg);
    }

    void MANN::forwardPass(){
        if(!network.empty()){
            network[0].input = inputSet;
            network[0].forwardPass();

            for(int i = 1; i < network.size(); i++){
                network[i].input = network[i - 1].a;
                network[i].forwardPass();
            }
            outputLayer->input = network[network.size() - 1].a;
        }
        else{
            outputLayer->input = inputSet;
        }
        outputLayer->forwardPass();
        y_hat = outputLayer->a;
    }
}