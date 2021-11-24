//
//  Reg.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include "Utilities.hpp"

namespace MLPP{

    std::vector<double> Utilities::weightInitialization(int n, std::string type){
        std::random_device rd;
        std::default_random_engine generator(rd()); 

        std::vector<double> weights; 
        for(int i = 0; i < n; i++){
            if(type == "XavierNormal"){
                std::normal_distribution<double> distribution(0, sqrt(2 / (n + 1)));
                weights.push_back(distribution(generator));
            }
            else if(type == "XavierUniform"){
                std::uniform_real_distribution<double> distribution(-sqrt(6 / (n + 1)), sqrt(6 / (n + 1)));
                weights.push_back(distribution(generator));
            }
            else if(type == "HeNormal"){
                std::normal_distribution<double> distribution(0, sqrt(2 / n));
                weights.push_back(distribution(generator));
            }
            else if(type == "HeUniform"){
                std::uniform_real_distribution<double> distribution(-sqrt(6 / n), sqrt(6 / n));
                weights.push_back(distribution(generator));
            }
            else if(type == "LeCunNormal"){
                std::normal_distribution<double> distribution(0, sqrt(1 / n));
                weights.push_back(distribution(generator));
            }
            else if(type == "LeCunUniform"){
                std::uniform_real_distribution<double> distribution(-sqrt(3/n), sqrt(3/n));
                weights.push_back(distribution(generator));
            }
            else if(type == "Uniform"){
                std::uniform_real_distribution<double> distribution(-1/sqrt(n), 1/sqrt(n));
                weights.push_back(distribution(generator));
            }
            else{
                std::uniform_real_distribution<double> distribution(0, 1);
                weights.push_back(distribution(generator));
            }
        }
        return weights;
    }

    double Utilities::biasInitialization(){
        std::random_device rd;
        std::default_random_engine generator(rd()); 
        std::uniform_real_distribution<double> distribution(0,1);

        return distribution(generator);
    }

    std::vector<std::vector<double>> Utilities::weightInitialization(int n, int m, std::string type){
        std::random_device rd;
        std::default_random_engine generator(rd()); 

        std::vector<std::vector<double>> weights; 
        weights.resize(n);

        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(type == "XavierNormal"){
                    std::normal_distribution<double> distribution(0, sqrt(2 / (n + m)));
                    weights[i].push_back(distribution(generator));
                }
                else if(type == "XavierUniform"){
                    std::uniform_real_distribution<double> distribution(-sqrt(6 / (n + m)), sqrt(6 / (n + m)));
                    weights[i].push_back(distribution(generator));
                }
                else if(type == "HeNormal"){
                    std::normal_distribution<double> distribution(0, sqrt(2 / n));
                    weights[i].push_back(distribution(generator));
                }
                else if(type == "HeUniform"){
                    std::uniform_real_distribution<double> distribution(-sqrt(6 / n), sqrt(6 / n));
                    weights[i].push_back(distribution(generator));
                }
                else if(type == "LeCunNormal"){
                    std::normal_distribution<double> distribution(0, sqrt(1 / n));
                    weights[i].push_back(distribution(generator));
                }
                else if(type == "LeCunUniform"){
                    std::uniform_real_distribution<double> distribution(-sqrt(3/n), sqrt(3/n));
                    weights[i].push_back(distribution(generator));
                }
                else if(type == "Uniform"){
                    std::uniform_real_distribution<double> distribution(-1/sqrt(n), 1/sqrt(n));
                    weights[i].push_back(distribution(generator));
                }
                else{
                    std::uniform_real_distribution<double> distribution(0, 1);
                    weights[i].push_back(distribution(generator));
                }
            }
        }
        return weights;
    }

    std::vector<double> Utilities::biasInitialization(int n){
        std::vector<double> bias; 
        std::random_device rd;
        std::default_random_engine generator(rd()); 
        std::uniform_real_distribution<double> distribution(0,1);

        for(int i = 0; i < n; i++){
          bias.push_back(distribution(generator));
        }
        return bias; 
    }

    double Utilities::performance(std::vector<double> y_hat, std::vector<double> outputSet){
        double correct = 0;
        for(int i = 0; i < y_hat.size(); i++){
            if(std::round(y_hat[i]) == outputSet[i]){
                correct++;
            }
        }
        return correct/y_hat.size();
    }

    double Utilities::performance(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y){
        double correct = 0;
        for(int i = 0; i < y_hat.size(); i++){
            int sub_correct = 0;
            for(int j = 0; j < y_hat[i].size(); j++){
                if(std::round(y_hat[i][j]) == y[i][j]){
                    sub_correct++;
                }
                if(sub_correct == y_hat[0].size()){
                    correct++;
                }
            }
        }
        return correct/y_hat.size();
    }

    void Utilities::saveParameters(std::string fileName, std::vector<double> weights, double bias, bool app, int layer){
        std::string layer_info = "";        
        std::ofstream saveFile;

        if(layer > -1){
            layer_info = " for layer " + std::to_string(layer);
        }

        if(app){
            saveFile.open(fileName.c_str(), std::ios_base::app); 
        }
        else { saveFile.open(fileName.c_str()); }

        if(!saveFile.is_open()){
            std::cout << fileName << " failed to open." << std::endl;
        }

        saveFile << "Weight(s)" << layer_info << std::endl;
        for(int i = 0; i < weights.size(); i++){
            saveFile << weights[i] << std::endl;
        }
        saveFile << "Bias" << layer_info << std::endl;
        saveFile << bias << std::endl;

        saveFile.close();
    }

    void Utilities::saveParameters(std::string fileName, std::vector<double> weights, std::vector<double> initial, double bias, bool app, int layer){
        std::string layer_info = "";        
        std::ofstream saveFile;

        if(layer > -1){
            layer_info = " for layer " + std::to_string(layer);
        }

        if(app){
            saveFile.open(fileName.c_str(), std::ios_base::app); 
        }
        else { saveFile.open(fileName.c_str()); }

        if(!saveFile.is_open()){
            std::cout << fileName << " failed to open." << std::endl;
        }

        saveFile << "Weight(s)" << layer_info << std::endl;
        for(int i = 0; i < weights.size(); i++){
            saveFile << weights[i] << std::endl;
        }

        saveFile << "Initial(s)" << layer_info << std::endl;
        for(int i = 0; i < initial.size(); i++){
            saveFile << initial[i] << std::endl;
        }

        saveFile << "Bias" << layer_info << std::endl;
        saveFile << bias << std::endl;

        saveFile.close();
    }

    void Utilities::saveParameters(std::string fileName, std::vector<std::vector<double>> weights, std::vector<double> bias, bool app, int layer){
        std::string layer_info = "";        
        std::ofstream saveFile;

        if(layer > -1){
            layer_info = " for layer " + std::to_string(layer);
        }

        if(app){
            saveFile.open(fileName.c_str(), std::ios_base::app); 
        }
        else { saveFile.open(fileName.c_str()); }

        if(!saveFile.is_open()){
            std::cout << fileName << " failed to open." << std::endl;
        }

        saveFile << "Weight(s)" << layer_info << std::endl;
        for(int i = 0; i < weights.size(); i++){
            for(int j = 0; j < weights[i].size(); j++){
                saveFile << weights[i][j] << std::endl;
            }
        }
        saveFile << "Bias(es)" << layer_info << std::endl;
        for(int i = 0; i < bias.size(); i++){
            saveFile << bias[i] << std::endl;
        }

        saveFile.close();
    }

    void Utilities::UI(std::vector<double> weights, double bias){
        std::cout << "Values of the weight(s):" << std::endl;
        for(int i = 0; i < weights.size(); i++){
            std::cout << weights[i] << std::endl;
        }
        std:: cout << "Value of the bias:" << std::endl;
        std::cout << bias << std::endl;
    }

    void Utilities::UI(std::vector<std::vector<double>> weights, std::vector<double> bias){
        std::cout << "Values of the weight(s):" << std::endl;
        for(int i = 0; i < weights.size(); i++){
            for(int j = 0; j < weights[i].size(); j++){
                std::cout << weights[i][j] << std::endl;
            }
        }
        std::cout << "Value of the biases:" << std::endl;
        for(int i = 0; i < bias.size(); i++){
            std::cout << bias[i] << std::endl;
        }
    }

    void Utilities::UI(std::vector<double> weights, std::vector<double> initial, double bias){
        std::cout << "Values of the weight(s):" << std::endl;
        for(int i = 0; i < weights.size(); i++){
            std::cout << weights[i] << std::endl;
        }
        std::cout << "Values of the initial(s):" << std::endl;
        for(int i = 0; i < initial.size(); i++){
            std::cout << initial[i] << std::endl;
        }
        std:: cout << "Value of the bias:" << std::endl;
        std::cout << bias << std::endl;
    }

    void Utilities::CostInfo(int epoch, double cost_prev, double Cost){
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "This is epoch: " << epoch << std::endl;
        std::cout << "The cost function has been minimized by " << cost_prev - Cost << std::endl;
        std::cout << "Current Cost:" << std::endl;
        std::cout << Cost << std::endl;
    }

    std::vector<std::vector<std::vector<double>>> Utilities::createMiniBatches(std::vector<std::vector<double>> inputSet, int n_mini_batch){
        int n = inputSet.size();
        
        std::vector<std::vector<std::vector<double>>> inputMiniBatches; 

        // Creating the mini-batches
        for(int i = 0; i < n_mini_batch; i++){
            std::vector<std::vector<double>> currentInputSet; 
            for(int j = 0; j < n/n_mini_batch; j++){
                currentInputSet.push_back(inputSet[n/n_mini_batch * i + j]);
            }
            inputMiniBatches.push_back(currentInputSet);
        }

        if(double(n)/double(n_mini_batch) - int(n/n_mini_batch) != 0){
            for(int i = 0; i < n - n/n_mini_batch * n_mini_batch; i++){
                inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n/n_mini_batch * n_mini_batch + i]);
            }
        }
        return inputMiniBatches;
    }

    std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> Utilities::createMiniBatches(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int n_mini_batch){
        int n = inputSet.size();
        
        std::vector<std::vector<std::vector<double>>> inputMiniBatches; 
        std::vector<std::vector<double>> outputMiniBatches; 

        for(int i = 0; i < n_mini_batch; i++){
            std::vector<std::vector<double>> currentInputSet; 
            std::vector<double> currentOutputSet; 
            for(int j = 0; j < n/n_mini_batch; j++){
                currentInputSet.push_back(inputSet[n/n_mini_batch * i + j]);
                currentOutputSet.push_back(outputSet[n/n_mini_batch * i + j]);
            }
            inputMiniBatches.push_back(currentInputSet);
            outputMiniBatches.push_back(currentOutputSet);
        }

        if(double(n)/double(n_mini_batch) - int(n/n_mini_batch) != 0){
            for(int i = 0; i < n - n/n_mini_batch * n_mini_batch; i++){
                inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n/n_mini_batch * n_mini_batch + i]);
                outputMiniBatches[n_mini_batch - 1].push_back(outputSet[n/n_mini_batch * n_mini_batch + i]);
            }
        }
        return {inputMiniBatches, outputMiniBatches};
    }

    std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>> Utilities::createMiniBatches(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet, int n_mini_batch){
        int n = inputSet.size();
        
        std::vector<std::vector<std::vector<double>>> inputMiniBatches; 
        std::vector<std::vector<std::vector<double>>> outputMiniBatches; 

        for(int i = 0; i < n_mini_batch; i++){
            std::vector<std::vector<double>> currentInputSet; 
            std::vector<std::vector<double>> currentOutputSet; 
            for(int j = 0; j < n/n_mini_batch; j++){
                currentInputSet.push_back(inputSet[n/n_mini_batch * i + j]);
                currentOutputSet.push_back(outputSet[n/n_mini_batch * i + j]);
            }
            inputMiniBatches.push_back(currentInputSet);
            outputMiniBatches.push_back(currentOutputSet);
        }

        if(double(n)/double(n_mini_batch) - int(n/n_mini_batch) != 0){
            for(int i = 0; i < n - n/n_mini_batch * n_mini_batch; i++){
                inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n/n_mini_batch * n_mini_batch + i]);
                outputMiniBatches[n_mini_batch - 1].push_back(outputSet[n/n_mini_batch * n_mini_batch + i]);
            }
        }
        return {inputMiniBatches, outputMiniBatches};
    }

    std::tuple<double, double, double, double> Utilities::TF_PN(std::vector<double> y_hat, std::vector<double> y){
        double TP, FP, TN, FN = 0;
        for(int i = 0; i < y_hat.size(); i++){
            if(y_hat[i] == y[i]){
                if(y_hat[i] == 1){
                    TP++;
                }
                else{
                    TN++;
                }
            }
            else{
                if(y_hat[i] == 1){
                    FP++;
                }
                else{
                    FN++;
                }
            }
        }
        return {TP, FP, TN, FN};
    }

    double Utilities::recall(std::vector<double> y_hat, std::vector<double> y){
        auto [TP, FP, TN, FN] = TF_PN(y_hat, y);
        return TP / (TP + FN);
    }

    double Utilities::precision(std::vector<double> y_hat, std::vector<double> y){
        auto [TP, FP, TN, FN] = TF_PN(y_hat, y);
        return TP / (TP + FP);
    }

    double Utilities::accuracy(std::vector<double> y_hat, std::vector<double> y){
        auto [TP, FP, TN, FN] = TF_PN(y_hat, y);
        return (TP + TN) / (TP + FP + FN + TN);
    }
    double Utilities::f1_score(std::vector<double> y_hat, std::vector<double> y){
        return 2 * precision(y_hat, y) * recall(y_hat, y) / (precision(y_hat, y) + recall(y_hat, y));
    }
}