//
//  Convolutions.cpp
//
//  Created by Marc Melikyan on 4/6/21.
//

#include <iostream>
#include "Convolutions/Convolutions.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Stat/Stat.hpp"

namespace MLPP{

    Convolutions::Convolutions()
    : prewittHorizontal({{1,1,1}, {0,0,0}, {-1,-1,-1}}), prewittVertical({{1,0,-1}, {1,0,-1}, {1,0,-1}}), 
    sobelHorizontal({{1,2,1}, {0,0,0}, {-1,-2,-1}}), sobelVertical({{-1,0,1}, {-2,0,2}, {-1,0,1}}), 
    scharrHorizontal({{3,10,3}, {0,0,0}, {-3,-10,-3}}), scharrVertical({{3,0,-3}, {10,0,-10}, {3,0,-3}}),
    robertsHorizontal({{0,1}, {-1,0}}), robertsVertical({{1,0}, {0,-1}}) 
    {

    }

    std::vector<std::vector<double>> Convolutions::convolve(std::vector<std::vector<double>> input, std::vector<std::vector<double>> filter, int S, int P){
        LinAlg alg;
        std::vector<std::vector<double>> featureMap;
        int N = input.size();
        int F = filter.size();
        int mapSize = (N - F + 2*P) / S + 1; // This is computed as ⌊mapSize⌋ by def- thanks C++!

        if(P != 0){
            std::vector<std::vector<double>> paddedInput; 
            paddedInput.resize(N + 2*P);
            for(int i = 0; i < paddedInput.size(); i++){
                paddedInput[i].resize(N + 2*P);
            }
            for(int i = 0; i < paddedInput.size(); i++){
                for(int j = 0; j < paddedInput[i].size(); j++){
                    if(i - P < 0 || j - P < 0 || i - P > input.size() - 1 || j - P > input[0].size() - 1){
                        paddedInput[i][j] = 0;
                    }
                    else{
                        paddedInput[i][j] = input[i - P][j - P];
                    }
                }
            }
            input.resize(paddedInput.size());
            for(int i = 0; i < paddedInput.size(); i++){
                input[i].resize(paddedInput[i].size());
            }
            input = paddedInput;
        }

        featureMap.resize(mapSize);
        for(int i = 0; i < mapSize; i++){
            featureMap[i].resize(mapSize);
        }

        for(int i = 0; i < mapSize; i++){
            for(int j = 0; j < mapSize; j++){
                std::vector<double> convolvingInput; 
                for(int k = 0; k < F; k++){       
                    for(int p = 0; p < F; p++){
                        if(i == 0 && j == 0){
                            convolvingInput.push_back(input[i + k][j + p]);
                        }
                        else if(i == 0){
                            convolvingInput.push_back(input[i + k][j + (S - 1) + p]);
                        }
                        else if(j == 0){
                            convolvingInput.push_back(input[i + (S - 1) + k][j + p]);
                        }
                        else{
                            convolvingInput.push_back(input[i + (S - 1) + k][j + (S - 1) + p]);
                        }   
                    }
                } 
                featureMap[i][j] = alg.dot(convolvingInput, alg.flatten(filter));
            }
        }
        return featureMap;
    }

    std::vector<std::vector<std::vector<double>>> Convolutions::convolve(std::vector<std::vector<std::vector<double>>> input, std::vector<std::vector<std::vector<double>>> filter, int S, int P){
        LinAlg alg;
        std::vector<std::vector<std::vector<double>>> featureMap;
        int N = input[0].size();
        int F = filter[0].size();
        int C = filter.size() / input.size();
        int mapSize = (N - F + 2*P) / S + 1; // This is computed as ⌊mapSize⌋ by def- thanks C++!

        if(P != 0){
            for(int c = 0; c < input.size(); c++){
                std::vector<std::vector<double>> paddedInput; 
                paddedInput.resize(N + 2*P);
                for(int i = 0; i < paddedInput.size(); i++){
                    paddedInput[i].resize(N + 2*P);
                }
                for(int i = 0; i < paddedInput.size(); i++){
                    for(int j = 0; j < paddedInput[i].size(); j++){
                        if(i - P < 0 || j - P < 0 || i - P > input[c].size() - 1 || j - P > input[c][0].size() - 1){
                            paddedInput[i][j] = 0;
                        }
                        else{
                            paddedInput[i][j] = input[c][i - P][j - P];
                        }
                    }
                }
                input[c].resize(paddedInput.size());
                for(int i = 0; i < paddedInput.size(); i++){
                    input[c][i].resize(paddedInput[i].size());
                }
                input[c] = paddedInput;
            }
        }

        featureMap.resize(C);
        for(int i = 0; i < featureMap.size(); i++){
            featureMap[i].resize(mapSize);
            for(int j = 0; j < featureMap[i].size(); j++){
                featureMap[i][j].resize(mapSize);
            }
        }

        for(int c = 0; c < C; c++){
            for(int i = 0; i < mapSize; i++){
                for(int j = 0; j < mapSize; j++){
                    std::vector<double> convolvingInput; 
                    for(int t = 0; t < input.size(); t++){
                        for(int k = 0; k < F; k++){       
                            for(int p = 0; p < F; p++){
                                if(i == 0 && j == 0){
                                    convolvingInput.push_back(input[t][i + k][j + p]);
                                }
                                else if(i == 0){
                                    convolvingInput.push_back(input[t][i + k][j + (S - 1) + p]);
                                }
                                else if(j == 0){
                                    convolvingInput.push_back(input[t][i + (S - 1) + k][j + p]);
                                }
                                else{
                                    convolvingInput.push_back(input[t][i + (S - 1) + k][j + (S - 1) + p]);
                                }   
                            }
                        } 
                    }
                    featureMap[c][i][j] = alg.dot(convolvingInput, alg.flatten(filter));
                }
            }
        }
        return featureMap;
    }

    std::vector<std::vector<double>> Convolutions::pool(std::vector<std::vector<double>> input, int F, int S, std::string type){
        LinAlg alg;
        std::vector<std::vector<double>> pooledMap;
        int N = input.size();
        int mapSize = floor((N - F) / S + 1); 
 
        pooledMap.resize(mapSize);
        for(int i = 0; i < mapSize; i++){
            pooledMap[i].resize(mapSize);
        }

        for(int i = 0; i < mapSize; i++){
            for(int j = 0; j < mapSize; j++){
                std::vector<double> poolingInput; 
                for(int k = 0; k < F; k++){       
                    for(int p = 0; p < F; p++){
                        if(i == 0 && j == 0){
                            poolingInput.push_back(input[i + k][j + p]);
                        }
                        else if(i == 0){
                            poolingInput.push_back(input[i + k][j + (S - 1) + p]);
                        }
                        else if(j == 0){
                            poolingInput.push_back(input[i + (S - 1) + k][j + p]);
                        }
                        else{
                            poolingInput.push_back(input[i + (S - 1) + k][j + (S - 1) + p]);
                        }   
                    }
                } 
                if(type == "Average"){
                    Stat stat; 
                    pooledMap[i][j] = stat.mean(poolingInput);
                }
                else if(type == "Min"){
                    pooledMap[i][j] = alg.min(poolingInput);
                }
                else{
                    pooledMap[i][j] = alg.max(poolingInput);
                }
            }
        }
        return pooledMap;
    }

    std::vector<std::vector<std::vector<double>>> Convolutions::pool(std::vector<std::vector<std::vector<double>>> input, int F, int S, std::string type){
        std::vector<std::vector<std::vector<double>>> pooledMap;
        for(int i = 0; i < input.size(); i++){
            pooledMap.push_back(pool(input[i], F, S, type));
        }
        return pooledMap; 
    }

    double Convolutions::globalPool(std::vector<std::vector<double>> input, std::string type){
        LinAlg alg;
        if(type == "Average"){
            Stat stat; 
            return stat.mean(alg.flatten(input));
        }
        else if(type == "Min"){
            return alg.min(alg.flatten(input));
        }
        else{
            return alg.max(alg.flatten(input));
        }
    }
            
    std::vector<double> Convolutions::globalPool(std::vector<std::vector<std::vector<double>>> input, std::string type){
        std::vector<double> pooledMap;
        for(int i = 0; i < input.size(); i++){
            pooledMap.push_back(globalPool(input[i], type));
        }
        return pooledMap; 
    }

    std::vector<std::vector<double>> Convolutions::getPrewittHorizontal(){
        return prewittHorizontal;
    }

    std::vector<std::vector<double>> Convolutions::getPrewittVertical(){
        return prewittVertical;
    }

    std::vector<std::vector<double>> Convolutions::getSobelHorizontal(){
        return sobelHorizontal;
    }

    std::vector<std::vector<double>> Convolutions::getSobelVertical(){
        return sobelVertical;
    }

    std::vector<std::vector<double>> Convolutions::getScharrHorizontal(){
        return scharrHorizontal;
    }

    std::vector<std::vector<double>> Convolutions::getScharrVertical(){
        return scharrVertical;
    }

    std::vector<std::vector<double>> Convolutions::getRobertsHorizontal(){
        return robertsHorizontal;
    }

    std::vector<std::vector<double>> Convolutions::getRobertsVertical(){
        return robertsVertical;
    }
}