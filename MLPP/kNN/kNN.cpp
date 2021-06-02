//
//  kNN.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "kNN.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Utilities/Utilities.hpp"

#include <iostream>
#include <map>
#include <algorithm>

namespace MLPP{
    kNN::kNN(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int k)
    : inputSet(inputSet), outputSet(outputSet), k(k)
    {
        
    }
    
    std::vector<double> kNN::modelSetTest(std::vector<std::vector<double>> X){
        std::vector<double> y_hat;
        for(int i = 0; i < X.size(); i++){
            y_hat.push_back(modelTest(X[i]));
        }
        return y_hat;
    }

    int kNN::modelTest(std::vector<double> x){
        return determineClass(nearestNeighbors(x));
    }
    
    double kNN::score(){
        Utilities util;
        return util.performance(modelSetTest(inputSet), outputSet);
    }

    int kNN::determineClass(std::vector<double> knn){
        std::map<int, int> class_nums;
        for(int i = 0; i < outputSet.size(); i++){
            class_nums[outputSet[i]] = 0;
        }
        for(int i = 0; i < knn.size(); i++){
            for(int j = 0; j < outputSet.size(); j++){
                if(knn[i] == outputSet[j]){
                    class_nums[outputSet[j]]++;
                }
            }
        }
        int max = class_nums[outputSet[0]];
        int final_class = outputSet[0];
        
        for(int i = 0; i < outputSet.size(); i++){
            if(class_nums[outputSet[i]] > max){
                max = class_nums[outputSet[i]];
            }
        }
        for(auto [c, v] : class_nums){
            if(v == max){
                final_class = c;
            }
        }
        return final_class;
    }
    
    std::vector<double> kNN::nearestNeighbors(std::vector<double> x){
        LinAlg alg;
        // The nearest neighbors
        std::vector<double> knn;
        
        std::vector<std::vector<double>> inputUseSet = inputSet;
        //Perfom this loop unless and until all k nearest neighbors are found, appended, and returned
        for(int i = 0; i < k; i++){
            int neighbor = 0;
            for(int j = 0; j < inputUseSet.size(); j++){
                bool isNeighborNearer = alg.euclideanDistance(x, inputUseSet[j]) < alg.euclideanDistance(x, inputUseSet[neighbor]);
                if(isNeighborNearer){
                    neighbor = j;
                }
            }
            knn.push_back(neighbor);
            inputUseSet.erase(inputUseSet.begin() + neighbor); // This is why we maintain an extra input"Use"Set
        }
        return knn;
    }
}