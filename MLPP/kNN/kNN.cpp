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
        // The nearest neighbors
        std::vector<double> knn;
        
        std::vector<std::vector<double>> inputUseSet = inputSet;
        //Perfom this loop unless and until all k nearest neighbors are found, appended, and returned
        for(int i = 0; i < k; i++){
            int neighbor = 0;
            for(int j = 0; j < inputUseSet.size(); j++){
                if(euclideanDistance(x, inputUseSet[j]) < euclideanDistance(x, inputUseSet[neighbor])){
                    neighbor = j;
                }
            }
            knn.push_back(neighbor);
            inputUseSet.erase(inputUseSet.begin() + neighbor);
        }
        return knn;
    }

    // Multidimensional Euclidean Distance
    double kNN::euclideanDistance(std::vector<double> A, std::vector<double> B){
        double dist = 0;
        for(int i = 0; i < A.size(); i++){
            dist += (A[i] - B[i])*(A[i] - B[i]);
        }
        return sqrt(dist);
    }
}