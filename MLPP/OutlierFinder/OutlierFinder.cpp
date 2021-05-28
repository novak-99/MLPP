//
//  OutlierFinder.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "OutlierFinder.hpp"
#include "Stat/Stat.hpp"
#include <iostream>

namespace MLPP{
    OutlierFinder::OutlierFinder(int threshold)
    : threshold(threshold){

    }

    std::vector<std::vector<double>> OutlierFinder::modelSetTest(std::vector<std::vector<double>> inputSet){
        Stat stat;
        std::vector<std::vector<double>> outliers;
        outliers.resize(inputSet.size());
        for(int i = 0; i < inputSet.size(); i++){
            for(int j = 0; j < inputSet[i].size(); j++){
                double z = (inputSet[i][j] - stat.mean(inputSet[i])) / stat.standardDeviation(inputSet[i]);
                if(abs(z) > threshold){
                    outliers[i].push_back(inputSet[i][j]);
                }
            }
        }
        return outliers; 
    }

    std::vector<double> OutlierFinder::modelTest(std::vector<double> inputSet){
        Stat stat;
        std::vector<double> outliers;
        for(int i = 0; i < inputSet.size(); i++){
            double z = (inputSet[i] - stat.mean(inputSet)) / stat.standardDeviation(inputSet);
            if(abs(z) > threshold){
                outliers.push_back(inputSet[i]);
            }
        }
        return outliers; 
    }
}