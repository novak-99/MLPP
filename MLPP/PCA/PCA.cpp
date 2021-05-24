//
//  PCA.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "PCA.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Data/Data.hpp"

#include <iostream>
#include <random>

namespace MLPP{

    PCA::PCA(std::vector<std::vector<double>> inputSet, int k)
    : inputSet(inputSet), k(k)
    {

    }

    std::vector<std::vector<double>> PCA::principalComponents(){
        LinAlg alg;
        Data data; 

        auto [U, S, Vt] = alg.SVD(alg.cov(inputSet));
        X_normalized = data.meanCentering(inputSet);
        U_reduce.resize(U.size());
        for(int i = 0; i < k; i++){
            for(int j = 0; j < U.size(); j++){
                U_reduce[j].push_back(U[j][i]);
            }
        }
        Z = alg.matmult(alg.transpose(U_reduce), X_normalized);
        return Z;
    }
    // Simply tells us the percentage of variance maintained. 
    double PCA::score(){
        LinAlg alg;
        std::vector<std::vector<double>> X_approx = alg.matmult(U_reduce, Z);
        double num, den = 0;
        for(int i = 0; i < X_normalized.size(); i++){
            num += alg.norm_sq(alg.subtraction(X_normalized[i], X_approx[i]));
        }
        num /= X_normalized.size();
        for(int i = 0; i < X_normalized.size(); i++){
            den += alg.norm_sq(X_normalized[i]);
        }

        den /= X_normalized.size();
        if(den == 0){
            den+=1e-10; // For numerical sanity as to not recieve a domain error
        }
        return 1 - num/den;
    }
}
