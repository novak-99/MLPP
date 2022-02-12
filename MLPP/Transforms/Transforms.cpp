//
//  Transforms.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "Transforms.hpp"
#include "LinAlg/LinAlg.hpp"
#include <iostream>
#include <string>
#include <cmath>

namespace MLPP{

    // DCT ii.
    // https://www.mathworks.com/help/images/discrete-cosine-transform.html
    std::vector<std::vector<double>> Transforms::discreteCosineTransform(std::vector<std::vector<double>> A){
        LinAlg alg;
        A = alg.scalarAdd(-128, A); // Center around 0.

        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[i].size());
        }

        int M = A.size();

        for(int i = 0; i < B.size(); i++){
            for(int j = 0; j < B[i].size(); j++){
                double sum = 0;
                double alphaI;
                if(i == 0){
                    alphaI = 1/std::sqrt(M);
                }
                else{ 
                    alphaI = std::sqrt(double(2)/double(M)); 
                }
                double alphaJ;
                if(j == 0){
                    alphaJ = 1/std::sqrt(M); 
                }
                else{ 
                    alphaJ = std::sqrt(double(2)/double(M)); 
                }

                for(int k = 0; k < B.size(); k++){
                    for(int f = 0; f < B[k].size(); f++){
                        sum += A[k][f] * std::cos( (M_PI * i * (2 * k + 1)) / (2 * M)) * std::cos( (M_PI * j * (2 * f + 1)) / (2 * M));
                    }
                }
                B[i][j] = sum;
                B[i][j] *= alphaI * alphaJ;

            }
        }
        return B;
    }
}