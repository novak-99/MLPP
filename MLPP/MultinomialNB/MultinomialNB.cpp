//
//  MultinomialNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "MultinomialNB.hpp"
#include "Utilities/Utilities.hpp"
#include "LinAlg/LinAlg.hpp"

#include <iostream>
#include <algorithm>
#include <random>

namespace MLPP{
    MultinomialNB::MultinomialNB(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int class_num)
    : inputSet(inputSet), outputSet(outputSet), class_num(class_num)
    {
        y_hat.resize(outputSet.size());
        Evaluate();
    }

    std::vector<double> MultinomialNB::modelSetTest(std::vector<std::vector<double>> X){
        std::vector<double> y_hat;
        for(int i = 0; i < X.size(); i++){
            y_hat.push_back(modelTest(X[i]));
        }
        return y_hat;
    }

    double MultinomialNB::modelTest(std::vector<double> x){
        double score[class_num];
        computeTheta();
        
        for(int j = 0; j < x.size(); j++){
            for(int k = 0; k < vocab.size(); k++){
                if(x[j] == vocab[k]){
                    for(int p = class_num - 1; p >= 0; p--){
                        score[p] += std::log(theta[p][vocab[k]]);
                    }
                }
            }
        }

        for(int i = 0; i < priors.size(); i++){
            score[i] += std::log(priors[i]);
        }

        return std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(double)));
    }

    double MultinomialNB::score(){
        Utilities util;
        return util.performance(y_hat, outputSet);
    }

    void MultinomialNB::computeTheta(){
        
        // Resizing theta for the sake of ease & proper access of the elements.
        theta.resize(class_num);
        
        // Setting all values in the hasmap by default to 0.
        for(int i = class_num - 1; i >= 0; i--){
            for(int j = 0; j < vocab.size(); j++){
                theta[i][vocab[j]] = 0; 
            }
        }

        for(int i = 0; i < inputSet.size(); i++){  
            for(int j = 0; j < inputSet[0].size(); j++){
                theta[outputSet[i]][inputSet[i][j]]++;
            }
        }
        
        for(int i = 0; i < theta.size(); i++){
            for(int j = 0; j < theta[i].size(); j++){
                theta[i][j] /= priors[i] * y_hat.size();
            }
        }
    }

    void MultinomialNB::Evaluate(){
        LinAlg alg;
        for(int i = 0; i < outputSet.size(); i++){
            // Pr(B | A) * Pr(A)
            double score[class_num];

            // Easy computation of priors, i.e. Pr(C_k)
            priors.resize(class_num);
            for(int i = 0; i < outputSet.size(); i++){
                priors[int(outputSet[i])]++;
            }
            priors = alg.scalarMultiply( double(1)/double(outputSet.size()), priors);
            
            // Evaluating Theta...
            computeTheta();
            
            for(int j = 0; j < inputSet.size(); j++){
                for(int k = 0; k < vocab.size(); k++){
                    if(inputSet[i][j] == vocab[k]){
                        for(int p = class_num - 1; p >= 0; p--){
                            score[p] += std::log(theta[i][vocab[k]]);
                        }
                    }
                }
            }

            for(int i = 0; i < priors.size(); i++){
                score[i] += std::log(priors[i]);
                score[i] = exp(score[i]);
            }

            for(int i = 0; i < 2; i++){
                std::cout << score[i] << std::endl;
            }
            
            // Assigning the traning example's y_hat to a class
            y_hat[i] = std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(double)));
        }
    }
}