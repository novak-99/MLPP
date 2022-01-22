//
//  BernoulliNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "BernoulliNB.hpp"
#include "Utilities/Utilities.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Data/Data.hpp"

#include <iostream>
#include <random>

namespace MLPP{
    BernoulliNB::BernoulliNB(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet)
    : inputSet(inputSet), outputSet(outputSet), class_num(2)
    {
        y_hat.resize(outputSet.size());
        Evaluate();
    }

    std::vector<double> BernoulliNB::modelSetTest(std::vector<std::vector<double>> X){
        std::vector<double> y_hat;
        for(int i = 0; i < X.size(); i++){
            y_hat.push_back(modelTest(X[i]));
        }
        return y_hat;
    }

    double BernoulliNB::modelTest(std::vector<double> x){
        double score_0 = 1;
        double score_1 = 1;
        
        std::vector<int> foundIndices;

        for(int j = 0; j < x.size(); j++){
            for(int k = 0; k < vocab.size(); k++){
                if(x[j] == vocab[k]){
                    score_0 *= theta[0][vocab[k]];
                    score_1 *= theta[1][vocab[k]];

                    foundIndices.push_back(k);
                }
            }
        }

        for(int i = 0; i < vocab.size(); i++){
            bool found = false; 
            for(int j = 0; j < foundIndices.size(); j++){
                if(vocab[i] == vocab[foundIndices[j]]){
                    found = true; 
                }
            }
            if(!found){
                score_0 *= 1 - theta[0][vocab[i]];
                score_1 *= 1 - theta[1][vocab[i]];
            }
        }
            
        score_0 *= prior_0;
        score_1 *= prior_1;
            
        // Assigning the traning example to a class

        if(score_0 > score_1){
            return 0;
        }
        else{
            return 1;
        }
    }

    double BernoulliNB::score(){
        Utilities util;
        return util.performance(y_hat, outputSet);
    }

    void BernoulliNB::computeVocab(){
        LinAlg alg;
        Data data;
        vocab = data.vecToSet<double>(alg.flatten(inputSet));
    }
    
    void BernoulliNB::computeTheta(){
        
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
                if(i == 0){
                    theta[i][j] /= prior_0 * y_hat.size();
                }
                else{
                    theta[i][j] /= prior_1 * y_hat.size();
                }
            }
        }
    }

    void BernoulliNB::Evaluate(){
        for(int i = 0; i < outputSet.size(); i++){
            // Pr(B | A) * Pr(A)
            double score_0 = 1;
            double score_1 = 1;

            
            double sum = 0;
            for(int i = 0; i < outputSet.size(); i++){
                if(outputSet[i] == 1){ sum += outputSet[i]; }
            }
            
            // Easy computation of priors, i.e. Pr(C_k)
            prior_1 = sum / y_hat.size();
            prior_0 = 1 - prior_1;
            
            // Evaluating Theta...
            computeTheta();

            // Evaluating the vocab set...
            computeVocab();

            std::vector<int> foundIndices;

            for(int j = 0; j < inputSet.size(); j++){
                for(int k = 0; k < vocab.size(); k++){
                    if(inputSet[i][j] == vocab[k]){
                        score_0 += std::log(theta[0][vocab[k]]);
                        score_1 += std::log(theta[1][vocab[k]]);

                        foundIndices.push_back(k);
                    }
                }
            }

            for(int i = 0; i < vocab.size(); i++){
                bool found = false; 
                for(int j = 0; j < foundIndices.size(); j++){
                    if(vocab[i] == vocab[foundIndices[j]]){
                        found = true; 
                    }
                }
                if(!found){
                    score_0 += std::log(1 - theta[0][vocab[i]]);
                    score_1 += std::log(1 - theta[1][vocab[i]]);
                }
            }
            
            score_0 += std::log(prior_0);
            score_1 += std::log(prior_1);

            score_0 = exp(score_0);
            score_1 = exp(score_1);

            std::cout << score_0 << std::endl;
            std::cout << score_1 << std::endl;
            
            // Assigning the traning example to a class

            if(score_0 > score_1){
                y_hat[i] = 0;
            }
            else{
                y_hat[i] = 1;
            }
        }
    }
}