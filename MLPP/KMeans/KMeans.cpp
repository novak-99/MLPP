//
//  KMeans.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "KMeans.hpp"
#include "Utilities/Utilities.hpp"
#include "LinAlg/LinAlg.hpp"

#include <iostream>
#include <random>
#include <climits>

namespace MLPP{
    KMeans::KMeans(std::vector<std::vector<double>> inputSet, int k, std::string init_type)
    : inputSet(inputSet), k(k), init_type(init_type)
    {
        if(init_type == "KMeans++"){ 
            kmeansppInitialization(k); 
        }
        else{
            centroidInitialization(k);
        }
    }

    std::vector<std::vector<double>> KMeans::modelSetTest(std::vector<std::vector<double>> X){
        LinAlg alg;
        std::vector<std::vector<double>> closestCentroids; 
        for(int i = 0; i < inputSet.size(); i++){
            std::vector<double> closestCentroid = mu[0];
            for(int j = 0; j < r[0].size(); j++){
                bool isCentroidCloser = alg.euclideanDistance(X[i], mu[j]) < alg.euclideanDistance(X[i], closestCentroid);
                if(isCentroidCloser){
                    closestCentroid = mu[j];
                }
            }
            closestCentroids.push_back(closestCentroid);
        }
        return closestCentroids;
    }

    std::vector<double> KMeans::modelTest(std::vector<double> x){
        LinAlg alg;
        std::vector<double> closestCentroid = mu[0];
        for(int j = 0; j < mu.size(); j++){
            if(alg.euclideanDistance(x, mu[j]) < alg.euclideanDistance(x, closestCentroid)){
                closestCentroid = mu[j];
            }
        }
        return closestCentroid;
    }

    void KMeans::train(int epoch_num, bool UI){
        double cost_prev = 0;
        int epoch = 1;
        
        Evaluate();
        
        while(true){
            
            // STEPS OF THE ALGORITHM
            // 1. DETERMINE r_nk
            // 2. DETERMINE J
            // 3. DETERMINE mu_k
            
            // STOP IF CONVERGED, ELSE REPEAT
            
            cost_prev = Cost();
            
            computeMu();
            Evaluate();
                
            // UI PORTION
            if(UI) { Utilities::CostInfo(epoch, cost_prev, Cost()); }
            epoch++;

            if(epoch > epoch_num) { break; }

        }
    }

    double KMeans::score(){
        return Cost();
    }

    std::vector<double> KMeans::silhouette_scores(){
        LinAlg alg;
        std::vector<std::vector<double>> closestCentroids = modelSetTest(inputSet);
        std::vector<double> silhouette_scores;
        for(int i = 0; i < inputSet.size(); i++){
            // COMPUTING a[i]
            double a = 0;
            for(int j = 0; j < inputSet.size(); j++){
                if(i != j && r[i] == r[j]){
                    a += alg.euclideanDistance(inputSet[i], inputSet[j]);
                }
            }   
            // NORMALIZE a[i]
            a /= closestCentroids[i].size() - 1; 


            // COMPUTING b[i]
            double b = INT_MAX; 
            for(int j = 0; j < mu.size(); j++){
                if(closestCentroids[i] != mu[j]){
                    double sum = 0;
                    for(int k = 0; k < inputSet.size(); k++){
                        sum += alg.euclideanDistance(inputSet[i], inputSet[k]);
                    }
                    // NORMALIZE b[i]
                    double k_clusterSize = 0;
                    for(int k = 0; k < closestCentroids.size(); k++){
                        if(closestCentroids[k] == mu[j]){
                            k_clusterSize++;
                        }
                    }
                    if(sum / k_clusterSize < b) { b = sum / k_clusterSize; }
                }
            }
            silhouette_scores.push_back((b - a)/fmax(a, b));
            // Or the expanded version: 
            // if(a < b) {
            //     silhouette_scores.push_back(1 - a/b); 
            // }
            // else if(a == b){
            //     silhouette_scores.push_back(0);
            // }
            // else{
            //     silhouette_scores.push_back(b/a - 1);
            // }
        }
        return silhouette_scores;
    }

    // This simply computes r_nk
    void KMeans::Evaluate(){
        LinAlg alg;
        r.resize(inputSet.size());
        
        for(int i = 0; i < r.size(); i++){
            r[i].resize(k);
        }
        
        for(int i = 0; i < r.size(); i++){
            std::vector<double> closestCentroid = mu[0];
            for(int j = 0; j < r[0].size(); j++){
                bool isCentroidCloser = alg.euclideanDistance(inputSet[i], mu[j]) < alg.euclideanDistance(inputSet[i], closestCentroid);
                if(isCentroidCloser){
                    closestCentroid = mu[j];
                }
            }
            for(int j = 0; j < r[0].size(); j++){
                if(mu[j] == closestCentroid) {
                    r[i][j] = 1;
                }
                else { r[i][j] = 0; }
            }
        }
        
    }

    // This simply computes or re-computes mu_k
    void KMeans::computeMu(){
        LinAlg alg;
        for(int i = 0; i < mu.size(); i++){
            std::vector<double> num;
            num.resize(r.size());
            
            for(int i = 0; i < num.size(); i++){
                num[i] = 0;
            }
            
            double den = 0;
            for(int j = 0; j < r.size(); j++){
                num = alg.addition(num, alg.scalarMultiply(r[j][i], inputSet[j]));
            }
            for(int j = 0; j < r.size(); j++){
                den += r[j][i];
            }
            mu[i] = alg.scalarMultiply(double(1)/double(den), num);
        }
        
    }

    void KMeans::centroidInitialization(int k){
        mu.resize(k);
        
        for(int i = 0; i < k; i++){
            std::random_device rd;
            std::default_random_engine generator(rd()); 
            std::uniform_int_distribution<int> distribution(0, int(inputSet.size() - 1));

            mu[i].resize(inputSet.size());
            mu[i] = inputSet[distribution(generator)];
        }
    }

    void KMeans::kmeansppInitialization(int k){
        LinAlg alg;
        std::random_device rd;
        std::default_random_engine generator(rd()); 
        std::uniform_int_distribution<int> distribution(0, int(inputSet.size() - 1));
        mu.push_back(inputSet[distribution(generator)]);

        for(int i = 0; i < k - 1; i++){
            std::vector<double> farthestCentroid;
            for(int j = 0; j < inputSet.size(); j++){
                double max_dist = 0; 
                /* SUM ALL THE SQUARED DISTANCES, CHOOSE THE ONE THAT'S FARTHEST
                AS TO SPREAD OUT THE CLUSTER CENTROIDS. */
                double sum = 0;
                for(int k = 0; k < mu.size(); k++){
                    sum += alg.euclideanDistance(inputSet[j], mu[k]);
                }
                if(sum * sum > max_dist){
                    farthestCentroid = inputSet[j];
                    max_dist = sum * sum;
                }
            }
            mu.push_back(farthestCentroid);
        }
    }

    double KMeans::Cost(){
        LinAlg alg;
        double sum = 0;
        for(int i = 0; i < r.size(); i++){
            for(int j = 0; j < r[0].size(); j++){
                sum += r[i][j] * alg.norm_sq(alg.subtraction(inputSet[i], mu[j]));
            }
        }
        return sum;
    }
}
