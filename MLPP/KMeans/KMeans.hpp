//
//  KMeans.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#ifndef KMeans_hpp
#define KMeans_hpp

#include <vector>
#include <string>

namespace MLPP{
    class KMeans{
        
        public:
            KMeans(std::vector<std::vector<double>> inputSet, int k, std::string init_type = "Default");
            std::vector<std::vector<double>> modelSetTest(std::vector<std::vector<double>> X);
            std::vector<double> modelTest(std::vector<double> x);
            void train(int epoch_num, bool UI = 1);
            double score();
            std::vector<double> silhouette_scores(); 
        private:
        
            void Evaluate();
            void computeMu();
        
            void centroidInitialization(int k);
            void kmeansppInitialization(int k);
            double Cost();
        
            std::vector<std::vector<double>> inputSet;
            std::vector<std::vector<double>> mu;
            std::vector<std::vector<double>> r;
        
            double euclideanDistance(std::vector<double> A, std::vector<double> B);
        
            double accuracy_threshold;
            int k;        

            std::string init_type;
    };
}

#endif /* KMeans_hpp */
