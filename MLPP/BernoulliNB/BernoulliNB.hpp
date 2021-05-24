//
//  BernoulliNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#ifndef BernoulliNB_hpp
#define BernoulliNB_hpp

#include <vector>
#include <map>

namespace MLPP{
    class BernoulliNB{
        
        public:
            BernoulliNB(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet);
            std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
            double modelTest(std::vector<double> x);
            double score();
            
        private:
        
            void computeVocab();
            void computeTheta();
            void Evaluate();
        
            // Model Params
            double prior_1 = 0;
            double prior_0 = 0;
        
            std::vector<std::map<double, int>> theta;
            std::vector<double> vocab;
            int class_num;
            
            // Datasets
            std::vector<std::vector<double>> inputSet;
            std::vector<double> outputSet;
            std::vector<double> y_hat;
            
        
            
        
    };

    #endif /* BernoulliNB_hpp */
}