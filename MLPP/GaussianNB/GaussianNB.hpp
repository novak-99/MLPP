//
//  GaussianNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#ifndef GaussianNB_hpp
#define GaussianNB_hpp

#include <vector>

namespace MLPP{
    class GaussianNB{
        
        public:
            GaussianNB(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int class_num);
            std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
            double modelTest(std::vector<double> x);
            double score();
            
        private:
        
            void Evaluate();

            int class_num;

            std::vector<double> priors; 
            std::vector<double> mu;
            std::vector<double> sigma;
            
            std::vector<std::vector<double>> inputSet;
            std::vector<double> outputSet;

            std::vector<double> y_hat;
            
        
            
        
    };

    #endif /* GaussianNB_hpp */
}