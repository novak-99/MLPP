//
//  SoftmaxReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#ifndef SoftmaxReg_hpp
#define SoftmaxReg_hpp


#include <vector>
#include <string>

namespace MLPP {

    class SoftmaxReg{
        
        public:
            SoftmaxReg(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet, std::string reg = "None", double lambda = 0.5, double alpha = 0.5);
            std::vector<double> modelTest(std::vector<double> x);
            std::vector<std::vector<double>> modelSetTest(std::vector<std::vector<double>> X);
            void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
            void SGD(double learning_rate, int max_epoch, bool UI = 1);
            void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
            double score();
            void save(std::string fileName);
        private:

            double Cost(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);
        
            std::vector<std::vector<double>> Evaluate(std::vector<std::vector<double>> X);
            std::vector<double> Evaluate(std::vector<double> x);
            void forwardPass();
        
            std::vector<std::vector<double>> inputSet;
            std::vector<std::vector<double>> outputSet;
            std::vector<std::vector<double>> y_hat;
            std::vector<std::vector<double>> weights;
            std::vector<double> bias;
    
            int n; 
            int k;    
            int n_class;

            // Regularization Params
            std::string reg;
            double lambda;
            double alpha; /* This is the controlling param for Elastic Net*/
        
        
    };
}

#endif /* SoftmaxReg_hpp */
