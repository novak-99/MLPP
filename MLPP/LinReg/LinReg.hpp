//
//  LinReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#ifndef LinReg_hpp
#define LinReg_hpp

#include <vector>
#include <string>

namespace MLPP{
    class LinReg{
        
        public:
            LinReg(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, std::string reg = "None", double lambda = 0.5, double alpha = 0.5);
            std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
            double modelTest(std::vector<double> x);
            void NewtonRaphson(double learning_rate, int max_epoch, bool UI);
            void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
            void SGD(double learning_rate, int max_epoch, bool UI = 1);
            void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
            void normalEquation();
            double score();
            void save(std::string fileName);
        private:

            double Cost(std::vector <double> y_hat, std::vector<double> y);
        
            std::vector<double> Evaluate(std::vector<std::vector<double>> X);
            double Evaluate(std::vector<double> x);
            void forwardPass();
        
            std::vector<std::vector<double>> inputSet;
            std::vector<double> outputSet;
            std::vector<double> y_hat;
            std::vector<double> weights;
            double bias;
        
            int n; 
            int k;
        
            // Regularization Params
            std::string reg;
            int lambda;
            int alpha; /* This is the controlling param for Elastic Net*/
        
        
    };
}

#endif /* LinReg_hpp */
