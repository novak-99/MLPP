//
//  ProbitReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#ifndef ProbitReg_hpp
#define ProbitReg_hpp


#include <vector>
#include <string>

namespace MLPP {

    class ProbitReg{
        
        public:
            ProbitReg(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, std::string reg = "None", double lambda = 0.5, double alpha = 0.5);
            std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
            double modelTest(std::vector<double> x);
            void gradientDescent(double learning_rate, int max_epoch = 0, bool UI = 1);
            void MLE(double learning_rate, int max_epoch = 0, bool UI = 1);
            void SGD(double learning_rate, int max_epoch = 0, bool UI = 1);
            void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
            double score();
            void save(std::string fileName);
        private:

            double Cost(std::vector <double> y_hat, std::vector<double> y);

            std::vector<double> Evaluate(std::vector<std::vector<double>> X);
            std::vector<double> propagate(std::vector<std::vector<double>> X);
            double Evaluate(std::vector<double> x);
            double propagate(std::vector<double> x);
            void forwardPass();
        
            std::vector<std::vector<double>> inputSet;
            std::vector<double> outputSet;
            std::vector<double> z;
            std::vector<double> y_hat;
            std::vector<double> weights;
            double bias;
        
            int n; 
            int k;

            // Regularization Params
            std::string reg;
            double lambda;
            double alpha; /* This is the controlling param for Elastic Net*/
        
        
    };
}

#endif /* ProbitReg_hpp */
