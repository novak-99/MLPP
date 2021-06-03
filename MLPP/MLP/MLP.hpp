//
//  MLP.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#ifndef MLP_hpp
#define MLP_hpp

#include <vector>
#include <map>
#include <string>

namespace  MLPP {

class MLP{
        public:
        MLP(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int n_hidden, std::string reg = "None", double lambda = 0.5, double alpha = 0.5);
        std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
        double modelTest(std::vector<double> x);
        void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
        void SGD(double learning_rate, int max_epoch, bool UI = 1);
        void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
        double score(); 
        void save(std::string fileName);
        
        private:
            double Cost(std::vector <double> y_hat, std::vector<double> y);

            std::vector<double> Evaluate(std::vector<std::vector<double>> X);
            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> propagate(std::vector<std::vector<double>> X);
            double Evaluate(std::vector<double> x);
            std::tuple<std::vector<double>, std::vector<double>> propagate(std::vector<double> x);
            void forwardPass();

            std::vector<std::vector<double>> inputSet;
            std::vector<double> outputSet;
            std::vector<double> y_hat;
        
            std::vector<std::vector<double>> weights1;
            std::vector<double> weights2;
           
            std::vector<double> bias1;
            double bias2;
        
            std::vector<std::vector<double>> z2;
            std::vector<std::vector<double>> a2;

            int n;
            int k;
            int n_hidden;


            // Regularization Params
            std::string reg;
            double lambda; /* Regularization Parameter */
            double alpha; /* This is the controlling param for Elastic Net*/
    };
}

#endif /* MLP_hpp */
