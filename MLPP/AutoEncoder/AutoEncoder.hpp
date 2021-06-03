//
//  AutoEncoder.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#ifndef AutoEncoder_hpp
#define AutoEncoder_hpp

#include <vector>
#include <tuple>
#include <string>

namespace  MLPP {

class AutoEncoder{
        public:
        AutoEncoder(std::vector<std::vector<double>> inputSet, int n_hidden);
        std::vector<std::vector<double>> modelSetTest(std::vector<std::vector<double>> X);
        std::vector<double> modelTest(std::vector<double> x);
        void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
        void SGD(double learning_rate, int max_epoch, bool UI = 1);
        void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
        double score(); 
        void save(std::string fileName);
        
        private:
            double Cost(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<std::vector<double>> Evaluate(std::vector<std::vector<double>> X);
            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> propagate(std::vector<std::vector<double>> X);
            std::vector<double> Evaluate(std::vector<double> x);
            std::tuple<std::vector<double>, std::vector<double>> propagate(std::vector<double> x);
            void forwardPass();

            std::vector<std::vector<double>> inputSet;
            std::vector<std::vector<double>> y_hat;
        
            std::vector<std::vector<double>> weights1;
            std::vector<std::vector<double>> weights2;
           
            std::vector<double> bias1;
            std::vector<double> bias2;
        
            std::vector<std::vector<double>> z2;
            std::vector<std::vector<double>> a2;

            int n;
            int k;
            int n_hidden;
    };
}

#endif /* AutoEncoder_hpp */
