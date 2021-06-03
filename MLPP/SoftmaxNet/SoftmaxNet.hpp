//
//  SoftmaxNet.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#ifndef SoftmaxNet_hpp
#define SoftmaxNet_hpp


#include <vector>
#include <string>

namespace MLPP {

    class SoftmaxNet{
        
        public:
            SoftmaxNet(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet, int n_hidden, std::string reg = "None", double lambda = 0.5, double alpha = 0.5);
            std::vector<double> modelTest(std::vector<double> x);
            std::vector<std::vector<double>> modelSetTest(std::vector<std::vector<double>> X);
            void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
            void SGD(double learning_rate, int max_epoch, bool UI = 1);
            void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
            double score();
            void save(std::string fileName);

            std::vector<std::vector<double>> getEmbeddings(); // This class is used (mostly) for word2Vec. This function returns our embeddings.
         private:

            double Cost(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);
        
            std::vector<std::vector<double>> Evaluate(std::vector<std::vector<double>> X);
            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> propagate(std::vector<std::vector<double>> X);
            std::vector<double> Evaluate(std::vector<double> x);
            std::tuple<std::vector<double>, std::vector<double>> propagate(std::vector<double> x);
            void forwardPass();
        
            std::vector<std::vector<double>> inputSet;
            std::vector<std::vector<double>> outputSet;
            std::vector<std::vector<double>> y_hat;

            std::vector<std::vector<double>> weights1;
            std::vector<std::vector<double>> weights2;
           
            std::vector<double> bias1;
            std::vector<double> bias2;

            std::vector<std::vector<double>> z2;
            std::vector<std::vector<double>> a2;
    
            int n; 
            int k;    
            int n_class;
            int n_hidden;

            // Regularization Params
            std::string reg;
            double lambda;
            double alpha; /* This is the controlling param for Elastic Net*/
        
        
    };
}

#endif /* SoftmaxNet_hpp */
