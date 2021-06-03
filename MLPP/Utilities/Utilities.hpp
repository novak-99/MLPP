//
//  Utilities.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#ifndef Utilities_hpp
#define Utilities_hpp

#include <vector>
#include <tuple>
#include <string>

namespace MLPP{
    class Utilities{
        public:
            // Weight Init
            static std::vector<double> weightInitialization(int n, std::string type = "Default");
            static double biasInitialization();

            static std::vector<std::vector<double>> weightInitialization(int n, int m, std::string type = "Default");
            static std::vector<double> biasInitialization(int n);

            // Cost/Performance related Functions
            double performance(std::vector<double> y_hat, std::vector<double> y);
            double performance(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y); 

            // Parameter Saving Functions
            void saveParameters(std::string fileName, std::vector<double> weights, double bias, bool app = 0, int layer = -1);
            void saveParameters(std::string fileName, std::vector<double> weights, std::vector<double> initial, double bias, bool app = 0, int layer = -1);
            void saveParameters(std::string fileName, std::vector<std::vector<double>> weights, std::vector<double> bias, bool app = 0, int layer = -1);

            // Gradient Descent related
            static void UI(std::vector<double> weights, double bias);
            static void UI(std::vector<double> weights, std::vector<double> initial, double bias);
            static void UI(std::vector<std::vector<double>>, std::vector<double> bias);
            static void CostInfo(int epoch, double cost_prev, double Cost);

            static std::vector<std::vector<std::vector<double>>> createMiniBatches(std::vector<std::vector<double>> inputSet, int n_mini_batch);
            static std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> createMiniBatches(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int n_mini_batch);
            static std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>> createMiniBatches(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet, int n_mini_batch);

            // F1 score, Precision/Recall, TP, FP, TN, FN, etc. 
            std::tuple<double, double, double, double> TF_PN(std::vector<double> y_hat, std::vector<double> y); //TF_PN = "True", "False", "Positive", "Negative"
            double recall(std::vector<double> y_hat, std::vector<double> y);
            double precision(std::vector<double> y_hat, std::vector<double> y);
            double accuracy(std::vector<double> y_hat, std::vector<double> y);
            double f1_score(std::vector<double> y_hat, std::vector<double> y);

        private:
    };
}

#endif /* Utilities_hpp */
