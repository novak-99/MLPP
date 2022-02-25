//
//  Cost.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#ifndef Cost_hpp
#define Cost_hpp

#include <vector>

namespace MLPP{
    class Cost{
        public:
            // Regression Costs
            double MSE(std::vector <double> y_hat, std::vector<double> y);
            double MSE(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<double> MSEDeriv(std::vector <double> y_hat, std::vector<double> y);
            std::vector<std::vector<double>> MSEDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            double RMSE(std::vector <double> y_hat, std::vector<double> y);
            double RMSE(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<double> RMSEDeriv(std::vector <double> y_hat, std::vector<double> y);
            std::vector<std::vector<double>> RMSEDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            double MAE(std::vector <double> y_hat, std::vector<double> y);
            double MAE(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<double> MAEDeriv(std::vector <double> y_hat, std::vector <double> y);
            std::vector<std::vector<double>> MAEDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            double MBE(std::vector <double> y_hat, std::vector <double> y);
            double MBE(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<double> MBEDeriv(std::vector <double> y_hat, std::vector <double> y);
            std::vector<std::vector<double>> MBEDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            // Classification Costs
            double LogLoss(std::vector <double> y_hat, std::vector<double> y);
            double LogLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<double> LogLossDeriv(std::vector <double> y_hat, std::vector<double> y);
            std::vector<std::vector<double>> LogLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            double CrossEntropy(std::vector<double> y_hat, std::vector<double> y);
            double CrossEntropy(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<double> CrossEntropyDeriv(std::vector<double> y_hat, std::vector<double> y);
            std::vector<std::vector<double>> CrossEntropyDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            double HuberLoss(std::vector <double> y_hat, std::vector<double> y, double delta);
            double HuberLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y, double delta);

            std::vector<double> HuberLossDeriv(std::vector <double> y_hat, std::vector<double> y, double delta); 
            std::vector<std::vector<double>> HuberLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y, double delta);

            double HingeLoss(std::vector <double> y_hat, std::vector<double> y);
            double HingeLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<double> HingeLossDeriv(std::vector <double> y_hat, std::vector<double> y); 
            std::vector<std::vector<double>> HingeLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            double HingeLoss(std::vector <double> y_hat, std::vector<double> y, std::vector<double> weights, double C);
            double HingeLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y, std::vector<std::vector<double>> weights, double C);

            std::vector<double> HingeLossDeriv(std::vector <double> y_hat, std::vector<double> y, double C); 
            std::vector<std::vector<double>> HingeLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y, double C);

            double WassersteinLoss(std::vector<double> y_hat, std::vector<double> y);
            double WassersteinLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            std::vector<double> WassersteinLossDeriv(std::vector<double> y_hat, std::vector<double> y);
            std::vector<std::vector<double>> WassersteinLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y);

            double dualFormSVM(std::vector<double> alpha, std::vector<std::vector<double>> X, std::vector<double> y); // TO DO: DON'T forget to add non-linear kernelizations. 

            std::vector<double> dualFormSVMDeriv(std::vector<double> alpha, std::vector<std::vector<double>> X, std::vector<double> y);
            

        private:
    };
}

#endif /* Cost_hpp */
