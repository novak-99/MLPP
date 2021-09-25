//
//  Activation.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#ifndef Activation_hpp
#define Activation_hpp

#include <vector>

namespace MLPP{
    class Activation{
        public:
            double linear(double z, bool deriv = 0); 
            std::vector<double> linear(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> linear(std::vector<std::vector<double>> z, bool deriv = 0);

            double sigmoid(double z, bool deriv = 0); 
            std::vector<double> sigmoid(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>> z, bool deriv = 0);

            std::vector<double> softmax(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> softmax(std::vector<std::vector<double>> z, bool deriv = 0);

            std::vector<double> adjSoftmax(std::vector<double> z);
            std::vector<std::vector<double>> adjSoftmax(std::vector<std::vector<double>> z);

            std::vector<std::vector<double>> softmaxDeriv(std::vector<double> z);
            std::vector<std::vector<std::vector<double>>> softmaxDeriv(std::vector<std::vector<double>> z);

            double softplus(double z, bool deriv = 0);
            std::vector<double> softplus(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> softplus(std::vector<std::vector<double>>  z, bool deriv = 0);

            double softsign(double z, bool deriv = 0);
            std::vector<double> softsign(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> softsign(std::vector<std::vector<double>>  z, bool deriv = 0);

            double gaussianCDF(double z, bool deriv = 0);
            std::vector<double> gaussianCDF(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> gaussianCDF(std::vector<std::vector<double>>  z, bool deriv = 0);

            double cloglog(double z, bool deriv = 0);
            std::vector<double> cloglog(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> cloglog(std::vector<std::vector<double>> z, bool deriv = 0);

            double logit(double z, bool deriv = 0);
            std::vector<double> logit(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> logit(std::vector<std::vector<double>> z, bool deriv = 0);

            double unitStep(double z, bool deriv = 0);
            std::vector<double> unitStep(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> unitStep(std::vector<std::vector<double>> z, bool deriv = 0);

            double swish(double z, bool deriv = 0);
            std::vector<double> swish(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> swish(std::vector<std::vector<double>> z, bool deriv = 0);

            double mish(double z, bool deriv = 0);
            std::vector<double> mish(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> mish(std::vector<std::vector<double>> z, bool deriv = 0);

            double sinc(double z, bool deriv = 0);
            std::vector<double> sinc(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> sinc(std::vector<std::vector<double>> z, bool deriv = 0);

            double RELU(double z, bool deriv = 0);
            std::vector<double> RELU(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> RELU(std::vector<std::vector<double>> z, bool deriv = 0);

            double leakyReLU(double z, double c, bool deriv = 0);
            std::vector<double> leakyReLU(std::vector<double> z, double c, bool deriv = 0);
            std::vector<std::vector<double>> leakyReLU(std::vector<std::vector<double>> z, double c, bool deriv = 0);

            double ELU(double z, double c, bool deriv = 0);
            std::vector<double> ELU(std::vector<double> z, double c, bool deriv = 0);
            std::vector<std::vector<double>> ELU(std::vector<std::vector<double>> z, double c, bool deriv = 0);

            double SELU(double z, double lambda, double c, bool deriv = 0);
            std::vector<double> SELU(std::vector<double> z, double lambda, double c, bool deriv = 0);
            std::vector<std::vector<double>> SELU(std::vector<std::vector<double>>, double lambda, double c, bool deriv = 0);

            double GELU(double z, bool deriv = 0);
            std::vector<double> GELU(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> GELU(std::vector<std::vector<double>> z, bool deriv = 0);

            double sign(double z, bool deriv = 0);
            std::vector<double> sign(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> sign(std::vector<std::vector<double>> z, bool deriv = 0);

            double sinh(double z, bool deriv = 0);
            std::vector<double> sinh(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> sinh(std::vector<std::vector<double>> z, bool deriv = 0);

            double cosh(double z, bool deriv = 0);
            std::vector<double> cosh(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> cosh(std::vector<std::vector<double>> z, bool deriv = 0);

            double tanh(double z, bool deriv = 0);
            std::vector<double> tanh(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> tanh(std::vector<std::vector<double>> z, bool deriv = 0);

            double csch(double z, bool deriv = 0);
            std::vector<double> csch(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> csch( std::vector<std::vector<double>> z, bool deriv = 0);

            double sech(double z, bool deriv = 0);
            std::vector<double> sech(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> sech(std::vector<std::vector<double>> z, bool deriv = 0);

            double coth(double z, bool deriv = 0);
            std::vector<double> coth(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> coth(std::vector<std::vector<double>> z, bool deriv = 0);

            double arsinh(double z, bool deriv = 0);
            std::vector<double> arsinh(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> arsinh(std::vector<std::vector<double>> z, bool deriv = 0);

            double arcosh(double z, bool deriv = 0);
            std::vector<double> arcosh(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> arcosh(std::vector<std::vector<double>> z, bool deriv = 0);

            double artanh(double z, bool deriv = 0);
            std::vector<double> artanh(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> artanh(std::vector<std::vector<double>> z, bool deriv = 0);

            double arcsch(double z, bool deriv = 0);
            std::vector<double> arcsch(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> arcsch(std::vector<std::vector<double>> z, bool deriv = 0);

            double arsech(double z, bool deriv = 0);
            std::vector<double> arsech(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> arsech(std::vector<std::vector<double>> z, bool deriv = 0);

            double arcoth(double z, bool deriv = 0);
            std::vector<double> arcoth(std::vector<double> z, bool deriv = 0);
            std::vector<std::vector<double>> arcoth(std::vector<std::vector<double>> z, bool deriv = 0);

            std::vector<double> activation(std::vector<double> z, bool deriv, double(*function)(double, bool));

        private:
    };
}

#endif /* Activation_hpp */
