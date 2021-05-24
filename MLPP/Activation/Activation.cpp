//
//  Activation.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include <iostream>
#include "LinAlg/LinAlg.hpp"
#include "Activation.hpp"

namespace MLPP{

    double Activation::linear(double z, bool deriv){
        if(deriv){ return 1; }
        return z; 
    }

    std::vector<double> Activation::linear(std::vector<double> z, bool deriv){
        if(deriv) { 
            LinAlg alg; 
            return alg.onevec(z.size());
         }
         return z; 

    }

    std::vector<std::vector<double>> Activation::linear(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            LinAlg alg;
            return alg.onemat(z.size(), z[0].size());
        }
        return z; 
    }

    double Activation::sigmoid(double z, bool deriv){
        if(deriv) { return sigmoid(z) * (1 - sigmoid(z)); }
        return 1 / (1 + exp(-z));
    }

    std::vector<double> Activation::sigmoid(std::vector<double> z, bool deriv){
        if(deriv) {
            LinAlg alg;
            return alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), sigmoid(z)));
         }

        std::vector<double> a;
        a.resize(z.size());

        for(int i = 0; i < z.size(); i++){
            a[i] = sigmoid(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::sigmoid(std::vector<std::vector<double>> z, bool deriv){
        if(deriv) {
            LinAlg alg;
            return alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), sigmoid(z)));
         }

        std::vector<std::vector<double>> a;
        a.resize(z.size());

        for(int i = 0; i < z.size(); i++){
            a[i] = sigmoid(z[i]);
        }
        return a;
    }

    std::vector<double> Activation::softmax(std::vector<double> z){
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            double sum = 0;
            for(int j = 0; j < z.size(); j++){
                sum += exp(z[j]);
            }
            a[i] = exp(z[i]) / sum;
        }
        
        return a;
    }

    std::vector<std::vector<double>> Activation::softmax(std::vector<std::vector<double>> z){
        LinAlg alg;
        std::vector<std::vector<double>> a;
        a.resize(z.size());

        for(int i = 0; i < z.size(); i++){
            a[i] = softmax(z[i]);
        }
        return a;
    }

    std::vector<double> Activation::adjSoftmax(std::vector<double> z){
        LinAlg alg;
        std::vector<double> a;
        double C = -*max_element(z.begin(), z.end());
        z = alg.scalarAdd(C, z);

        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            double sum = 0;
            for(int j = 0; j < z.size(); j++){
                sum += exp(z[j]);
            }
            a[i] = exp(z[i]) / sum;
        }
        
        return a;
    }
    
    std::vector<std::vector<double>> Activation::adjSoftmax(std::vector<std::vector<double>> z){
        LinAlg alg;
        std::vector<std::vector<double>> a;
        a.resize(z.size());

        for(int i = 0; i < z.size(); i++){
            a[i] = adjSoftmax(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::softmaxDeriv(std::vector<double> z){
        LinAlg alg;
        std::vector<std::vector<double>> deriv;
        std::vector<double> a = softmax(z);
        deriv.resize(a.size());
        for(int i = 0; i < deriv.size(); i++){
            deriv[i].resize(a.size());
        }
        for(int i = 0; i < a.size(); i++){
            for(int j = 0; j < z.size(); j++){
                if(i == j){
                    deriv[i][j] = a[i] * (1 - a[i]);
                }
                else{
                    deriv[i][j] = -a[i] * a[j];
                }
            }
        }
        return deriv;
    }

    std::vector<std::vector<std::vector<double>>> Activation::softmaxDeriv(std::vector<std::vector<double>> z){
        LinAlg alg;
        std::vector<std::vector<std::vector<double>>> deriv;
        std::vector<std::vector<double>> a = softmax(z);
        
        deriv.resize(a.size());
        for(int i = 0; i < deriv.size(); i++){
            deriv[i].resize(a.size());
        }
        for(int i = 0; i < a.size(); i++){
            for(int j = 0; j < z.size(); j++){
                if(i == j){
                    deriv[i][j] = alg.subtraction(a[i], alg.hadamard_product(a[i], a[i]));
                }
                else{
                    deriv[i][j] = alg.scalarMultiply(-1, alg.hadamard_product(a[i], a[j])); 
                }
            }
        }
        return deriv;
    }

    double Activation::softplus(double z, bool deriv){
        if(deriv){ return sigmoid(z); }
        return log(1 + exp(z)); 
    }
            
    std::vector<double> Activation::softplus(std::vector<double> z, bool deriv){
        if(deriv) { return sigmoid(z); }
        LinAlg alg;
        return alg.log(alg.addition(alg.onevec(z.size()), alg.exp(z)));
    }
    
    std::vector<std::vector<double>> Activation::softplus(std::vector<std::vector<double>>  z, bool deriv){
        if(deriv) { return sigmoid(z); }
        LinAlg alg;
        return alg.log(alg.addition(alg.onemat(z.size(), z[0].size()), alg.exp(z)));
    }

    double Activation::gaussianCDF(double z, bool deriv){
        if(deriv) {
            return (1 / sqrt(2 * M_PI)) * exp(-z * z / 2);
        }
        return 0.5 * (1 + erf(z / sqrt(2)));
    }

    std::vector<double> Activation::gaussianCDF(std::vector<double> z, bool deriv){
        LinAlg alg; 
        if(deriv) {
            return alg.scalarMultiply(1 / sqrt(2 * M_PI), alg.exp(alg.scalarMultiply(-1/2, alg.hadamard_product(z, z))));
        }
        return alg.scalarMultiply(0.5, alg.addition(alg.onevec(z.size()), alg.erf(alg.scalarMultiply(1/sqrt(2), z))));
    }

    std::vector<std::vector<double>> Activation::gaussianCDF(std::vector<std::vector<double>> z, bool deriv){
        LinAlg alg; 
        if(deriv) {
            return alg.scalarMultiply(1 / sqrt(2 * M_PI), alg.exp(alg.scalarMultiply(-1/2, alg.hadamard_product(z, z))));
        }
        return alg.scalarMultiply(0.5, alg.addition(alg.onemat(z.size(), z[0].size()), alg.erf(alg.scalarMultiply(1/sqrt(2), z))));
    }

    double Activation::cloglog(double z, bool deriv){
        if(deriv) { return exp(z-exp(z)); }
        return 1 - exp(-exp(z));
    }

    std::vector<double> Activation::cloglog(std::vector<double> z, bool deriv){
        LinAlg alg;
        if(deriv) {
            return alg.exp(alg.scalarMultiply(-1, alg.exp(z)));
        }
        return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.exp(alg.scalarMultiply(-1, alg.exp(z)))));
    }

    std::vector<std::vector<double>> Activation::cloglog(std::vector<std::vector<double>> z, bool deriv){
        LinAlg alg;
        if(deriv) {
            return alg.exp(alg.scalarMultiply(-1, alg.exp(z)));
        }
        return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.exp(alg.scalarMultiply(-1, alg.exp(z)))));
    }

    double Activation::unitStep(double z, bool deriv){
        if(deriv) { 
            return 0;
        }
        return z < 0 ? 0 : 1;
    }

    std::vector<double> Activation::unitStep(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = unitStep(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = unitStep(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::unitStep(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = unitStep(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = unitStep(z[i]);
        }
        return a;
    }

    double Activation::swish(double z, bool deriv){
        if(deriv){
            return swish(z) + sigmoid(z) * (1 - swish(z));
        }
        return z * sigmoid(z);
    }

    std::vector<double> Activation::swish(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = swish(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = swish(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::swish(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = swish(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = swish(z[i]);
        }
        return a;
    }

    double Activation::RELU(double z, bool deriv){
        if (deriv){
            if(z <= 0){
                return 0;
            }
            else {
                return 1;
            }
        }
        return fmax(0, z);
    }

    std::vector<double> Activation::RELU(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = RELU(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = RELU(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::RELU(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = RELU(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = RELU(z[i]);
        }
        return a;
    }

    double Activation::leakyReLU(double z, double c, bool deriv){
        if (deriv){
            if(z <= 0){
                return c;
            }
            else {
                return 1;
            }
        }
        return fmax(c * z, z);
    }

    std::vector<double> Activation::leakyReLU(std::vector<double> z, double c, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = leakyReLU(z[i], c, 1);
            }
            return deriv;
        }
        std::vector<double> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = leakyReLU(z[i], c);
        }
        return a;
    }

    double Activation::ELU(double z, double c, bool deriv){
        if (deriv){
            if(z <= 0){
                return c * exp(z);
            }
            else {
                return 1;
            }
        }
        if(z >= 0){
            return z;
        }
        else{
            return c * (exp(z) - 1);
        }
    }

    std::vector<double> Activation::ELU(std::vector<double> z, double c, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = ELU(z[i], c, 1);
            }
            return deriv;
        }
        std::vector<double> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = ELU(z[i], c);
        }
        return a;
    }

    double Activation::SELU(double z, double lambda, double c, bool deriv){
        if (deriv){
            return ELU(z, c, 1);
        }
        return lambda * ELU(z, c);
    }

    std::vector<double> Activation::SELU(std::vector<double> z, double lambda, double c, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = SELU(z[i], lambda, c, 1);
            }
            return deriv;
        }
        std::vector<double> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = SELU(z[i], lambda, c);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::SELU(std::vector<std::vector<double>> z, double lambda, double c, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = SELU(z[i], lambda, c, 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = SELU(z[i], lambda, c);
        }
        return a;
    }

    double Activation::GELU(double z, bool deriv){
        if (deriv){
            return 0.5 * tanh(0.0356774 * pow(z, 3) + 0.797885 * z) + (0.0535161 * pow(z, 3) + 0.398942 * z) * pow(sech(0.0356774 * pow(z, 3) + 0.797885 * z), 2) + 0.5;
        }
        return 0.5 * z * (1 + tanh(sqrt(2/M_PI) * (z + 0.044715 * pow(z, 3))));
    }

    std::vector<double> Activation::GELU(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = GELU(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = GELU(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::GELU(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = GELU(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a; 
        a.resize(z.size());

        for(int i = 0; i < a.size(); i++){
            a[i] = GELU(z[i]);
        }
        return a;
    }

    double Activation::sinh(double z, bool deriv){
        if(deriv){ return cosh(z); }
        return 0.5 * (exp(z) - exp(-z));
    }

    std::vector<double> Activation::sinh(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = sinh(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = sinh(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::sinh(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = sinh(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = sinh(z[i]);
        }
        return a;
    }

    double Activation::cosh(double z, bool deriv){
        if(deriv){ return cosh(z); }
        return 0.5 * (exp(z) + exp(-z));
    }

    std::vector<double> Activation::cosh(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = sinh(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = cosh(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::cosh(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = cosh(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = cosh(z[i]);
        }
        return a;
    }

    double Activation::tanh(double z, bool deriv){
        if(deriv){ return 1 - tanh(z) * tanh(z); }
        return (exp(z) - exp(-z)) / (exp(z) + exp(-z));
    }

    std::vector<double> Activation::tanh(std::vector<double> z, bool deriv){
        LinAlg alg;
        if(deriv){ 
            return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.hadamard_product(tanh(z), tanh(z)))); 
        }
        return alg.elementWiseDivision(alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))), alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
    }

    std::vector<std::vector<double>> Activation::tanh(std::vector<std::vector<double>> z, bool deriv){
        LinAlg alg;
        if(deriv){ 
            return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.hadamard_product(tanh(z), tanh(z)))); 
        }

        return alg.elementWiseDivision(alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))), alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
    }

    double Activation::csch(double z, bool deriv){
        if(deriv){ return -csch(z) * coth(z); }
        return 1 / sinh(z);
    }

    std::vector<double> Activation::csch(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = csch(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = csch(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::csch(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = csch(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = csch(z[i]);
        }
        return a;
    }

    double Activation::sech(double z, bool deriv){
        if(deriv){ return -sech(z) * tanh(z); }
        return 2 / (exp(z) + exp(-z));
    }

    std::vector<double> Activation::sech(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = sech(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = sech(z[i]);
        }
        return a;

        // return activation(z, deriv, static_cast<void (*)(double, bool)>(&sech));
    }

    std::vector<std::vector<double>> Activation::sech(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = sech(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = sech(z[i]);
        }
        return a;

        // return activation(z, deriv, static_cast<void (*)(double, bool)>(&sech));
    }


    double Activation::coth(double z, bool deriv){
        if(deriv){ return -csch(z) * csch(z); }
        return 1 / tanh(z);
    }

    std::vector<double> Activation::coth(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = coth(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = coth(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::coth(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = coth(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = coth(z[i]);
        }
        return a;
    }

    double Activation::arsinh(double z, bool deriv){
        if(deriv){ return 1 / sqrt(z * z + 1); }
        return log(z + sqrt(z * z + 1));
    }

    std::vector<double> Activation::arsinh(std::vector<double> z, bool deriv){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arsinh(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arsinh(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::arsinh(std::vector<std::vector<double>> z, bool deriv){
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arsinh(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arsinh(z[i]);
        }
        return a;
    }

    double Activation::arcosh(double z, bool deriv){
        if(deriv){
            return 1/sqrt(z * z - 1);
        }
        return log(z + sqrt(z * z - 1));
    }

    std::vector<double> Activation::arcosh(std::vector<double> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arcosh(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arcosh(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::arcosh(std::vector<std::vector<double>> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arcosh(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arcosh(z[i]);
        }
        return a;
    }

    double Activation::artanh(double z, bool deriv){
        if(deriv){
            return 1/(1 - z * z);
        }
        return 0.5 * log((1 + z)/(1 - z));
    }

    std::vector<double> Activation::artanh(std::vector<double> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = artanh(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = artanh(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::artanh(std::vector<std::vector<double>> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = artanh(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = artanh(z[i]);
        }
        return a;
    }

    double Activation::arcsch(double z, bool deriv){
        if(deriv){
            return -1/((z * z) * sqrt(1 + (1/(z * z))));
        }
        return log(sqrt(1 + (1 / (z * z))) + (1/z));
    }

    std::vector<double> Activation::arcsch(std::vector<double> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arcsch(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arcsch(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::arcsch(std::vector<std::vector<double>> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arcsch(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arcsch(z[i]);
        }
        return a;
    }


    double Activation::arsech(double z, bool deriv){
        if(deriv){
            return -1/(z * sqrt(1 - z * z));
        }
        return log((1/z) + ((1/z) + 1) * ((1/z) - 1));
    }

    std::vector<double> Activation::arsech(std::vector<double> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arsech(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arsech(z[i]);
        }
        return a;
    }

    std::vector<std::vector<double>> Activation::arsech(std::vector<std::vector<double>> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arsech(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arsech(z[i]);
        }
        return a;
    }

    double Activation::arcoth(double z, bool deriv){
        if(deriv){
            return 1/(1 - z * z);
        }
        return 0.5 * log((1 + z)/(z - 1));
    }

    std::vector<std::vector<double>> Activation::arcoth(std::vector<std::vector<double>> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<std::vector<double>> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arcoth(z[i], 1);
            }
            return deriv;
        }
        std::vector<std::vector<double>> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arcoth(z[i]);
        }
        return a;
    }

    std::vector<double> Activation::arcoth(std::vector<double> z, bool deriv){
        LinAlg alg;
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = arcoth(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = arcoth(z[i]);
        }
        return a;
    }

    // TO DO: Implement this template activation
    std::vector<double> Activation::activation(std::vector<double> z, bool deriv, double(*function)(double, bool)){
        if(deriv){
            std::vector<double> deriv;
            deriv.resize(z.size());
            for(int i = 0; i < z.size(); i++){
                deriv[i] = function(z[i], 1);
            }
            return deriv;
        }
        std::vector<double> a;
        a.resize(z.size());
        for(int i = 0; i < z.size(); i++){
            a[i] = function(z[i], deriv);
        }
        return a;
    }
}