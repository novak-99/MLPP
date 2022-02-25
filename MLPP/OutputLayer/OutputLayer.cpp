//
//  OutputLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "OutputLayer.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Utilities/Utilities.hpp"

#include <iostream>
#include <random>

namespace MLPP {
    OutputLayer::OutputLayer(int n_hidden, std::string activation, std::string cost, std::vector<std::vector<double>> input, std::string weightInit, std::string reg, double lambda, double alpha)
    : n_hidden(n_hidden), activation(activation), cost(cost), input(input), weightInit(weightInit), reg(reg), lambda(lambda), alpha(alpha)
    {
        weights = Utilities::weightInitialization(n_hidden, weightInit);
        bias = Utilities::biasInitialization();

        activation_map["Linear"] = &Activation::linear;
        activationTest_map["Linear"] = &Activation::linear;

        activation_map["Sigmoid"] = &Activation::sigmoid;
        activationTest_map["Sigmoid"] = &Activation::sigmoid;

        activation_map["Swish"] = &Activation::swish;
        activationTest_map["Swish"] = &Activation::swish;

        activation_map["Mish"] = &Activation::mish;
        activationTest_map["Mish"] = &Activation::mish;

        activation_map["SinC"] = &Activation::sinc;
        activationTest_map["SinC"] = &Activation::sinc;

        activation_map["Softplus"] = &Activation::softplus;
        activationTest_map["Softplus"] = &Activation::softplus;

        activation_map["Softsign"] = &Activation::softsign;
        activationTest_map["Softsign"] = &Activation::softsign;

        activation_map["CLogLog"] = &Activation::cloglog;
        activationTest_map["CLogLog"] = &Activation::cloglog;

        activation_map["Logit"] = &Activation::logit;
        activationTest_map["Logit"] = &Activation::logit;

        activation_map["GaussianCDF"] = &Activation::gaussianCDF;
        activationTest_map["GaussianCDF"] = &Activation::gaussianCDF;

        activation_map["RELU"] = &Activation::RELU;
        activationTest_map["RELU"] = &Activation::RELU;

        activation_map["GELU"] = &Activation::GELU;
        activationTest_map["GELU"] = &Activation::GELU;

        activation_map["Sign"] = &Activation::sign;
        activationTest_map["Sign"] = &Activation::sign;

        activation_map["UnitStep"] = &Activation::unitStep;
        activationTest_map["UnitStep"] = &Activation::unitStep;

        activation_map["Sinh"] = &Activation::sinh;
        activationTest_map["Sinh"] = &Activation::sinh;

        activation_map["Cosh"] = &Activation::cosh;
        activationTest_map["Cosh"] = &Activation::cosh;

        activation_map["Tanh"] = &Activation::tanh;
        activationTest_map["Tanh"] = &Activation::tanh;

        activation_map["Csch"] = &Activation::csch;
        activationTest_map["Csch"] = &Activation::csch;   

        activation_map["Sech"] = &Activation::sech;
        activationTest_map["Sech"] = &Activation::sech;  

        activation_map["Coth"] = &Activation::coth;
        activationTest_map["Coth"] = &Activation::coth;  

        activation_map["Arsinh"] = &Activation::arsinh;
        activationTest_map["Arsinh"] = &Activation::arsinh;

        activation_map["Arcosh"] = &Activation::arcosh;
        activationTest_map["Arcosh"] = &Activation::arcosh;

        activation_map["Artanh"] = &Activation::artanh;
        activationTest_map["Artanh"] = &Activation::artanh;

        activation_map["Arcsch"] = &Activation::arcsch;
        activationTest_map["Arcsch"] = &Activation::arcsch;

        activation_map["Arsech"] = &Activation::arsech;
        activationTest_map["Arsech"] = &Activation::arsech;

        activation_map["Arcoth"] = &Activation::arcoth;
        activationTest_map["Arcoth"] = &Activation::arcoth;

        costDeriv_map["MSE"] = &Cost::MSEDeriv;
        cost_map["MSE"] = &Cost::MSE;
        costDeriv_map["RMSE"] = &Cost::RMSEDeriv;
        cost_map["RMSE"] = &Cost::RMSE;
        costDeriv_map["MAE"] = &Cost::MAEDeriv;
        cost_map["MAE"] = &Cost::MAE;
        costDeriv_map["MBE"] = &Cost::MBEDeriv;
        cost_map["MBE"] = &Cost::MBE;
        costDeriv_map["LogLoss"] = &Cost::LogLossDeriv;
        cost_map["LogLoss"] = &Cost::LogLoss;
        costDeriv_map["CrossEntropy"] = &Cost::CrossEntropyDeriv;
        cost_map["CrossEntropy"] = &Cost::CrossEntropy;
        costDeriv_map["HingeLoss"] = &Cost::HingeLossDeriv;
        cost_map["HingeLoss"] = &Cost::HingeLoss;
        costDeriv_map["WassersteinLoss"] = &Cost::HingeLossDeriv;
        cost_map["WassersteinLoss"] = &Cost::HingeLoss;
    }
    
    void OutputLayer::forwardPass(){
        LinAlg alg;
        Activation avn;
        z = alg.scalarAdd(bias, alg.mat_vec_mult(input, weights));
        a = (avn.*activation_map[activation])(z, 0); 
    }

    void OutputLayer::Test(std::vector<double> x){
        LinAlg alg;
        Activation avn;
        z_test = alg.dot(weights, x) + bias;
        a_test = (avn.*activationTest_map[activation])(z_test, 0);
    }
}