//
//  NumericalAnalysis.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "NumericalAnalysis.hpp"
#include "LinAlg/LinAlg.hpp"
#include <iostream>

namespace MLPP{

    double NumericalAnalysis::numDiff(double(*function)(double), double x){ 
        double eps = 1e-10;
        return (function(x + eps) - function(x)) / eps; // This is just the formal def. of the derivative.
    }

    
    double NumericalAnalysis::numDiff_2(double(*function)(double), double x){ 
        double eps = 1e-5;
        return (function(x + eps) -  2 * function(x) + function(x - eps)) / (eps * eps);
    }

    double  NumericalAnalysis::constantApproximation(double(*function)(double), double c){
        return function(c);
    }

    double  NumericalAnalysis::linearApproximation(double(*function)(double), double c, double x){
        return constantApproximation(function, c) + numDiff(function, c) * (x - c);
    }

    double NumericalAnalysis::quadraticApproximation(double(*function)(double), double c, double x){
        return linearApproximation(function, c, x) + 0.5 * numDiff_2(function, c) * (x - c) * (x - c);
    }

    double NumericalAnalysis::numDiff(double(*function)(std::vector<double>), std::vector<double> x, int axis){
        // For multivariable function analysis. 
        // This will be used for calculating Jacobian vectors. 
        // Diffrentiate with respect to indicated axis. (0, 1, 2 ...)
        double eps = 1e-10;
        std::vector<double> x_eps = x;
        x_eps[axis] += eps;

        return (function(x_eps) - function(x)) / eps; 
    }

    double NumericalAnalysis::numDiff_2(double(*function)(std::vector<double>), std::vector<double> x, int axis1, int axis2){
        //For Hessians. 
        double eps = 1e-5;

        std::vector<double> x_pp = x;
        x_pp[axis1] += eps; 
        x_pp[axis2] += eps; 

        std::vector<double> x_np = x;
        x_np[axis2] += eps; 
            
        std::vector<double> x_pn = x;
        x_pn[axis1] += eps;

        return (function(x_pp) - function(x_np) - function(x_pn) + function(x))/(eps * eps);
    }

    double NumericalAnalysis::newtonRaphsonMethod(double(*function)(double), double x_0, double epoch_num){
        double x = x_0;
        for(int i = 0; i < epoch_num; i++){
            x = x - function(x)/numDiff(function, x);
        }
        return x;
    }
    
    std::vector<double> NumericalAnalysis::jacobian(double(*function)(std::vector<double>), std::vector<double> x){
        std::vector<double> jacobian; 
        jacobian.resize(x.size());
        for(int i = 0; i < jacobian.size(); i++){
            jacobian[i] = numDiff(function, x, i); // Derivative w.r.t axis i evaluated at x. For all x_i.
        }
        return jacobian;
    }
    std::vector<std::vector<double>> NumericalAnalysis::hessian(double(*function)(std::vector<double>), std::vector<double> x){
        std::vector<std::vector<double>> hessian; 
        hessian.resize(x.size());
        for(int i = 0; i < hessian.size(); i++){
            hessian[i].resize(x.size());
        }
        for(int i = 0; i < hessian.size(); i++){
            for(int j = 0; j < hessian[i].size(); j++){
                hessian[i][j] = numDiff_2(function, x, i, j);
            }
        }
        return hessian;
    }

    double NumericalAnalysis::constantApproximation(double(*function)(std::vector<double>), std::vector<double> c){
        return function(c);
    }

    double NumericalAnalysis::linearApproximation(double(*function)(std::vector<double>), std::vector<double> c, std::vector<double> x){
        LinAlg alg;
        return constantApproximation(function, c) + alg.matmult(alg.transpose({jacobian(function, c)}), {alg.subtraction(x, c)})[0][0];
    }

    double NumericalAnalysis::quadraticApproximation(double(*function)(std::vector<double>), std::vector<double> c, std::vector<double> x){
        LinAlg alg;
        return linearApproximation(function, c, x) + 0.5 * alg.matmult({(alg.subtraction(x, c))}, alg.matmult(hessian(function, c), alg.transpose({alg.subtraction(x, c)})))[0][0];
    }
}