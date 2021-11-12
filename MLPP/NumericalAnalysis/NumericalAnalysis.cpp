//
//  NumericalAnalysis.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "NumericalAnalysis.hpp"
#include <iostream>

namespace MLPP{

    double NumericalAnalysis::numDiff(double(*function)(double), double x){ 
        double eps = 1e-10;
        return (function(x + eps) - function(x)) / eps; // This is just the formal def. of the derivative.
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

    double NumericalAnalysis::newtonRaphsonMethod(double(*function)(double), double x_0, double epoch){
        double x = x_0;
        for(int i = 0; i < epoch; i++){
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
}