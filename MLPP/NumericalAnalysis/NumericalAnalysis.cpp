//
//  NumericalAnalysis.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "NumericalAnalysis.hpp"
#include "LinAlg/LinAlg.hpp"
#include <iostream>
#include <string>
#include <cmath>
#include <climits>

namespace MLPP{

    double NumericalAnalysis::numDiff(double(*function)(double), double x){ 
        double eps = 1e-10;
        return (function(x + eps) - function(x)) / eps; // This is just the formal def. of the derivative.
    }

    
    double NumericalAnalysis::numDiff_2(double(*function)(double), double x){ 
        double eps = 1e-5;
        return (function(x + 2 * eps) -  2 * function(x + eps) + function(x)) / (eps * eps);
    }

    double NumericalAnalysis::numDiff_3(double(*function)(double), double x){ 
        double eps = 1e-5;
        double t1 = function(x + 3 * eps) - 2 * function(x + 2 * eps) + function(x + eps);
        double t2 = function(x + 2 * eps) -  2 * function(x + eps) + function(x);
        return (t1 - t2)/(eps * eps * eps);
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

    double NumericalAnalysis::cubicApproximation(double(*function)(double), double c, double x){
        return quadraticApproximation(function, c, x) + (1/6) * numDiff_3(function, c) * (x - c) * (x - c) * (x - c);
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

    double NumericalAnalysis::numDiff_3(double(*function)(std::vector<double>), std::vector<double> x, int axis1, int axis2, int axis3){
        // For third order derivative tensors. 
        // NOTE: Approximations do not appear to be accurate for sinusodial functions...
        // Should revisit this later. 
        double eps = INT_MAX; 

        std::vector<double> x_ppp = x;
        x_ppp[axis1] += eps; 
        x_ppp[axis2] += eps; 
        x_ppp[axis3] += eps; 

        std::vector<double> x_npp = x;
        x_npp[axis2] += eps; 
        x_npp[axis3] += eps; 
            
        std::vector<double> x_pnp = x;
        x_pnp[axis1] += eps; 
        x_pnp[axis3] += eps; 

        std::vector<double> x_nnp = x;
        x_nnp[axis3] += eps; 


        std::vector<double> x_ppn = x;
        x_ppn[axis1] += eps; 
        x_ppn[axis2] += eps; 

        std::vector<double> x_npn = x;
        x_npn[axis2] += eps; 
            
        std::vector<double> x_pnn = x;
        x_pnn[axis1] += eps;

        double thirdAxis = function(x_ppp) - function(x_npp) - function(x_pnp) + function(x_nnp);
        double noThirdAxis = function(x_ppn) - function(x_npn) - function(x_pnn) + function(x);
        return (thirdAxis - noThirdAxis)/(eps * eps * eps);
    }

    double NumericalAnalysis::newtonRaphsonMethod(double(*function)(double), double x_0, double epoch_num){
        double x = x_0;
        for(int i = 0; i < epoch_num; i++){
            x -= function(x)/numDiff(function, x);
        }
        return x;
    }

    double NumericalAnalysis::halleyMethod(double (*function)(double), double x_0, double epoch_num){
        double x = x_0;
        for(int i = 0; i < epoch_num; i++){
            x -= ((2 * function(x) * numDiff(function, x))/(2 * numDiff(function, x) * numDiff(function, x) - function(x) * numDiff_2(function, x)));
        }
        return x; 
    }

    double NumericalAnalysis::invQuadraticInterpolation(double (*function)(double), std::vector<double> x_0, double epoch_num){
        double x = 0;
        std::vector<double> currentThree = x_0;
        for(int i = 0; i < epoch_num; i++){
            double t1 = ((function(currentThree[1]) * function(currentThree[2]))/( (function(currentThree[0]) - function(currentThree[1])) * (function(currentThree[0]) - function(currentThree[2])) ) ) * currentThree[0];
            double t2 = ((function(currentThree[0]) * function(currentThree[2]))/( (function(currentThree[1]) - function(currentThree[0])) * (function(currentThree[1]) - function(currentThree[2])) ) ) * currentThree[1];
            double t3 = ((function(currentThree[0]) * function(currentThree[1]))/( (function(currentThree[2]) - function(currentThree[0])) * (function(currentThree[2]) - function(currentThree[1])) ) ) * currentThree[2];
            x = t1 + t2 + t3; 

            currentThree.erase(currentThree.begin());
            currentThree.push_back(x);

        }
        return x; 
    }

    double NumericalAnalysis::eulerianMethod(double(*derivative)(double), std::vector<double> q_0, double p, double h){
        double max_epoch = (p - q_0[0])/h; 
        double x = q_0[0];
        double y = q_0[1];
        for(int i = 0; i < max_epoch; i++){
            y = y + h * derivative(x);
            x += h;
        }
        return y;
    }

    double NumericalAnalysis::eulerianMethod(double(*derivative)(std::vector<double>), std::vector<double> q_0, double p, double h){
        double max_epoch = (p - q_0[0])/h; 
        double x = q_0[0];
        double y = q_0[1];
        for(int i = 0; i < max_epoch; i++){
            y = y + h * derivative({x, y});
            x += h;
        }
        return y;
    }

    double NumericalAnalysis::growthMethod(double C, double k, double t){
        /* 
        dP/dt = kP
        dP/P = kdt
        integral(1/P)dP = integral(k) dt
        ln|P| = kt + C_initial
        |P| = e^(kt + C_initial)
        |P| = e^(C_initial) * e^(kt)
        P = +/- e^(C_initial) * e^(kt)
        P = C * e^(kt)
        */

        // auto growthFunction = [&C, &k](double t) { return C * exp(k * t); };
        return C * std::exp(k * t);
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

    std::vector<std::vector<std::vector<double>>> NumericalAnalysis::thirdOrderTensor(double(*function)(std::vector<double>), std::vector<double> x){
        std::vector<std::vector<std::vector<double>>> tensor; 
        tensor.resize(x.size());
        for(int i = 0; i < tensor.size(); i++){
            tensor[i].resize(x.size());
            for(int j = 0; j < tensor[i].size(); j++){
                tensor[i][j].resize(x.size());
            }
        }
        for(int i = 0; i < tensor.size(); i++){ // O(n^3) time complexity :(
            for(int j = 0; j < tensor[i].size(); j++){
                for(int k = 0; k < tensor[i][j].size(); k++)
                tensor[i][j][k] = numDiff_3(function, x, i, j, k);
            }
        }
        return tensor;
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

    double NumericalAnalysis::cubicApproximation(double(*function)(std::vector<double>), std::vector<double> c, std::vector<double> x){
        /* 
        Not completely sure as the literature seldom discusses the third order taylor approximation, 
        in particular for multivariate cases, but ostensibly, the matrix/tensor/vector multiplies 
        should look something like this: 

        (N x N x N) (N x 1) [tensor vector mult] => (N x N x 1) => (N x N)
        Perform remaining multiplies as done for the 2nd order approximation.
        Result is a scalar. 
        */
        LinAlg alg;
        std::vector<std::vector<double>> resultMat = alg.tensor_vec_mult(thirdOrderTensor(function, c), alg.subtraction(x, c));
        double resultScalar = alg.matmult({(alg.subtraction(x, c))}, alg.matmult(resultMat, alg.transpose({alg.subtraction(x, c)})))[0][0];

        return quadraticApproximation(function, c, x) + (1/6) * resultScalar;
    }

    double NumericalAnalysis::laplacian(double(*function)(std::vector<double>), std::vector<double> x){
        LinAlg alg;
        std::vector<std::vector<double>> hessian_matrix = hessian(function, x);
        double laplacian = 0;
        for(int i = 0; i < hessian_matrix.size(); i++){
            laplacian += hessian_matrix[i][i]; // homogenous 2nd derivs w.r.t i, then i
        }
        return laplacian;
    }

    std::string NumericalAnalysis::secondPartialDerivativeTest(double(*function)(std::vector<double>), std::vector<double> x){
        LinAlg alg;
        std::vector<std::vector<double>> hessianMatrix = hessian(function, x);
        /* 
        The reason we do this is because the 2nd partial derivative test is less conclusive for functions of variables greater than
        2, and the calculations specific to the bivariate case are less computationally intensive. 
        */
        if(x.size() == 2){ 
            double det = alg.det(hessianMatrix, hessianMatrix.size());
            double secondDerivative = numDiff_2(function, x, 0, 0);
            if(secondDerivative > 0 && det > 0){
                return "min";
            }
            else if(secondDerivative < 0 && det > 0){
                return "max";
            }
            else if(det < 0){
                return "saddle";
            }
            else{
                return "test was inconclusive";
            }
        }
        else {
            if(alg.positiveDefiniteChecker(hessianMatrix)){
                return "min";
            }
            else if(alg.negativeDefiniteChecker(hessianMatrix)){
                return "max";
            }
            else if(!alg.zeroEigenvalue(hessianMatrix)){
                return "saddle";
            }
            else{
                return "test was inconclusive";
            }
        }
    }
}