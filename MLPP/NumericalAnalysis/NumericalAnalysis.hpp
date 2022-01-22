//
//  NumericalAnalysis.hpp
//
//

#ifndef NumericalAnalysis_hpp
#define NumericalAnalysis_hpp

#include <vector>
#include <string>

namespace MLPP{
    class NumericalAnalysis{
        public:
            /* A numerical method for derivatives is used. This may be subject to change,
            as an analytical method for calculating derivatives will most likely be used in
            the future.
            */
            double numDiff(double(*function)(double), double x);
            double numDiff_2(double(*function)(double), double x);
            double numDiff_3(double(*function)(double), double x);

            double constantApproximation(double(*function)(double), double c);
            double linearApproximation(double(*function)(double), double c, double x);
            double quadraticApproximation(double(*function)(double), double c, double x);
            double cubicApproximation(double(*function)(double), double c, double x);

            double numDiff(double(*function)(std::vector<double>), std::vector<double> x, int axis);
            double numDiff_2(double(*function)(std::vector<double>), std::vector<double> x, int axis1, int axis2);
            double numDiff_3(double(*function)(std::vector<double>), std::vector<double> x, int axis1, int axis2, int axis3);

            double newtonRaphsonMethod(double(*function)(double), double x_0, double epoch_num);
            double halleyMethod(double(*function)(double), double x_0, double epoch_num);
            double invQuadraticInterpolation(double (*function)(double), std::vector<double> x_0, double epoch_num);

            double eulerianMethod(double(*derivative)(double), std::vector<double> q_0, double p, double h); // Euler's method for solving diffrential equations. 
            double eulerianMethod(double(*derivative)(std::vector<double>), std::vector<double> q_0, double p, double h); // Euler's method for solving diffrential equations. 

            double growthMethod(double C, double k, double t); // General growth-based diffrential equations can be solved by seperation of variables.

            std::vector<double> jacobian(double(*function)(std::vector<double>), std::vector<double> x); // Indeed, for functions with scalar outputs the Jacobians will be vectors.
            std::vector<std::vector<double>> hessian(double(*function)(std::vector<double>), std::vector<double> x);
            std::vector<std::vector<std::vector<double>>> thirdOrderTensor(double(*function)(std::vector<double>), std::vector<double> x);

            double constantApproximation(double(*function)(std::vector<double>), std::vector<double> c);
            double linearApproximation(double(*function)(std::vector<double>), std::vector<double> c, std::vector<double> x);
            double quadraticApproximation(double(*function)(std::vector<double>), std::vector<double> c, std::vector<double> x);
            double cubicApproximation(double(*function)(std::vector<double>), std::vector<double> c, std::vector<double> x);

            double laplacian(double(*function)(std::vector<double>), std::vector<double> x); // laplacian

            std::string secondPartialDerivativeTest(double(*function)(std::vector<double>), std::vector<double> x);

    };
}

#endif /* NumericalAnalysis_hpp */
