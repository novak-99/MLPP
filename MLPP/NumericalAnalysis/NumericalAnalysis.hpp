//
//  NumericalAnalysis.hpp
//
//

#ifndef NumericalAnalysis_hpp
#define NumericalAnalysis_hpp

#include <vector>

namespace MLPP{
    class NumericalAnalysis{
        public:
            /* A numerical method for derivatives is used. This may be subject to change,
            as an analytical method for calculating derivatives will most likely be used in
            the future.
            */
            double numDiff(double(*function)(double), double x);
            double numDiff_2(double(*function)(double), double x);

            double constantApproximation(double(*function)(double), double c);
            double linearApproximation(double(*function)(double), double c, double x);
            double quadraticApproximation(double(*function)(double), double c, double x);

            double numDiff(double(*function)(std::vector<double>), std::vector<double> x, int axis);
            double numDiff_2(double(*function)(std::vector<double>), std::vector<double> x, int axis1, int axis2);

            double newtonRaphsonMethod(double(*function)(double), double x_0, double epoch_num);

            std::vector<double> jacobian(double(*function)(std::vector<double>), std::vector<double> x); // Indeed, for functions with scalar outputs the Jacobians will be vectors.
            std::vector<std::vector<double>> hessian(double(*function)(std::vector<double>), std::vector<double> x);

            double constantApproximation(double(*function)(std::vector<double>), std::vector<double> c);
            double linearApproximation(double(*function)(std::vector<double>), std::vector<double> c, std::vector<double> x);
            double quadraticApproximation(double(*function)(std::vector<double>), std::vector<double> c, std::vector<double> x);

        
    };
}

#endif /* NumericalAnalysis_hpp */
