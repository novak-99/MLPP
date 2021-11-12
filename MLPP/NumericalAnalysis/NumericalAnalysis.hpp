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
            double numDiff(double(*function)(std::vector<double>), std::vector<double> x, int axis);
            double newtonRaphsonMethod(double(*function)(double), double x_0, double epoch);

            std::vector<double> jacobian(double(*function)(std::vector<double>), std::vector<double>);
        
    };
}

#endif /* NumericalAnalysis_hpp */
