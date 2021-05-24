//
//  Stat.hpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#ifndef Stat_hpp
#define Stat_hpp

#include <vector>

namespace MLPP{
    class Stat{
      
        public:
            double b0Estimation(std::vector<double> x, std::vector<double> y);
            double b1Estimation(std::vector<double> x, std::vector<double> y);
        
            // Statistical Functions
            double mean(std::vector <double> x);
            double variance(std::vector <double> x);
            double covariance(std::vector <double> x, std::vector <double> y);
            double correlation(std::vector <double> x, std::vector<double> y);
            double R2(std::vector <double> x, std::vector<double> y);

            // Extras
            double weightedMean(std::vector<double> x, std::vector<double> weights);
            double geometricMean(std::vector <double> x);
            double harmonicMean(std::vector <double> x);
            double RMS(std::vector<double> x);
            double powerMean(std::vector<double> x, double p);
            double lehmerMean(std::vector<double> x, double p);
            double weightedLehmerMean(std::vector<double> x, std::vector<double> weights, double p);
            double contraharmonicMean(std::vector<double> x);
            double heronianMean(double A, double B);
            double heinzMean(double A, double B, double x);
            double neumanSandorMean(double a, double b);
            double stolarskyMean(double x, double y, double p);
            double identricMean(double x, double y);
            double logMean(double x, double y);
            double standardDeviation(std::vector <double> x);
            double absAvgDeviation(std::vector <double> x);
            double chebyshevIneq(double k);

        private:

        
        
            
    };
}

#endif /* Stat_hpp */
