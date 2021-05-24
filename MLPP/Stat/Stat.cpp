//
//  Stat.cpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "Stat.hpp"
#include "Activation/Activation.hpp"
#include <cmath>

namespace MLPP{
    double Stat::b0Estimation(std::vector<double> x, std::vector<double> y){
        return mean(y) - b1Estimation(x, y) * mean(x);
    }

    double Stat::b1Estimation(std::vector<double> x, std::vector<double> y){
        return covariance(x, y) / variance(x);
    }

    double Stat::mean(std::vector<double> x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += x[i];
        }
        return sum / x.size();
    }

    double Stat::variance(std::vector<double> x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += (x[i] - mean(x)) * (x[i] - mean(x));
        }
        return sum / (x.size() - 1);
    }

    double Stat::covariance(std::vector<double> x, std::vector<double> y){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += (x[i] - mean(x)) * (y[i] - mean(y));
        }
        return sum / (x.size() - 1);
    }

    double Stat::correlation(std::vector<double> x, std::vector<double> y){
        return covariance(x, y) / (standardDeviation(x) * standardDeviation(y));
    }

    double Stat::R2(std::vector<double> x, std::vector<double> y){
        return correlation(x, y) * correlation(x, y);
    }

    double Stat::weightedMean(std::vector<double> x, std::vector<double> weights){
        double sum = 0;
        double weights_sum = 0; 
        for(int i = 0; i < x.size(); i++){
            sum += x[i] * weights[i];
            weights_sum += weights[i];
        }
        return sum / weights_sum;
    }

    double Stat::geometricMean(std::vector<double> x){
        double product = 1;
        for(int i = 0; i < x.size(); i++){
            product *= x[i];
        }
        return std::pow(product, 1.0/x.size());
    }

    double Stat::harmonicMean(std::vector<double> x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += 1/x[i];
        }
        return x.size()/sum;
    }

    double Stat::RMS(std::vector<double> x){
        double sum = 0; 
        for(int i = 0; i < x.size(); i++){
            sum += x[i] * x[i];
        }
        return sqrt(sum / x.size());
    }

    double Stat::powerMean(std::vector<double> x, double p){
        double sum = 0; 
        for(int i = 0; i < x.size(); i++){
            sum += pow(x[i], p); 
        }
        return pow(sum / x.size(), 1/p);
    }
    
    double Stat::lehmerMean(std::vector<double> x, double p){
        double num = 0; 
        double den = 0; 
        for(int i = 0; i < x.size(); i++){
            num += pow(x[i], p); 
            den += pow(x[i], p - 1);
        }
        return num/den;
    }

    double Stat::weightedLehmerMean(std::vector<double> x, std::vector<double> weights, double p){
        double num = 0; 
        double den = 0; 
        for(int i = 0; i < x.size(); i++){
            num += weights[i] * pow(x[i], p); 
            den += weights[i] * pow(x[i], p - 1);
        }
        return num/den;
    }

    double Stat::heronianMean(double A, double B){
        return (A + sqrt(A * B) + B) / 3;
    }

    double Stat::contraharmonicMean(std::vector<double> x){
        return lehmerMean(x, 2);
    }

    double Stat::heinzMean(double A, double B, double x){
        return (pow(A, x) * pow(B, 1 - x) + pow(A, 1 - x) * pow(B, x)) / 2;
    }

    double Stat::neumanSandorMean(double a, double b){
        Activation avn;
        return (a - b) / 2 * avn.arsinh((a - b)/(a + b));
    }

    double Stat::stolarskyMean(double x, double y, double p){
        if(x == y){
            return x; 
        }
        return pow((pow(x, p) - pow(y, p)) / (p * (x - y)), 1/(p - 1));
    }

    double Stat::identricMean(double x, double y){
        if(x == y){
            return x; 
        }
        return (1/M_E) * pow(pow(x, x) / pow(y, y), 1/(x-y));
    }

    double Stat::logMean(double x, double y){
        if(x == y){
            return x; 
        }
        return (y - x) / (log(y) - log(x)); 
    }

    double Stat::standardDeviation(std::vector<double> x){
        return std::sqrt(variance(x));
    }

    double Stat::absAvgDeviation(std::vector<double> x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += std::abs(x[i] - mean(x));
        }
        return sum / x.size();
    }

    double Stat::chebyshevIneq(double k){
        //Pr(|X - mu| >= k * sigma) <= 1/k^2, X may or may not belong to a Gaussian Distribution
        return 1 - 1 / (k * k);
    }
}