//
//  Stat.cpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "Stat.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Data/Data.hpp"
#include <algorithm>
#include <map>
#include <cmath>

#include <iostream>

namespace MLPP{
    double Stat::b0Estimation(const std::vector<double>& x, const std::vector<double>& y){
        return mean(y) - b1Estimation(x, y) * mean(x);
    }

    double Stat::b1Estimation(const std::vector<double>& x, const std::vector<double>& y){
        return covariance(x, y) / variance(x);
    }

    double Stat::mean(const std::vector<double>& x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += x[i];
        }
        return sum / x.size();
    }

    double Stat::median(std::vector<double> x){
        double center = double(x.size())/double(2); 
        sort(x.begin(), x.end());
        if(x.size() % 2 == 0){
            return mean({x[center - 1], x[center]});
        }
        else{
            return x[center - 1 + 0.5];
        }
    }

    std::vector<double> Stat::mode(const std::vector<double>& x){
        Data data;
        std::vector<double> x_set = data.vecToSet(x);
        std::map<double, int> element_num;
        for(int i = 0; i < x_set.size(); i++){
            element_num[x[i]] = 0;
        }
        for(int i = 0; i < x.size(); i++){
            element_num[x[i]]++;
        }
        std::vector<double> modes;
        double max_num = element_num[x_set[0]];
        for(int i = 0; i < x_set.size(); i++){
            if(element_num[x_set[i]] > max_num){
                max_num = element_num[x_set[i]];
                modes.clear();
                modes.push_back(x_set[i]);
            }
            else if(element_num[x_set[i]] == max_num){
                modes.push_back(x_set[i]);
            }
        }
        return modes;
    }

    double Stat::range(const std::vector<double>& x){
        LinAlg alg;
        return alg.max(x) - alg.min(x);
    }

    double Stat::midrange(const std::vector<double>& x){
        return range(x)/2;
    }

    double Stat::absAvgDeviation(const std::vector<double>& x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += std::abs(x[i] - mean(x));
        }
        return sum / x.size();
    }

    double Stat::standardDeviation(const std::vector<double>& x){
        return std::sqrt(variance(x));
    }

    double Stat::variance(const std::vector<double>& x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += (x[i] - mean(x)) * (x[i] - mean(x));
        }
        return sum / (x.size() - 1);
    }

    double Stat::covariance(const std::vector<double>& x, const std::vector<double>& y){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += (x[i] - mean(x)) * (y[i] - mean(y));
        }
        return sum / (x.size() - 1);
    }

    double Stat::correlation(const std::vector<double>& x, const std::vector<double>& y){
        return covariance(x, y) / (standardDeviation(x) * standardDeviation(y));
    }

    double Stat::R2(const std::vector<double>& x, const std::vector<double>& y){
        return correlation(x, y) * correlation(x, y);
    }

    double Stat::chebyshevIneq(const double k){
        // X may or may not belong to a Gaussian Distribution
        return 1 - 1 / (k * k);
    }

    double Stat::weightedMean(const std::vector<double>& x, const std::vector<double>& weights){
        double sum = 0;
        double weights_sum = 0; 
        for(int i = 0; i < x.size(); i++){
            sum += x[i] * weights[i];
            weights_sum += weights[i];
        }
        return sum / weights_sum;
    }

    double Stat::geometricMean(const std::vector<double>& x){
        double product = 1;
        for(int i = 0; i < x.size(); i++){
            product *= x[i];
        }
        return std::pow(product, 1.0/x.size());
    }

    double Stat::harmonicMean(const std::vector<double>& x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += 1/x[i];
        }
        return x.size()/sum;
    }

    double Stat::RMS(const std::vector<double>& x){
        double sum = 0; 
        for(int i = 0; i < x.size(); i++){
            sum += x[i] * x[i];
        }
        return sqrt(sum / x.size());
    }

    double Stat::powerMean(const std::vector<double>& x, const double p){
        double sum = 0; 
        for(int i = 0; i < x.size(); i++){
            sum += std::pow(x[i], p); 
        }
        return std::pow(sum / x.size(), 1/p);
    }
    
    double Stat::lehmerMean(const std::vector<double>& x, const double p){
        double num = 0; 
        double den = 0; 
        for(int i = 0; i < x.size(); i++){
            num += std::pow(x[i], p); 
            den += std::pow(x[i], p - 1);
        }
        return num/den;
    }

    double Stat::weightedLehmerMean(const std::vector<double>& x, const std::vector<double>& weights, const double p){
        double num = 0; 
        double den = 0; 
        for(int i = 0; i < x.size(); i++){
            num += weights[i] * std::pow(x[i], p); 
            den += weights[i] * std::pow(x[i], p - 1);
        }
        return num/den;
    }

    double Stat::heronianMean(const double A, const double B){
        return (A + sqrt(A * B) + B) / 3;
    }

    double Stat::contraHarmonicMean(const std::vector<double>& x){
        return lehmerMean(x, 2);
    }

    double Stat::heinzMean(const double A, const double B, const double x){
        return (std::pow(A, x) * std::pow(B, 1 - x) + std::pow(A, 1 - x) * std::pow(B, x)) / 2;
    }

    double Stat::neumanSandorMean(const double a, const double b){
        Activation avn;
        return (a - b) / 2 * avn.arsinh((a - b)/(a + b));
    }

    double Stat::stolarskyMean(const double x, const double y, const double p){
        if(x == y){
            return x; 
        }
        return std::pow((std::pow(x, p) - std::pow(y, p)) / (p * (x - y)), 1/(p - 1));
    }

    double Stat::identricMean(const double x, const double y){
        if(x == y){
            return x; 
        }
        return (1/M_E) * std::pow(std::pow(x, x) / std::pow(y, y), 1/(x-y));
    }

    double Stat::logMean(const double x, const double y){
        if(x == y){
            return x; 
        }
        return (y - x) / (log(y) - std::log(x)); 
    }
}