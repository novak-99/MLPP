//
//  Reg.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#ifndef Reg_hpp
#define Reg_hpp

#include <vector>

namespace MLPP{
    class Reg{
        public:
        
            double regTerm(std::vector<double> weights, double lambda, double alpha, std::string reg);
            double regTerm(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg);
            
            std::vector<double> regWeights(std::vector<double> weights, double lambda, double alpha, std::string reg);
            std::vector<std::vector<double>> regWeights(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg);

            std::vector<double> regDerivTerm(std::vector<double> weights, double lambda, double alpha, std::string reg);
            std::vector<std::vector<double>> regDerivTerm(std::vector<std::vector<double>>, double lambda, double alpha, std::string reg);

        private:
            double regDerivTerm(std::vector<double> weights, double lambda, double alpha, std::string reg, int j);
            double regDerivTerm(std::vector<std::vector<double>> weights, double lambda, double alpha, std::string reg, int i, int j);
    };
}

#endif /* Reg_hpp */
