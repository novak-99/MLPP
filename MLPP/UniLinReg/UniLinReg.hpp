//
//  UniLinReg.hpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#ifndef UniLinReg_hpp
#define UniLinReg_hpp

#include <vector>

namespace MLPP{
    class UniLinReg{
        
        public:
            UniLinReg(std::vector <double> x, std::vector<double> y);
            std::vector<double> modelSetTest(std::vector<double> x);
            double modelTest(double x);
        
        private:
            std::vector <double> inputSet;
            std::vector <double> outputSet;
        
            double b0;
            double b1;
        
    };
}

#endif /* UniLinReg_hpp */
