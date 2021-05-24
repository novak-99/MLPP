//
//  HypothesisTesting.hpp
//
//  Created by Marc Melikyan on 3/10/21.
//

#ifndef HypothesisTesting_hpp
#define HypothesisTesting_hpp

#include <vector>
#include <tuple>

namespace MLPP{
    class HypothesisTesting{
      
        public:
            std::tuple<bool, double> chiSquareTest(std::vector<double> observed, std::vector<double> expected);
        
        private:
            
    };
}

#endif /* HypothesisTesting_hpp */
