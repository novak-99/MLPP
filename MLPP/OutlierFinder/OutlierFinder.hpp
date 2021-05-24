//
//  OutlierFinder.hpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#ifndef OutlierFinder_hpp
#define OutlierFinder_hpp

#include <vector>

namespace MLPP{
    class OutlierFinder{
        public:
            // Cnstr
            OutlierFinder(int threshold);

            std::vector<std::vector<double>> modelSetTest(std::vector<std::vector<double>> inputSet);
            std::vector<double> modelTest(std::vector<double> inputSet);

            // Variables required 
            int threshold;
        
    };
}

#endif /* OutlierFinder_hpp */
