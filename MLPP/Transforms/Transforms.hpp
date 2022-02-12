//
//  Transforms.hpp
//
//

#ifndef Transforms_hpp
#define Transforms_hpp

#include <vector>
#include <string>

namespace MLPP{
    class Transforms{
        public:
            std::vector<std::vector<double>> discreteCosineTransform(std::vector<std::vector<double>> A);
            
    };
}

#endif /* Transforms_hpp */
