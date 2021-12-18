#ifndef Convolutions_hpp
#define Convolutions_hpp

#include <vector>

namespace MLPP{
    class Convolutions{
        public:
            Convolutions();
            std::vector<std::vector<double>> convolve(std::vector<std::vector<double>> input, std::vector<std::vector<double>> filter, int S, int P = 0);
            std::vector<std::vector<std::vector<double>>> convolve(std::vector<std::vector<std::vector<double>>> input, std::vector<std::vector<std::vector<double>>> filter, int S, int P = 0);
            std::vector<std::vector<double>> pool(std::vector<std::vector<double>> input, int F, int S, std::string type);
            std::vector<std::vector<std::vector<double>>> pool(std::vector<std::vector<std::vector<double>>> input, int F, int S, std::string type);
            double globalPool(std::vector<std::vector<double>> input, std::string type);
            std::vector<double> globalPool(std::vector<std::vector<std::vector<double>>> input, std::string type);

            double gaussian2D(double x, double y, double std);
            std::vector<std::vector<double>> gaussianFilter2D(int size, double std);

            std::vector<std::vector<double>> dx(std::vector<std::vector<double>> input);
            std::vector<std::vector<double>> dy(std::vector<std::vector<double>> input);

            std::vector<std::vector<double>> gradMagnitude(std::vector<std::vector<double>> input);
            std::vector<std::vector<double>> gradOrientation(std::vector<std::vector<double>> input);

            std::vector<std::vector<std::vector<double>>> computeM(std::vector<std::vector<double>> input);
            std::vector<std::vector<std::string>> harrisCornerDetection(std::vector<std::vector<double>> input);

            std::vector<std::vector<double>> getPrewittHorizontal();
            std::vector<std::vector<double>> getPrewittVertical();
            std::vector<std::vector<double>> getSobelHorizontal();
            std::vector<std::vector<double>> getSobelVertical();
            std::vector<std::vector<double>> getScharrHorizontal();
            std::vector<std::vector<double>> getScharrVertical();
            std::vector<std::vector<double>> getRobertsHorizontal();
            std::vector<std::vector<double>> getRobertsVertical();
 
        private: 
            std::vector<std::vector<double>> prewittHorizontal;
            std::vector<std::vector<double>> prewittVertical;
            std::vector<std::vector<double>> sobelHorizontal;
            std::vector<std::vector<double>> sobelVertical;
            std::vector<std::vector<double>> scharrHorizontal;
            std::vector<std::vector<double>> scharrVertical;
            std::vector<std::vector<double>> robertsHorizontal;
            std::vector<std::vector<double>> robertsVertical;

    };
}

#endif // Convolutions_hpp