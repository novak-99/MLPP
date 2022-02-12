//
//  LinAlg.hpp
//
//  Created by Marc Melikyan on 1/8/21.
//

#ifndef LinAlg_hpp
#define LinAlg_hpp

#include <vector>
#include <tuple>

namespace MLPP{
    class LinAlg{
        public:
        
        // MATRIX FUNCTIONS

        std::vector<std::vector<double>> gramMatrix(std::vector<std::vector<double>> A);

        bool linearIndependenceChecker(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> gaussianNoise(int n, int m);

        std::vector<std::vector<double>> addition(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);

        std::vector<std::vector<double>> subtraction(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);
        
        std::vector<std::vector<double>> matmult(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);
        
        std::vector<std::vector<double>> hadamard_product(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);

        std::vector<std::vector<double>> kronecker_product(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);

        std::vector<std::vector<double>> elementWiseDivision(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);
        
        std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> A);
        
        std::vector<std::vector<double>> scalarMultiply(double scalar, std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> scalarAdd(double scalar, std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> log(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> log10(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> exp(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> erf(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> exponentiate(std::vector<std::vector<double>> A, double p);

        std::vector<std::vector<double>> sqrt(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> cbrt(std::vector<std::vector<double>> A);   

        std::vector<std::vector<double>> matrixPower(std::vector<std::vector<double>> A, int n);

        std::vector<std::vector<double>> abs(std::vector<std::vector<double>> A);
        
        double det(std::vector<std::vector<double>> A, int d);

        double trace(std::vector<std::vector<double>> A); 
        
        std::vector<std::vector<double>> cofactor(std::vector<std::vector<double>> A, int n, int i, int j);
        
        std::vector<std::vector<double>> adjoint(std::vector<std::vector<double>> A);
        
        std::vector<std::vector<double>> inverse(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> pinverse(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> zeromat(int n, int m);

        std::vector<std::vector<double>> onemat(int n, int m);

        std::vector<std::vector<double>> full(int n, int m, int k);

        std::vector<std::vector<double>> sin(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> cos(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> rotate(std::vector<std::vector<double>> A, double theta, int axis = -1);

        std::vector<std::vector<double>> max(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);

        double max(std::vector<std::vector<double>> A);

        double min(std::vector<std::vector<double>> A);

        std::vector<std::vector<double>> round(std::vector<std::vector<double>> A);

        double norm_2(std::vector<std::vector<double>> A);
        
        std::vector<std::vector<double>> identity(double d);

        std::vector<std::vector<double>> cov(std::vector<std::vector<double>> A);

        std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> eig(std::vector<std::vector<double>> A);

        std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> SVD(std::vector<std::vector<double>> A);

        std::vector<double> vectorProjection(std::vector<double> a, std::vector<double> b);

        std::vector<std::vector<double>> gramSchmidtProcess(std::vector<std::vector<double>> A);

        std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> QRD(std::vector<std::vector<double>> A);

        std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> chol(std::vector<std::vector<double>> A);

        double sum_elements(std::vector<std::vector<double>> A);

        std::vector<double> flatten(std::vector<std::vector<double>> A);

        std::vector<double> solve(std::vector<std::vector<double>> A, std::vector<double> b);

        bool positiveDefiniteChecker(std::vector<std::vector<double>> A);

        bool negativeDefiniteChecker(std::vector<std::vector<double>> A);

        bool zeroEigenvalue(std::vector<std::vector<double>> A);
        
        void printMatrix(std::vector<std::vector<double>> A);
        
        // VECTOR FUNCTIONS

        std::vector<std::vector<double>> outerProduct(std::vector<double> a, std::vector<double> b); // This multiplies a, bT 
        
        std::vector<double> hadamard_product(std::vector<double> a, std::vector<double> b);

        std::vector<double> elementWiseDivision(std::vector<double> a, std::vector<double> b);
        
        std::vector<double> scalarMultiply(double scalar, std::vector<double> a);

        std::vector<double> scalarAdd(double scalar, std::vector<double> a);
        
        std::vector<double> addition(std::vector<double> a, std::vector<double> b);
        
        std::vector<double> subtraction(std::vector<double> a, std::vector<double> b);

        std::vector<double> subtractMatrixRows(std::vector<double> a, std::vector<std::vector<double>> B);

        std::vector<double> log(std::vector<double> a);

        std::vector<double> log10(std::vector<double> a);

        std::vector<double> exp(std::vector<double> a);

        std::vector<double> erf(std::vector<double> a);

        std::vector<double> exponentiate(std::vector<double> a, double p);

        std::vector<double> sqrt(std::vector<double> a);

        std::vector<double> cbrt(std::vector<double> a);
        
        double dot(std::vector<double> a, std::vector<double> b);

        std::vector<double> cross(std::vector<double> a, std::vector<double> b);

        std::vector<double> abs(std::vector<double> a);

        std::vector<double> zerovec(int n);

        std::vector<double> onevec(int n);

        std::vector<std::vector<double>> diag(std::vector<double> a);

        std::vector<double> full(int n, int k);

        std::vector<double> sin(std::vector<double> a);

        std::vector<double> cos(std::vector<double> a);

        std::vector<double> max(std::vector<double> a, std::vector<double> b);

        double max(std::vector<double> a);

        double min(std::vector<double> a);

        std::vector<double> round(std::vector<double> a);

        double euclideanDistance(std::vector<double> a, std::vector<double> b);
        
        double norm_2(std::vector<double> a);

        double norm_sq(std::vector<double> a);
        
        double sum_elements(std::vector<double> a);

        double cosineSimilarity(std::vector<double> a, std::vector<double> b);
        
        void printVector(std::vector<double> a);
        
        // MATRIX-VECTOR FUNCTIONS
        std::vector<std::vector<double>> mat_vec_add(std::vector<std::vector<double>> A, std::vector<double> b);

        std::vector<double> mat_vec_mult(std::vector<std::vector<double>> A, std::vector<double> b);

        // TENSOR FUNCTIONS
        std::vector<std::vector<std::vector<double>>> addition(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B);

        std::vector<std::vector<std::vector<double>>> elementWiseDivision(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B);

        std::vector<std::vector<std::vector<double>>> sqrt(std::vector<std::vector<std::vector<double>>> A);

        std::vector<std::vector<std::vector<double>>> exponentiate(std::vector<std::vector<std::vector<double>>> A, double p);

        std::vector<std::vector<double>> tensor_vec_mult(std::vector<std::vector<std::vector<double>>> A, std::vector<double> b);

        std::vector<double> flatten(std::vector<std::vector<std::vector<double>>> A);
        
        void printTensor(std::vector<std::vector<std::vector<double>>> A);

        std::vector<std::vector<std::vector<double>>> scalarMultiply(double scalar, std::vector<std::vector<std::vector<double>>> A);

        std::vector<std::vector<std::vector<double>>> scalarAdd(double scalar, std::vector<std::vector<std::vector<double>>> A);

        std::vector<std::vector<std::vector<double>>> resize(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B);

        std::vector<std::vector<std::vector<double>>> hadamard_product(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B);

        std::vector<std::vector<std::vector<double>>> max(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B);

        std::vector<std::vector<std::vector<double>>> abs(std::vector<std::vector<std::vector<double>>> A);

        double norm_2(std::vector<std::vector<std::vector<double>>> A);

        std::vector<std::vector<std::vector<double>>> vector_wise_tensor_product(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<double>> B);

        private:
    };

}

#endif /* LinAlg_hpp */