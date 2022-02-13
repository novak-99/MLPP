//
//  LinAlg.cpp
//
//  Created by Marc Melikyan on 1/8/21.
//

#include "LinAlg.hpp"
#include "Stat/Stat.hpp"
#include <iostream>
#include <random>
#include <map>
#include <cmath>

namespace MLPP{

    std::vector<std::vector<double>> LinAlg::gramMatrix(std::vector<std::vector<double>> A){
        return matmult(transpose(A), A); // AtA
    }

    bool LinAlg::linearIndependenceChecker(std::vector<std::vector<double>> A){
        if (det(gramMatrix(A), A.size()) == 0){
            return false; 
        }
        return true; 
    }

    std::vector<std::vector<double>> LinAlg::gaussianNoise(int n, int m){
        std::random_device rd;
        std::default_random_engine generator(rd());

        std::vector<std::vector<double>> A;
        A.resize(n);
        for(int i = 0; i < n; i++){
            A[i].resize(m);
            for(int j = 0; j < m; j++){
                std::normal_distribution<double> distribution(0, 1); // Standard normal distribution. Mean of 0, std of 1. 
                A[i][j] = distribution(generator);
            }
        }
        return A; 
    }

    std::vector<std::vector<double>> LinAlg::addition(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
        std::vector<std::vector<double>> C;
        C.resize(A.size());
        for(int i = 0; i < C.size(); i++){
            C[i].resize(A[0].size());
        }
        
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[0].size(); j++){
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        return C;
    }

    std::vector<std::vector<double>> LinAlg::subtraction(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
        std::vector<std::vector<double>> C;
        C.resize(A.size());
        for(int i = 0; i < C.size(); i++){
            C[i].resize(A[0].size());
        }

        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[0].size(); j++){
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    std::vector<std::vector<double>> LinAlg::matmult(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
        std::vector<std::vector<double>> C;
        C.resize(A.size());
        for(int i = 0; i < C.size(); i++){
            C[i].resize(B[0].size());
        }
        
        for(int i = 0; i < A.size(); i++){ 
            for(int k = 0; k < B.size(); k++){ 
                for(int j = 0; j < B[0].size(); j++){ 
                    C[i][j] += A[i][k] * B[k][j]; 
                } 
            } 
        } 
        return C;
    }

    std::vector<std::vector<double>> LinAlg::hadamard_product(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
        std::vector<std::vector<double>> C;
        C.resize(A.size());
        for(int i = 0; i < C.size(); i++){
            C[i].resize(A[0].size());
        }
        
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[0].size(); j++){
                C[i][j] = A[i][j] * B[i][j];
            }
        }
        return C;
    }

    std::vector<std::vector<double>> LinAlg::kronecker_product(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
        std::vector<std::vector<double>> C;

        // [1,1,1,1]   [1,2,3,4,5]
        // [1,1,1,1]   [1,2,3,4,5]    
        //             [1,2,3,4,5]

        // [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
        // [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
        // [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
        // [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
        // [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
        // [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]

        // Resulting matrix: A.size() * B.size()
        //                   A[0].size() * B[0].size()

        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < B.size(); j++){
                std::vector<std::vector<double>> row;
                for(int k = 0; k < A[0].size(); k++){
                    row.push_back(scalarMultiply(A[i][k], B[j]));
                } 
                C.push_back(flatten(row));
            }
        }
        return C;    
    }

    std::vector<std::vector<double>> LinAlg::elementWiseDivision(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
        std::vector<std::vector<double>> C;
        C.resize(A.size());
        for(int i = 0; i < C.size(); i++){
            C[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                C[i][j] = A[i][j] / B[i][j];
            }
        }
        return C;
    }

    std::vector<std::vector<double>> LinAlg::transpose(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> AT;
        AT.resize(A[0].size());
        for(int i = 0; i < AT.size(); i++){
            AT[i].resize(A.size());
        }
        
        for(int i = 0; i < A[0].size(); i++){
            for(int j = 0; j < A.size(); j++){
                AT[i][j] = A[j][i];
            }
        }
        return AT;
    }

    std::vector<std::vector<double>> LinAlg::scalarMultiply(double scalar, std::vector<std::vector<double>> A){
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                A[i][j] *= scalar;
            }
        }
        return A;
    }

    std::vector<std::vector<double>> LinAlg::scalarAdd(double scalar, std::vector<std::vector<double>> A){
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                A[i][j] += scalar;
            }
        }
        return A;
    }

    std::vector<std::vector<double>> LinAlg::log(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                B[i][j] = std::log(A[i][j]);
            }
        }
        return B;
    }

    std::vector<std::vector<double>> LinAlg::log10(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                B[i][j] = std::log10(A[i][j]);
            }
        }
        return B;
    }

    std::vector<std::vector<double>> LinAlg::exp(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                B[i][j] = std::exp(A[i][j]);
            }
        }
        return B;
    }

    std::vector<std::vector<double>> LinAlg::erf(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                B[i][j] = std::erf(A[i][j]);
            }
        }
        return B;
    }

    std::vector<std::vector<double>> LinAlg::exponentiate(std::vector<std::vector<double>> A, double p){
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                A[i][j] = std::pow(A[i][j], p); 
            }
        }
        return A; 
    }

    std::vector<std::vector<double>> LinAlg::sqrt(std::vector<std::vector<double>> A){
        return exponentiate(A, 0.5); 
    }

    std::vector<std::vector<double>> LinAlg::cbrt(std::vector<std::vector<double>> A){
        return exponentiate(A, double(1)/double(3)); 
    }

    std::vector<std::vector<double>> LinAlg::matrixPower(std::vector<std::vector<double>> A, int n){
        std::vector<std::vector<double>> B = identity(A.size());
        if(n == 0){
            return identity(A.size());
        }
        else if(n < 0){
            A = inverse(A);
        }
        for(int i = 0; i < std::abs(n); i++){
            B = matmult(B, A);
        }
        return B;
    }

    std::vector<std::vector<double>> LinAlg::abs(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }
        for(int i = 0; i < B.size(); i++){
            for(int j = 0; j < B[i].size(); j++){
                B[i][j] = std::abs(A[i][j]);
            }
        }
        return B;
    }

    double LinAlg::det(std::vector<std::vector<double>> A, int d){

        double deter = 0;
        std::vector<std::vector<double>> B;
        B.resize(d);
        for(int i = 0; i < d; i++){
            B[i].resize(d);
        }

        /* This is the base case in which the input is a 2x2 square matrix.
        Recursion is performed unless and until we reach this base case,
        such that we recieve a scalar as the result. */
        if(d == 2){
            return A[0][0] * A[1][1] - A[0][1] * A[1][0];
        }

        else{
            for(int i = 0; i < d; i++){
                int sub_i = 0;
                for(int j = 1; j < d; j++){
                    int sub_j = 0;
                    for(int k = 0; k < d; k++){
                        if(k == i){
                            continue;
                        }
                        B[sub_i][sub_j] = A[j][k];
                        sub_j++;
                    }
                    sub_i++;
                }
                deter += std::pow(-1, i) * A[0][i] * det(B, d-1);
            }
        }
        return deter;
    }

    double LinAlg::trace(std::vector<std::vector<double>> A){
        double trace = 0;
        for(int i = 0; i < A.size(); i++){
            trace += A[i][i];
        }
        return trace;
    }

    std::vector<std::vector<double>> LinAlg::cofactor(std::vector<std::vector<double>> A, int n, int i, int j){
        std::vector<std::vector<double>> cof;
        cof.resize(A.size());
        for(int i = 0; i < cof.size(); i++){
          cof[i].resize(A.size());
        }
        int sub_i = 0, sub_j = 0;
      
        for (int row = 0; row < n; row++){
            for (int col = 0; col < n; col++){
                if (row != i && col != j) {
                    cof[sub_i][sub_j++] = A[row][col];
      
                    if (sub_j == n - 1){
                        sub_j = 0;
                        sub_i++;
                    }
                }
            }
        }
        return cof;
    }

    std::vector<std::vector<double>> LinAlg::adjoint(std::vector<std::vector<double>> A){

        //Resizing the initial adjoint matrix
        std::vector<std::vector<double>> adj;
        adj.resize(A.size());
        for(int i = 0; i < adj.size(); i++){
            adj[i].resize(A.size());
        }

        // Checking for the case where the given N x N matrix is a scalar
        if(A.size() == 1){
            adj[0][0] = 1;
            return adj;
        }
        
        if(A.size() == 2){
            adj[0][0] = A[1][1];
            adj[1][1] = A[0][0];
            
            adj[0][1] = -A[0][1];
            adj[1][0] = -A[1][0];
            return adj;
        }

      for(int i = 0; i < A.size(); i++){
        for(int j = 0; j < A.size(); j++){
          std::vector<std::vector<double>> cof = cofactor(A, int(A.size()), i, j);
          // 1 if even, -1 if odd
          int sign = (i + j) % 2 == 0 ? 1 : -1;
          adj[j][i] = sign * det(cof, int(A.size()) - 1);
        }
      }
      return adj;
    }

    // The inverse can be computed as (1 / determinant(A)) * adjoint(A)
    std::vector<std::vector<double>> LinAlg::inverse(std::vector<std::vector<double>> A){
      return scalarMultiply(1/det(A, int(A.size())), adjoint(A));
    }
    
    // This is simply the Moore-Penrose least squares approximation of the inverse. 
    std::vector<std::vector<double>> LinAlg::pinverse(std::vector<std::vector<double>> A){
        return matmult(inverse(matmult(transpose(A), A)), transpose(A));
    }

    std::vector<std::vector<double>> LinAlg::zeromat(int n, int m){
        std::vector<std::vector<double>> zeromat;
        zeromat.resize(n);
        for(int i = 0; i < zeromat.size(); i++){
            zeromat[i].resize(m);
        }
        return zeromat; 
    }

    std::vector<std::vector<double>> LinAlg::onemat(int n, int m){
        return full(n, m, 1);
    }

    std::vector<std::vector<double>> LinAlg::full(int n, int m, int k){
        std::vector<std::vector<double>> full;
        full.resize(n);
        for(int i = 0; i < full.size(); i++){
            full[i].resize(m);
        }
        for(int i = 0; i < full.size(); i++){
            for(int j = 0; j < full[i].size(); j++){
                full[i][j] = k; 
            }
        }
        return full; 
    }

    std::vector<std::vector<double>> LinAlg::sin(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                B[i][j] = std::sin(A[i][j]);
            }
        }
        return B;
    }

    std::vector<std::vector<double>> LinAlg::cos(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                B[i][j] = std::cos(A[i][j]);
            }
        }
        return B;
    }

    std::vector<double> LinAlg::max(std::vector<double> a, std::vector<double> b){
        std::vector<double> c; 
        c.resize(a.size());
        for(int i = 0; i < c.size(); i++){
            if(a[i] >= b[i]) { 
                c[i] = a[i]; 
            }
            else { c[i] = b[i]; }
        }
        return c;
    }

    double LinAlg::max(std::vector<std::vector<double>> A){
        return max(flatten(A));
    }

    double LinAlg::min(std::vector<std::vector<double>> A){
        return min(flatten(A));
    }

    std::vector<std::vector<double>> LinAlg::round(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                B[i][j] = std::round(A[i][j]);
            }
        }
        return B;
    }

     double LinAlg::norm_2(std::vector<std::vector<double>> A){
         double sum = 0; 
         for(int i = 0; i < A.size(); i++){
             for(int j = 0; j < A[i].size(); j++){
                sum += A[i][j] * A[i][j];
             }
         }
         return std::sqrt(sum);
     }

    std::vector<std::vector<double>> LinAlg::identity(double d){
        std::vector<std::vector<double>> identityMat; 
        identityMat.resize(d);
        for(int i = 0; i < identityMat.size(); i++){
            identityMat[i].resize(d);
        }
        for(int i = 0; i < identityMat.size(); i++){
            for(int j = 0; j < identityMat.size(); j++){
                if(i == j){
                    identityMat[i][j] = 1;
                }
                else { identityMat[i][j] = 0; }
            }
        }
        return identityMat;
    }

    std::vector<std::vector<double>> LinAlg::cov(std::vector<std::vector<double>> A){
        Stat stat;
        std::vector<std::vector<double>> covMat;
        covMat.resize(A.size());
        for(int i = 0; i < covMat.size(); i++){
            covMat[i].resize(A.size());
        }
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A.size(); j++){
                covMat[i][j] = stat.covariance(A[i], A[j]);
            }
        }
        return covMat;
    }

    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> LinAlg::eig(std::vector<std::vector<double>> A){
        /*
        A (the entered parameter) in most use cases will be X'X, XX', etc. and must be symmetric.
        That simply means that 1) X' = X and 2) X is a square matrix. This function that computes the 
        eigenvalues of a matrix is utilizing Jacobi's method. 
        */

        double diagonal = true; // Perform the iterative Jacobi algorithm unless and until we reach a diagonal matrix which yields us the eigenvals. 
        
        std::map<int, int> val_to_vec; 
        std::vector<std::vector<double>> a_new;
        std::vector<std::vector<double>> eigenvectors = identity(A.size());
        do{
            double a_ij = A[0][1];
            double sub_i = 0; 
            double sub_j = 1;
            for(int i = 0; i < A.size(); i++){
                for(int j = 0; j < A[i].size(); j++){
                    if(i != j && std::abs(A[i][j]) > a_ij){
                        a_ij = A[i][j];
                        sub_i = i; 
                        sub_j = j;
                    }
                    else if(i != j && std::abs(A[i][j]) == a_ij){
                        if(i < sub_i){
                            a_ij = A[i][j];
                            sub_i = i; 
                            sub_j = j;
                        }
                    }
                }
            }

            double a_ii = A[sub_i][sub_i];
            double a_jj = A[sub_j][sub_j]; 
            double a_ji = A[sub_j][sub_i]; 
            double theta; 

            if(a_ii == a_jj) {
                theta = M_PI / 4; 
            }
            else{
                theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
            }

            std::vector<std::vector<double>> P = identity(A.size());
            P[sub_i][sub_j] = -std::sin(theta);
            P[sub_i][sub_i] = std::cos(theta);
            P[sub_j][sub_j] = std::cos(theta);
            P[sub_j][sub_i] = std::sin(theta);

            a_new = matmult(matmult(inverse(P), A), P);

            for(int i = 0; i < a_new.size(); i++){
                for(int j = 0; j < a_new[i].size(); j++){
                    if(i != j && std::round(a_new[i][j]) == 0){
                        a_new[i][j] = 0;
                    }
                }
            }

            bool non_zero = false;
            for(int i = 0; i < a_new.size(); i++){
                for(int j = 0; j < a_new[i].size(); j++){
                    if(i != j && std::round(a_new[i][j]) != 0){
                        non_zero = true;
                    }
                }
            }   

            if(non_zero) { 
                diagonal = false;
            }
            else{
                diagonal = true;
            }

            if(a_new == A){
                diagonal = true; 
                for(int i = 0; i < a_new.size(); i++){
                    for(int j = 0; j < a_new[i].size(); j++){
                        if(i != j){
                            a_new[i][j] = 0;
                        }
                    }
                }   
            }
            
            eigenvectors = matmult(eigenvectors, P);
            A = a_new;

        } while(!diagonal);

        std::vector<std::vector<double>> a_new_prior = a_new;
        
        // Bubble Sort. Should change this later.
        for(int i = 0; i < a_new.size() - 1; i++){
            for(int j = 0; j < a_new.size() - 1 - i; j++){
                if(a_new[j][j] < a_new[j + 1][j + 1]){
                    double temp = a_new[j + 1][j + 1];
                    a_new[j + 1][j + 1] = a_new[j][j];
                    a_new[j][j] = temp;
                }
            }
        }


        for(int i = 0; i < a_new.size(); i++){
            for(int j = 0; j < a_new.size(); j++){
                if(a_new[i][i] == a_new_prior[j][j]){
                    val_to_vec[i] = j;
                }
            }
        }

        std::vector<std::vector<double>> eigen_temp = eigenvectors;
        for(int i = 0; i < eigenvectors.size(); i++){
            for(int j = 0; j < eigenvectors[i].size(); j++){
                eigenvectors[i][j] = eigen_temp[i][val_to_vec[j]];  
            }
        }
        return {eigenvectors, a_new};

    }

    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> LinAlg::SVD(std::vector<std::vector<double>> A){
        auto [left_eigenvecs, eigenvals] = eig(matmult(A, transpose(A)));
        auto [right_eigenvecs, right_eigenvals] = eig(matmult(transpose(A), A));

        std::vector<std::vector<double>> singularvals = sqrt(eigenvals);
        std::vector<std::vector<double>> sigma = zeromat(A.size(), A[0].size());
         for(int i = 0; i < singularvals.size(); i++){
            for(int j = 0; j < singularvals[i].size(); j++){
                sigma[i][j] = singularvals[i][j];
            }
        }
        return {left_eigenvecs, sigma, right_eigenvecs};
    }

    std::vector<double> LinAlg::vectorProjection(std::vector<double> a, std::vector<double> b){
        double product = dot(a, b)/dot(a, a);
        return scalarMultiply(product, a); // Projection of vector a onto b. Denotated as proj_a(b).
    }

    std::vector<std::vector<double>> LinAlg::gramSchmidtProcess(std::vector<std::vector<double>> A){
        A = transpose(A); // C++ vectors lack a mechanism to directly index columns. So, we transpose *a copy* of A for this purpose for ease of use.
        std::vector<std::vector<double>> B;
        B.resize(A.size());
        for(int i = 0; i < B.size(); i++){
            B[i].resize(A[0].size());
        }

        B[0] = A[0]; // We set a_1 = b_1 as an initial condition.
        B[0] = scalarMultiply(1/norm_2(B[0]), B[0]);
        for(int i = 1; i < B.size(); i++){
            B[i] = A[i];
            for(int j = i-1; j >= 0; j--){
                B[i] = subtraction(B[i], vectorProjection(B[j], A[i]));
            }
            B[i] = scalarMultiply(1/norm_2(B[i]), B[i]); // Very simply multiply all elements of vec B[i] by 1/||B[i]||_2
        }
        return transpose(B); // We re-transpose the marix. 
    }

    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> LinAlg::QRD(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> Q = gramSchmidtProcess(A);
        std::vector<std::vector<double>> R = matmult(transpose(Q), A);
        return {Q, R};

    }
    
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> LinAlg::chol(std::vector<std::vector<double>> A){
        std::vector<std::vector<double>> L = zeromat(A.size(), A[0].size());
        for(int j = 0; j < L.size(); j++){ // Matrices entered must be square. No problem here.
            for(int i = j; i < L.size(); i++){
                if(i == j){
                    double sum = 0;
                    for(int k = 0; k < j; k++){ 
                        sum += L[i][k] * L[i][k];
                    }
                    L[i][j] = std::sqrt(A[i][j] - sum);
                }
                else{ // That is, i!=j
                    double sum = 0;
                    for(int k = 0; k < j; k++){ 
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (A[i][j] - sum)/L[j][j];
                }
            }
        }
        return {L, transpose(L)}; // Indeed, L.T is our upper triangular matrix. 
    }

    double LinAlg::sum_elements(std::vector<std::vector<double>> A){
        double sum = 0;
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                sum += A[i][j];
            }
        }
        return sum;
    }

    std::vector<double> LinAlg::flatten(std::vector<std::vector<double>> A){
        std::vector<double> a; 
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                a.push_back(A[i][j]);
            }
        }
        return a;
    }

    std::vector<double> LinAlg::solve(std::vector<std::vector<double>> A, std::vector<double> b){
        return mat_vec_mult(inverse(A), b);
    }

    bool LinAlg::positiveDefiniteChecker(std::vector<std::vector<double>> A){
        auto [eigenvectors, eigenvals] = eig(A);
        std::vector<double> eigenvals_vec;
        for(int i = 0; i < eigenvals.size(); i++){
            eigenvals_vec.push_back(eigenvals[i][i]);
        }
        for(int i = 0; i < eigenvals_vec.size(); i++){
            if(eigenvals_vec[i] <= 0){ // Simply check to ensure all eigenvalues are positive.
                return false;
            }
        }
        return true;
    }

    bool LinAlg::negativeDefiniteChecker(std::vector<std::vector<double>> A){
        auto [eigenvectors, eigenvals] = eig(A);
        std::vector<double> eigenvals_vec;
        for(int i = 0; i < eigenvals.size(); i++){
            eigenvals_vec.push_back(eigenvals[i][i]);
        }
        for(int i = 0; i < eigenvals_vec.size(); i++){
            if(eigenvals_vec[i] >= 0){ // Simply check to ensure all eigenvalues are negative.
                return false;
            }
        }
        return true;
    }

    bool LinAlg::zeroEigenvalue(std::vector<std::vector<double>> A){
        auto [eigenvectors, eigenvals] = eig(A);
        std::vector<double> eigenvals_vec;
        for(int i = 0; i < eigenvals.size(); i++){
            eigenvals_vec.push_back(eigenvals[i][i]);
        }
        for(int i = 0; i < eigenvals_vec.size(); i++){
            if(eigenvals_vec[i] == 0){ 
                return true;
            }
        }
        return false;
    }

    void LinAlg::printMatrix(std::vector<std::vector<double>> A){
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                std::cout << A[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<std::vector<double>> LinAlg::outerProduct(std::vector<double> a, std::vector<double> b){
        std::vector<std::vector<double>> C;
        C.resize(a.size());
        for(int i = 0; i < C.size(); i++){
            C[i] = scalarMultiply(a[i], b);
        }
        return C;
    }

    std::vector<double> LinAlg::hadamard_product(std::vector<double> a, std::vector<double> b){
        std::vector<double> c;
        c.resize(a.size());
        
        for(int i = 0; i < a.size(); i++){
            c[i] = a[i] * b[i];
        }
        
        return c;
    }

    std::vector<double> LinAlg::elementWiseDivision(std::vector<double> a, std::vector<double> b){
        std::vector<double> c;
        c.resize(a.size());

        for(int i = 0; i < a.size(); i++){
            c[i] = a[i] / b[i];
        }
        return c;
    }

    std::vector<double> LinAlg::scalarMultiply(double scalar, std::vector<double> a){
        for(int i = 0; i < a.size(); i++){
            a[i] *= scalar;
        }
        return a;
    }

    std::vector<double> LinAlg::scalarAdd(double scalar, std::vector<double> a){
        for(int i = 0; i < a.size(); i++){
            a[i] += scalar;
        }
        return a;
    }

    std::vector<double> LinAlg::addition(std::vector<double> a, std::vector<double> b){
        std::vector<double> c;
        c.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            c[i] = a[i] + b[i];
        }
        return c;
    }

    std::vector<double> LinAlg::subtraction(std::vector<double> a, std::vector<double> b){
        std::vector<double> c;
        c.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            c[i] = a[i] - b[i];
        }
        return c;
    }

    std::vector<double> LinAlg::subtractMatrixRows(std::vector<double> a, std::vector<std::vector<double>> B){
        for(int i = 0; i < B.size(); i++){
            a = subtraction(a, B[i]);
        }
        return a; 
    }

    std::vector<double> LinAlg::log(std::vector<double> a){
        std::vector<double> b; 
        b.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            b[i] = std::log(a[i]);
        }
        return b; 
    }

    std::vector<double> LinAlg::log10(std::vector<double> a){
        std::vector<double> b; 
        b.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            b[i] = std::log10(a[i]);
        }
        return b; 
    }

    std::vector<double> LinAlg::exp(std::vector<double> a){
        std::vector<double> b;
        b.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            b[i] = std::exp(a[i]);
        }
        return b;
    }

    std::vector<double> LinAlg::erf(std::vector<double> a){
        std::vector<double> b;
        b.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            b[i] = std::erf(a[i]);
        }
        return b;
    }

    std::vector<double> LinAlg::exponentiate(std::vector<double> a, double p){
        std::vector<double> b; 
        b.resize(a.size());
        for(int i = 0; i < b.size(); i++){
            b[i] = std::pow(a[i], p); 
        }
        return b;
    }

    std::vector<double> LinAlg::sqrt(std::vector<double> a){
        return exponentiate(a, 0.5);
    }

    std::vector<double> LinAlg::cbrt(std::vector<double> a){
        return exponentiate(a, double(1)/double(3));
    } 

    double LinAlg::dot(std::vector<double> a, std::vector<double> b){
        double c = 0;
        for(int i = 0; i < a.size(); i++){
            c += a[i] * b[i];
        }
        return c;
    }

    std::vector<double> LinAlg::cross(std::vector<double> a, std::vector<double> b){
        // Cross products exist in R^7 also. Though, I will limit it to R^3 as Wolfram does this. 
        std::vector<std::vector<double>> mat = {onevec(3), a, b};
        
        double det1 = det({{a[1], a[2]}, {b[1], b[2]}}, 2);
        double det2 = -det({{a[0], a[2]}, {b[0], b[2]}}, 2);
        double det3 = det({{a[0], a[1]}, {b[0], b[1]}}, 2);

        return {det1, det2, det3};
    }

    std::vector<double> LinAlg::abs(std::vector<double> a){
        std::vector<double> b; 
        b.resize(a.size());
        for(int i = 0; i < b.size(); i++){
            b[i] = std::abs(a[i]);
        }
        return b;
    }

    std::vector<double> LinAlg::zerovec(int n){
        std::vector<double> zerovec; 
        zerovec.resize(n);
        return zerovec;
    }

    std::vector<double> LinAlg::onevec(int n){
        return full(n, 1);
    }

    std::vector<std::vector<double>> LinAlg::diag(std::vector<double> a){
        std::vector<std::vector<double>> B = zeromat(a.size(), a.size());
        for(int i = 0; i < B.size(); i++){
            B[i][i] = a[i];
        }
        return B;
    }

    std::vector<double> LinAlg::full(int n, int k){
        std::vector<double> full; 
        full.resize(n);
        for(int i = 0; i < full.size(); i++){
            full[i] = k;
        }
        return full;
    }

    std::vector<double> LinAlg::sin(std::vector<double> a){
        std::vector<double> b; 
        b.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            b[i] = std::sin(a[i]);
        }
        return b; 
    }

    std::vector<double> LinAlg::cos(std::vector<double> a){
        std::vector<double> b; 
        b.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            b[i] = std::cos(a[i]);
        }
        return b; 
    }

    std::vector<std::vector<double>> LinAlg::rotate(std::vector<std::vector<double>> A, double theta, int axis){
        std::vector<std::vector<double>> rotationMatrix = {{std::cos(theta), -std::sin(theta)}, {std::sin(theta), std::cos(theta)}};
        if(axis == 0) {rotationMatrix = {{1, 0, 0}, {0, std::cos(theta), -std::sin(theta)}, {0, std::sin(theta), std::cos(theta)}};}
        else if(axis == 1) {rotationMatrix = {{std::cos(theta), 0, std::sin(theta)}, {0, 1, 0}, {-std::sin(theta), 0, std::cos(theta)}};}
        else if (axis == 2) {rotationMatrix = {{std::cos(theta), -std::sin(theta), 0}, {std::sin(theta), std::cos(theta), 0}, {1, 0, 0}};}

        return matmult(A, rotationMatrix);
    }

    std::vector<std::vector<double>> LinAlg::max(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B){
        std::vector<std::vector<double>> C;
        C.resize(A.size());
        for(int i = 0; i < C.size(); i++){
            C[i].resize(A[0].size());
        }
        for(int i = 0; i < A.size(); i++){
            C[i] = max(A[i], B[i]);
        }
        return C;
    }

    double LinAlg::max(std::vector<double> a){
        int max = a[0];
        for(int i = 0; i < a.size(); i++){
            if(a[i] > max){
                max = a[i];
            }
        }
        return max; 
    }

    double LinAlg::min(std::vector<double> a){
        int min = a[0];
        for(int i = 0; i < a.size(); i++){
            if(a[i] < min){
                min = a[i];
            }
        }
        return min; 
    }

    std::vector<double> LinAlg::round(std::vector<double> a){
        std::vector<double> b;
        b.resize(a.size());
        for(int i = 0; i < a.size(); i++){
            b[i] = std::round(a[i]);
        }
        return b;
    }

    // Multidimensional Euclidean Distance
    double LinAlg::euclideanDistance(std::vector<double> a, std::vector<double> b){
        double dist = 0;
        for(int i = 0; i < a.size(); i++){
            dist += (a[i] - b[i])*(a[i] - b[i]);
        }
        return std::sqrt(dist);
    }

    double LinAlg::norm_2(std::vector<double> a){
        return std::sqrt(norm_sq(a));
    }

    double LinAlg::norm_sq(std::vector<double> a){
        double n_sq = 0;
        for(int i = 0; i < a.size(); i++){
            n_sq += a[i] * a[i];
        }
        return n_sq;
    }

    double LinAlg::sum_elements(std::vector<double> a){
        double sum = 0;
        for(int i = 0; i < a.size(); i++){
            sum += a[i];
        }
        return sum;
    }

    double LinAlg::cosineSimilarity(std::vector<double> a, std::vector<double> b){
        return dot(a, b) / (norm_2(a) * norm_2(b));
    }

    void LinAlg::printVector(std::vector<double> a){
        for(int i = 0; i < a.size(); i++){
            std::cout << a[i] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<std::vector<double>> LinAlg::mat_vec_add(std::vector<std::vector<double>> A, std::vector<double> b){
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                A[i][j] += b[j];
            }
        }
        return A;
    }

    std::vector<double> LinAlg::mat_vec_mult(std::vector<std::vector<double>> A, std::vector<double> b){
        std::vector<double> c;
        c.resize(A.size());
            
        for(int i = 0; i < A.size(); i++){
            for(int k = 0; k < b.size(); k++){
                c[i] += A[i][k] * b[k];
            }
        }
        return c;
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::addition(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B){
        for(int i = 0; i < A.size(); i++){
            A[i] = addition(A[i], B[i]);
        }
        return A;
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::elementWiseDivision(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B){
        for(int i = 0; i < A.size(); i++){
            A[i] = elementWiseDivision(A[i], B[i]);
        }
        return A;
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::sqrt(std::vector<std::vector<std::vector<double>>> A){
        for(int i = 0; i < A.size(); i++){
            A[i] = sqrt(A[i]);
        }
        return A;
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::exponentiate(std::vector<std::vector<std::vector<double>>> A, double p){
        for(int i = 0; i < A.size(); i++){
            A[i] = exponentiate(A[i], p);
        }
        return A;
    }

    std::vector<std::vector<double>> LinAlg::tensor_vec_mult(std::vector<std::vector<std::vector<double>>> A, std::vector<double> b){
        std::vector<std::vector<double>> C;
        C.resize(A.size());
        for(int i = 0; i < C.size(); i++){
            C[i].resize(A[0].size());
        }
        for(int i = 0; i < C.size(); i++){
            for(int j = 0; j < C[i].size(); j++){
                C[i][j] = dot(A[i][j], b);
            }
        }
        return C;
    }

    std::vector<double> LinAlg::flatten(std::vector<std::vector<std::vector<double>>> A){
        std::vector<double> c;
        for(int i = 0; i < A.size(); i++){
            std::vector<double> flattenedVec = flatten(A[i]);
            c.insert(c.end(), flattenedVec.begin(), flattenedVec.end());
        }
        return c;
    }

    void LinAlg::printTensor(std::vector<std::vector<std::vector<double>>> A){
        for(int i = 0; i < A.size(); i++){
            printMatrix(A[i]);
            if(i != A.size() - 1) { std::cout << std::endl; }
        }
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::scalarMultiply(double scalar, std::vector<std::vector<std::vector<double>>> A){
        for(int i = 0; i < A.size(); i++){
            A[i] = scalarMultiply(scalar, A[i]);
        }
        return A; 
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::scalarAdd(double scalar, std::vector<std::vector<std::vector<double>>> A){
        for(int i = 0; i < A.size(); i++){
            A[i] = scalarAdd(scalar, A[i]);
        }
        return A;
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::resize(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B){
        A.resize(B.size());
        for(int i = 0; i < B.size(); i++){
            A[i].resize(B[i].size());
            for(int j = 0; j < B[i].size(); j++){
                A[i][j].resize(B[i][j].size());
            }
        }
        return A; 
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::max(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<std::vector<double>>> B){
        for(int i = 0; i < A.size(); i++){
            A[i] = max(A[i], B[i]);
        }
        return A;
    }

    std::vector<std::vector<std::vector<double>>> LinAlg::abs(std::vector<std::vector<std::vector<double>>> A){
        for(int i = 0; i < A.size(); i++){
            A[i] = abs(A[i]);
        }
        return A;
    }

    double LinAlg::norm_2(std::vector<std::vector<std::vector<double>>> A){
        double sum = 0; 
        for(int i = 0; i < A.size(); i++){
            for(int j = 0; j < A[i].size(); j++){
                for(int k = 0; k < A[i][j].size(); k++){
                    sum += A[i][j][k] * A[i][j][k];
                }
            }
        }
        return std::sqrt(sum);
    }

    // Bad implementation. Change this later. 
    std::vector<std::vector<std::vector<double>>> LinAlg::vector_wise_tensor_product(std::vector<std::vector<std::vector<double>>> A, std::vector<std::vector<double>> B){
        std::vector<std::vector<std::vector<double>>> C; 
        C = resize(C, A);
        for(int i = 0; i < A[0].size(); i++){
            for(int j = 0; j < A[0][i].size(); j++){
                std::vector<double> currentVector;
                currentVector.resize(A.size());

                for(int k = 0; k < C.size(); k++){
                    currentVector[k] = A[k][i][j];
                }

                currentVector = mat_vec_mult(B, currentVector);

                for(int k = 0; k < C.size(); k++){
                    C[k][i][j] = currentVector[k];
                }
            }
        }
        return C;
    }
}