# ML++

Machine learning is a vast and exiciting discipline, garnering attention from specialists of many fields. Unfortunately, for C++ programmers and enthusiasts, there appears to be a lack of support in the field of machine learning. To fill that void and give C++ a true foothold in the ML sphere, this library was written. The intent with this library is for it to act as a crossroad between low-level developers and machine learning engineers.

<p align="center">
    <img src="https://user-images.githubusercontent.com/78002988/119920911-f3338d00-bf21-11eb-89b3-c84bf7c9f4ac.gif" 
    width = 600 height = 400>
</p>

## Installation
Begin by downloading the header files for the ML++ library. You can do this by cloning the repository and extracting the MLPP directory within it:
```
git clone https://github.com/novak-99/MLPP
```
Next, execute the "buildSO.sh" shell script:
```
sudo ./buildSO.sh
```
After doing so, maintain the ML++ source files in a local directory and include them in this fashion: 
```cpp
#include "MLPP/Stat/Stat.hpp" // Including the ML++ statistics module. 

int main(){
...
}
```
Finally, after you have concluded creating a project, compile it using g++:
```
g++ main.cpp /usr/local/lib/MLPP.so --std=c++17
```

## Usage
Please note that ML++ uses the ```std::vector<double>``` data type for emulating vectors, and the ```std::vector<std::vector<double>>``` data type for emulating matrices.

Begin by including the respective header file of your choice.
```cpp
#include "MLPP/LinReg/LinReg.hpp"
```
Next, instantiate an object of the class. Don't forget to pass the input set and output set as parameters.
```cpp
LinReg model(inputSet, outputSet);
```
Afterwards, call the optimizer that you would like to use. For iterative optimizers such as gradient descent, include the learning rate, epoch number, and whether or not to utilize the UI panel. 
```cpp
model.gradientDescent(0.001, 1000, 0);
```
Great, you are now ready to test! To test a singular testing instance, utilize the following function:
```cpp
model.modelTest(testSetInstance);
```
This will return the model's singular prediction for that example. 

To test an entire test set, use the following function: 
```cpp
model.modelSetTest(testSet);
```
The result will be the model's predictions for the entire dataset.


## Contents of the Library
1. ***Regression***
    1. Linear Regression 
    2. Logistic Regression
    3. Softmax Regression
    4. Exponential Regression
    5. Probit Regression
    6. CLogLog Regression
    7. Tanh Regression
2. ***Deep, Dynamically Sized Neural Networks***
    1. Possible Activation Functions
        - Linear
        - Sigmoid
        - Softmax
        - Swish
        - Mish
        - SinC
        - Softplus
        - Softsign
        - CLogLog
        - Logit
        - Gaussian CDF
        - RELU
        - GELU
        - Sign
        - Unit Step 
        - Sinh
        - Cosh
        - Tanh
        - Csch
        - Sech
        - Coth
        - Arsinh
        - Arcosh
        - Artanh
        - Arcsch
        - Arsech
        - Arcoth
    2. Possible Optimization Algorithms
        - Batch Gradient Descent
        - Mini-Batch Gradient Descent 
        - Stochastic Gradient Descent 
        - Gradient Descent with Momentum
        - Nesterov Accelerated Gradient
        - Adagrad Optimizer 
        - Adadelta Optimizer 
        - Adam Optimizer 
        - Adamax Optimizer 
        - Nadam Optimizer 
        - AMSGrad Optimizer 
        - 2nd Order Newton-Raphson Optimizer*
        - Normal Equation*
        <p></p>
        *Only available for linear regression
    3. Possible Loss Functions
        - MSE
        - RMSE 
        - MAE
        - MBE
        - Log Loss
        - Cross Entropy
        - Hinge Loss
        - Wasserstein Loss
    4. Possible Regularization Methods
        - Lasso
        - Ridge
        - ElasticNet
        - Weight Clipping
    5. Possible Weight Initialization Methods
        - Uniform 
        - Xavier Normal
        - Xavier Uniform
        - He Normal
        - He Uniform
        - LeCun Normal
        - LeCun Uniform
    6. Possible Learning Rate Schedulers
        - Time Based 
        - Epoch Based
        - Step Based
        - Exponential 
3. ***Prebuilt Neural Networks***
    1. Multilayer Peceptron
    2. Autoencoder
    3. Softmax Network
4. ***Generative Modeling***
    1. Tabular Generative Adversarial Networks
    2. Tabular Wasserstein Generative Adversarial Networks
5. ***Natural Language Processing***
    1. Word2Vec (Continous Bag of Words, Skip-Gram)
    2. Stemming
    3. Bag of Words
    4. TFIDF
    5. Tokenization 
    6. Auxiliary Text Processing Functions
6. ***Computer Vision***
    1. The Convolution Operation
    2. Max, Min, Average Pooling
    3. Global Max, Min, Average Pooling
    4. Prebuilt Feature Detectors
        - Horizontal/Vertical Prewitt Filter
        - Horizontal/Vertical Sobel Filter
        - Horizontal/Vertical Scharr Filter
        - Horizontal/Vertical Roberts Filter
        - Gaussian Filter
        - Harris Corner Detector
7. ***Principal Component Analysis***
8. ***Naive Bayes Classifiers***
    1. Multinomial Naive Bayes
    2. Bernoulli Naive Bayes 
    3. Gaussian Naive Bayes
9. ***Support Vector Classification***
    1. Primal Formulation (Hinge Loss Objective) 
    2. Dual Formulation (Via Lagrangian Multipliers)
10. ***K-Means***
11. ***k-Nearest Neighbors***
12. ***Outlier Finder (Using z-scores)***
13. ***Matrix Decompositions***    
    1. SVD Decomposition
    2. Cholesky Decomposition
        - Positive Definiteness Checker 
    3. QR Decomposition
14. ***Numerical Analysis***
    1. Numerical Diffrentiation 
        - Univariate Functions 
        - Multivariate Functions 
    2. Jacobian Vector Calculator
    3. Hessian Matrix Calculator
    4. Function approximator
        - Constant Approximation
        - Linear Approximation 
        - Quadratic Approximation
        - Cubic Approximation
    5. Diffrential Equations Solvers 
        - Euler's Method 
        - Growth Method
15. ***Mathematical Transforms***
    1. Discrete Cosine Transform
16. ***Linear Algebra Module***
17. ***Statistics Module***
18. ***Data Processing Module***
    1. Setting and Printing Datasets 
    2. Available Datasets
        1. Wisconsin Breast Cancer Dataset
            - Binary
            - SVM 
        2. MNIST Dataset
            - Train
            - Test
        3. Iris Flower Dataset
        4. Wine Dataset
        5. California Housing Dataset
        6. Fires and Crime Dataset (Chicago)
    3. Feature Scaling 
    4. Mean Normalization
    5. One Hot Representation
    6. Reverse One Hot Representation
    7. Supported Color Space Conversions 
        - RGB to Grayscale
        - RGB to HSV
        - RGB to YCbCr
        - RGB to XYZ
        - XYZ to RGB
19. ***Utilities***
    1. TP, FP, TN, FN function
    2. Precision
    3. Recall 
    4. Accuracy
    5. F1 score


## What's in the Works? 
ML++, like most frameworks, is dynamic, and constantly changing. This is especially important in the world of ML, as new algorithms and techniques are being developed day by day. Here are a couple of things currently being developed for ML++:
    <p>
    - Convolutional Neural Networks 
    </p>
    <p>
    - Kernels for SVMs 
    </p>
    <p>
    - Support Vector Regression
    </p>    
    
## Citations
Various different materials helped me along the way of creating ML++, and I would like to give credit to several of them here. [This](https://www.tutorialspoint.com/cplusplus-program-to-compute-determinant-of-a-matrix) article by TutorialsPoint was a big help when trying to implement the determinant of a matrix, and [this](https://www.geeksforgeeks.org/adjoint-inverse-matrix/) article by GeeksForGeeks was very helpful when trying to take the adjoint and inverse of a matrix.
