# ML++

Machine learning is a vast and exiciting discipline, garnering attention from specialists of many fields. Unfortunately, for C++ programmers and enthusiasts, there appears to be a lack of support for this magnificient language in the field of machine learning. As a consequence, this library was created in order to fill that void and give C++ a true foothold in the ML sphere to act as a crossroad between low level developers and machine learning engineers. 

<p align="center">
    <img src="https://user-images.githubusercontent.com/78002988/119920911-f3338d00-bf21-11eb-89b3-c84bf7c9f4ac.gif" 
    width = 600 height = 400>
</p>

## Usage
Please note that ML++ uses the ```std::vector<double>``` data type for emulating vectors, and the ```std::vector<std::vector<double>>``` data type for emulating matricies.
<p> Begin by including the respective header file of your choice. </p>
```cpp
#include "MLPP/LinReg/LinReg.hpp"
```
Next, instantiate an object of the class. Don't forget to pass the input set, output set as parameters.
```cpp
LinReg model(inputSet, outputSet);
```
Next, call the optimizer that you would like to use. For iterative optimizers such as gradient descent, include the learning rate, epoch number, and whether or not to utilize the UI pannel. 
```cpp
model.gradientDescent(0.001, 1000, 0);
```
Great, you are now ready to test! To test a singular testing instance, utilize the following function:
```cpp
model.modelTest(testSetInstance);
```
This will return the model's singular prediction for that example. 
To test an entire dataset of instances, use the following function: 
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
        - Swish
        - Softplus
        - CLogLog
        - Gaussian CDF
        - GELU
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
    2. Possible Loss Functions
        - MSE
        - RMSE 
        - MAE
        - MBE
        - Log Loss
        - Cross Entropy
        - Hinge Loss
    3. Possible Regularization Methods
        - Lasso
        - Ridge
        - ElasticNet
    4. Possible Weight Initialization Methods
        - Uniform 
        - Xavier Normal
        - Xavier Uniform
        - He Normal
        - He Uniform
3. ***Prebuilt Neural Networks***
    1. Multilayer Peceptron
    2. Autoencoder
    3. Softmax Network
4. ***Natural Language Processing***
    1. Word2Vec (Continous Bag of Words, Skip-N Gram)
    2. Stemming
    3. Bag of Words
    4. TFIDF
    5. Tokenization 
    6. Auxiliary Text Processing Functions
5. ***Computer Vision***
    1. The Convolution Operation
    2. Max, Min, Average Pooling
    3. Global Max, Min, Average Pooling
    4. Prebuilt Feature Detectors
        - Horizontal/Vertical Prewitt Filter
        - Horizontal/Vertical Sobel Filter
        - Horizontal/Vertical Scharr Filter
        - Horizontal/Vertical Roberts Filter
6. ***Principal Component Analysis***
7. ***Naive Bayes Classifiers***
    1. Multinomial Naive Bayes
    2. Bernoulli Naive Bayes 
    3. Gaussian Naive Bayes
8. ***K-Means***
9. ***k-Nearest Neighbors***
10. ***Outlier Finder (Using z-scores)***
11. ***Linear Algebra Module***
12. ***Statistics Module***
13. ***Data Processing Module***
    1. Setting and Printing Datasets 
    2. Feature Scaling 
    3. Mean Normalization
    4. One Hot Representation 
    5. Reverse One Hot Representation
