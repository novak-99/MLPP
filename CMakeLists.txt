cmake_minimum_required(VERSION 3.12)

project(MLPP-Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(MLPP)

add_library(MLPP
  MLPP/Stat/Stat.cpp
  MLPP/LinAlg/LinAlg.cpp
  MLPP/Regularization/Reg.cpp
  MLPP/Activation/Activation.cpp
  MLPP/Utilities/Utilities.cpp
  MLPP/Data/Data.cpp
  MLPP/Cost/Cost.cpp
  MLPP/ANN/ANN.cpp
  MLPP/HiddenLayer/HiddenLayer.cpp
  MLPP/OutputLayer/OutputLayer.cpp
  MLPP/MLP/MLP.cpp
  MLPP/LinReg/LinReg.cpp
  MLPP/LogReg/LogReg.cpp
  MLPP/UniLinReg/UniLinReg.cpp
  MLPP/CLogLogReg/CLogLogReg.cpp
  MLPP/ExpReg/ExpReg.cpp
  MLPP/ProbitReg/ProbitReg.cpp
  MLPP/SoftmaxReg/SoftmaxReg.cpp
  MLPP/TanhReg/TanhReg.cpp
  MLPP/SoftmaxNet/SoftmaxNet.cpp
  MLPP/Convolutions/Convolutions.cpp
  MLPP/AutoEncoder/AutoEncoder.cpp
  MLPP/MultinomialNB/MultinomialNB.cpp
  MLPP/BernoulliNB/BernoulliNB.cpp
  MLPP/GaussianNB/GaussianNB.cpp
  MLPP/KMeans/KMeans.cpp
  MLPP/kNN/kNN.cpp
  MLPP/PCA/PCA.cpp
  MLPP/OutlierFinder/OutlierFinder.cpp
  MLPP/MANN/MANN.cpp
  MLPP/MultiOutputLayer/MultiOutputLayer.cpp
  MLPP/SVC/SVC.cpp
  MLPP/NumericalAnalysis/NumericalAnalysis.cpp
  MLPP/DualSVC/DualSVC.cpp
  MLPP/Transforms/Transforms.cpp
  )

add_executable(mlpp-app
  main.cpp
  )

target_link_libraries(mlpp-app
  MLPP
  )
