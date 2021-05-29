g++ -I MLPP -c MLPP/Stat/Stat.cpp MLPP/LinAlg/LinAlg.cpp MLPP/Regularization/Reg.cpp MLPP/Activation/Activation.cpp MLPP/Utilities/Utilities.cpp MLPP/Data/Data.cpp MLPP/Cost/Cost.cpp MLPP/ANN/ANN.cpp MLPP/HiddenLayer/HiddenLayer.cpp MLPP/OutputLayer/OutputLayer.cpp MLPP/MLP/MLP.cpp MLPP/LinReg/LinReg.cpp MLPP/LogReg/LogReg.cpp MLPP/UniLinReg/UniLinReg.cpp MLPP/CLogLogReg/CLogLogReg.cpp MLPP/ExpReg/ExpReg.cpp MLPP/ProbitReg/ProbitReg.cpp MLPP/SoftmaxReg/SoftmaxReg.cpp MLPP/TanhReg/TanhReg.cpp MLPP/SoftmaxNet/SoftmaxNet.cpp MLPP/Convolutions/Convolutions.cpp MLPP/AutoEncoder/AutoEncoder.cpp MLPP/MultinomialNB/MultinomialNB.cpp MLPP/BernoulliNB/BernoulliNB.cpp MLPP/GaussianNB/GaussianNB.cpp MLPP/KMeans/KMeans.cpp MLPP/kNN/kNN.cpp MLPP/PCA/PCA.cpp MLPP/OutlierFinder/OutlierFinder.cpp --std=c++17 -pthread

g++ -shared -o MLPP.so Reg.o LinAlg.o Stat.o Activation.o LinReg.o Utilities.o Cost.o LogReg.o ProbitReg.o ExpReg.o CLogLogReg.o SoftmaxReg.o TanhReg.o kNN.o KMeans.o UniLinReg.o SoftmaxNet.o MLP.o AutoEncoder.o HiddenLayer.o OutputLayer.o ANN.o BernoulliNB.o GaussianNB.o MultinomialNB.o Convolutions.o OutlierFinder.o Data.o

mv MLPP.so SharedLib

rm *.o