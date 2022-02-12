//
//  ANN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "ANN.hpp"
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Regularization/Reg.hpp"
#include "Utilities/Utilities.hpp"
#include "Cost/Cost.hpp"

#include <iostream>
#include <cmath>
#include <random>

namespace MLPP {
    ANN::ANN(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet)
    : inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), lrScheduler("None"), decayConstant(0), dropRate(0)
    {

    }

    ANN::~ANN(){
        delete outputLayer;
    }

    std::vector<double> ANN::modelSetTest(std::vector<std::vector<double>> X){
        if(!network.empty()){
            network[0].input = X;
            network[0].forwardPass();

            for(int i = 1; i < network.size(); i++){
                network[i].input = network[i - 1].a;
                network[i].forwardPass();
            }
            outputLayer->input = network[network.size() - 1].a;
        }
        else{
            outputLayer->input = X;
        }
        outputLayer->forwardPass();
        return outputLayer->a;
    }

    double ANN::modelTest(std::vector<double> x){
        if(!network.empty()){
            network[0].Test(x);
            for(int i = 1; i < network.size(); i++){
                network[i].Test(network[i - 1].a_test);
            }
            outputLayer->Test(network[network.size() - 1].a_test);
        }
        else{
            outputLayer->Test(x);
        }
        return outputLayer->a_test;
    }

    void ANN::gradientDescent(double learning_rate, int max_epoch, bool UI){
        class Cost cost; 
        LinAlg alg;
        double cost_prev = 0;
        int epoch = 1;
        forwardPass();
        double initial_learning_rate = learning_rate;

        alg.printMatrix(network[network.size() - 1].weights);
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            cost_prev = Cost(y_hat, outputSet);

            auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputSet);

            cumulativeHiddenLayerWGrad = alg.scalarMultiply(learning_rate/n, cumulativeHiddenLayerWGrad);
            outputWGrad = alg.scalarMultiply(learning_rate/n, outputWGrad);
            updateParameters(cumulativeHiddenLayerWGrad, outputWGrad, learning_rate); // subject to change. may want bias to have this matrix too.

            std::cout << learning_rate << std::endl;

            forwardPass();

            if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputSet); }

            epoch++;
            if(epoch > max_epoch) { break; }
        }
    }

    void ANN::SGD(double learning_rate, int max_epoch, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);

            std::random_device rd;
            std::default_random_engine generator(rd()); 
            std::uniform_int_distribution<int> distribution(0, int(n - 1));
            int outputIndex = distribution(generator);

            std::vector<double> y_hat = modelSetTest({inputSet[outputIndex]});
            cost_prev = Cost({y_hat}, {outputSet[outputIndex]});

            auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat,  {outputSet[outputIndex]});
            cumulativeHiddenLayerWGrad = alg.scalarMultiply(learning_rate/n, cumulativeHiddenLayerWGrad);
            outputWGrad = alg.scalarMultiply(learning_rate/n, outputWGrad);

            updateParameters(cumulativeHiddenLayerWGrad, outputWGrad, learning_rate); // subject to change. may want bias to have this matrix too.
            y_hat = modelSetTest({inputSet[outputIndex]});

            if(UI) { ANN::UI(epoch, cost_prev, y_hat, {outputSet[outputIndex]}); }
            
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void ANN::MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        // always evaluate the result 
        // always do forward pass only ONCE at end.
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = modelSetTest(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputMiniBatches[i]);
                cumulativeHiddenLayerWGrad = alg.scalarMultiply(learning_rate/n, cumulativeHiddenLayerWGrad);
                outputWGrad = alg.scalarMultiply(learning_rate/n, outputWGrad);

                updateParameters(cumulativeHiddenLayerWGrad, outputWGrad, learning_rate); // subject to change. may want bias to have this matrix too.
                y_hat = modelSetTest(inputMiniBatches[i]);

                if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]); }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void ANN::Momentum(double learning_rate, int max_epoch, int mini_batch_size, double gamma, bool NAG, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        // always evaluate the result 
        // always do forward pass only ONCE at end.
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        
        // Initializing necessary components for Adam. 
        std::vector<std::vector<std::vector<double>>> v_hidden;
        
        std::vector<double> v_output;
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = modelSetTest(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);
                
                auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputMiniBatches[i]);

                if(!network.empty() && v_hidden.empty()){ // Initing our tensor
                    v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
                }

                if(v_output.empty()){
                    v_output.resize(outputWGrad.size());
                }

                if(NAG){ // "Aposterori" calculation
                    updateParameters(v_hidden, v_output, 0); // DON'T update bias.
                }

                v_hidden = alg.addition(alg.scalarMultiply(gamma, v_hidden), alg.scalarMultiply(learning_rate/n, cumulativeHiddenLayerWGrad));

                v_output = alg.addition(alg.scalarMultiply(gamma, v_output), alg.scalarMultiply(learning_rate/n, outputWGrad));

                updateParameters(v_hidden, v_output, learning_rate); // subject to change. may want bias to have this matrix too.
                y_hat = modelSetTest(inputMiniBatches[i]);

                if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]); }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void ANN::Adagrad(double learning_rate, int max_epoch, int mini_batch_size, double e, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        // always evaluate the result 
        // always do forward pass only ONCE at end.
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        
        // Initializing necessary components for Adam. 
        std::vector<std::vector<std::vector<double>>> v_hidden;
        
        std::vector<double> v_output;
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = modelSetTest(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputMiniBatches[i]);

                if(!network.empty() && v_hidden.empty()){ // Initing our tensor
                    v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
                }

                if(v_output.empty()){
                    v_output.resize(outputWGrad.size());
                }

                v_hidden = alg.addition(v_hidden, alg.exponentiate(cumulativeHiddenLayerWGrad, 2));

                v_output = alg.addition(v_output, alg.exponentiate(outputWGrad, 2));

                std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(cumulativeHiddenLayerWGrad, alg.scalarAdd(e, alg.sqrt(v_hidden))));
                std::vector<double> outputLayerUpdation = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(outputWGrad, alg.scalarAdd(e, alg.sqrt(v_output))));

                updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
                y_hat = modelSetTest(inputMiniBatches[i]);

                if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]); }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void ANN::Adadelta(double learning_rate, int max_epoch, int mini_batch_size, double b1, double e, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        // always evaluate the result 
        // always do forward pass only ONCE at end.
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        
        // Initializing necessary components for Adam. 
        std::vector<std::vector<std::vector<double>>> v_hidden;
        
        std::vector<double> v_output;
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = modelSetTest(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputMiniBatches[i]);

                if(!network.empty() && v_hidden.empty()){ // Initing our tensor
                    v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
                }

                if(v_output.empty()){
                    v_output.resize(outputWGrad.size());
                }

                v_hidden = alg.addition(alg.scalarMultiply(1 - b1, v_hidden), alg.scalarMultiply(b1, alg.exponentiate(cumulativeHiddenLayerWGrad, 2)));

                v_output = alg.addition(v_output, alg.exponentiate(outputWGrad, 2));

                std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(cumulativeHiddenLayerWGrad, alg.scalarAdd(e, alg.sqrt(v_hidden))));
                std::vector<double> outputLayerUpdation = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(outputWGrad, alg.scalarAdd(e, alg.sqrt(v_output))));

                updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
                y_hat = modelSetTest(inputMiniBatches[i]);

                if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]); }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

void ANN::Adam(double learning_rate, int max_epoch, int mini_batch_size, double b1, double b2, double e, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        // always evaluate the result 
        // always do forward pass only ONCE at end.
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        
        // Initializing necessary components for Adam. 
        std::vector<std::vector<std::vector<double>>> m_hidden;
        std::vector<std::vector<std::vector<double>>> v_hidden;

        std::vector<double> m_output;
        std::vector<double> v_output;
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = modelSetTest(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputMiniBatches[i]);
                if(!network.empty() && m_hidden.empty() && v_hidden.empty()){ // Initing our tensor
                    m_hidden = alg.resize(m_hidden, cumulativeHiddenLayerWGrad);
                    v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
                }

                if(m_output.empty() && v_output.empty()){
                    m_output.resize(outputWGrad.size());
                    v_output.resize(outputWGrad.size());
                }

                m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulativeHiddenLayerWGrad));
                v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulativeHiddenLayerWGrad, 2)));

                m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, outputWGrad));
                v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(outputWGrad, 2)));

                std::vector<std::vector<std::vector<double>>> m_hidden_hat = alg.scalarMultiply(1/(1 - std::pow(b1, epoch)), m_hidden);
                std::vector<std::vector<std::vector<double>>> v_hidden_hat = alg.scalarMultiply(1/(1 - std::pow(b2, epoch)), v_hidden);

                std::vector<double> m_output_hat = alg.scalarMultiply(1/(1 - std::pow(b1, epoch)), m_output);
                std::vector<double> v_output_hat = alg.scalarMultiply(1/(1 - std::pow(b2, epoch)), v_output);

                std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(m_hidden_hat, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
                std::vector<double> outputLayerUpdation = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(m_output_hat, alg.scalarAdd(e, alg.sqrt(v_output_hat))));


                updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
                y_hat = modelSetTest(inputMiniBatches[i]);

                if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]); }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void ANN::Adamax(double learning_rate, int max_epoch, int mini_batch_size, double b1, double b2, double e, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        // always evaluate the result 
        // always do forward pass only ONCE at end.
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        
        // Initializing necessary components for Adam. 
        std::vector<std::vector<std::vector<double>>> m_hidden;
        std::vector<std::vector<std::vector<double>>> u_hidden;

        std::vector<double> m_output;
        std::vector<double> u_output;
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = modelSetTest(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputMiniBatches[i]);
                if(!network.empty() && m_hidden.empty() && u_hidden.empty()){ // Initing our tensor
                    m_hidden = alg.resize(m_hidden, cumulativeHiddenLayerWGrad);
                    u_hidden = alg.resize(u_hidden, cumulativeHiddenLayerWGrad);
                }

                if(m_output.empty() && u_output.empty()){
                    m_output.resize(outputWGrad.size());
                    u_output.resize(outputWGrad.size());
                }

                m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulativeHiddenLayerWGrad));
                u_hidden = alg.max(alg.scalarMultiply(b2, u_hidden), alg.abs(cumulativeHiddenLayerWGrad));

                m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, outputWGrad));
                u_output = alg.max(alg.scalarMultiply(b2, u_output), alg.abs(outputWGrad));

                std::vector<std::vector<std::vector<double>>> m_hidden_hat = alg.scalarMultiply(1/(1 - std::pow(b1, epoch)), m_hidden);

                std::vector<double> m_output_hat = alg.scalarMultiply(1/(1 - std::pow(b1, epoch)), m_output);

                std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(m_hidden_hat, alg.scalarAdd(e, u_hidden)));
                std::vector<double> outputLayerUpdation = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(m_output_hat, alg.scalarAdd(e, u_output)));


                updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
                y_hat = modelSetTest(inputMiniBatches[i]);

                if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]); }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }
    
    void ANN::Nadam(double learning_rate, int max_epoch, int mini_batch_size, double b1, double b2, double e, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        // always evaluate the result 
        // always do forward pass only ONCE at end.
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        
        // Initializing necessary components for Adam. 
        std::vector<std::vector<std::vector<double>>> m_hidden;
        std::vector<std::vector<std::vector<double>>> v_hidden;
        std::vector<std::vector<std::vector<double>>> m_hidden_final;

        std::vector<double> m_output;
        std::vector<double> v_output;
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = modelSetTest(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputMiniBatches[i]);
                if(!network.empty() && m_hidden.empty() && v_hidden.empty()){ // Initing our tensor
                    m_hidden = alg.resize(m_hidden, cumulativeHiddenLayerWGrad);
                    v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
                }

                if(m_output.empty() && v_output.empty()){
                    m_output.resize(outputWGrad.size());
                    v_output.resize(outputWGrad.size());
                }

                m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulativeHiddenLayerWGrad));
                v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulativeHiddenLayerWGrad, 2)));
                

                m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, outputWGrad));
                v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(outputWGrad, 2)));

                std::vector<std::vector<std::vector<double>>> m_hidden_hat = alg.scalarMultiply(1/(1 - std::pow(b1, epoch)), m_hidden);
                std::vector<std::vector<std::vector<double>>> v_hidden_hat = alg.scalarMultiply(1/(1 - std::pow(b2, epoch)), v_hidden);
                std::vector<std::vector<std::vector<double>>> m_hidden_final = alg.addition(alg.scalarMultiply(b1, m_hidden_hat), alg.scalarMultiply((1 - b1)/(1 - std::pow(b1, epoch)), cumulativeHiddenLayerWGrad));

                std::vector<double> m_output_hat = alg.scalarMultiply(1/(1 - std::pow(b1, epoch)), m_output);
                std::vector<double> v_output_hat = alg.scalarMultiply(1/(1 - std::pow(b2, epoch)), v_output);
                std::vector<double> m_output_final = alg.addition(alg.scalarMultiply(b1, m_output_hat), alg.scalarMultiply((1 - b1)/(1 - std::pow(b1, epoch)), outputWGrad));

                std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(m_hidden_final, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
                std::vector<double> outputLayerUpdation = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(m_output_final, alg.scalarAdd(e, alg.sqrt(v_output_hat))));


                updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
                y_hat = modelSetTest(inputMiniBatches[i]);

                if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]); }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    void ANN::AMSGrad(double learning_rate, int max_epoch, int mini_batch_size, double b1, double b2, double e, bool UI){
        class Cost cost; 
        LinAlg alg;

        double cost_prev = 0;
        int epoch = 1;
        double initial_learning_rate = learning_rate;

        // Creating the mini-batches
        int n_mini_batch = n/mini_batch_size;
        // always evaluate the result 
        // always do forward pass only ONCE at end.
        auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
        
        // Initializing necessary components for Adam. 
        std::vector<std::vector<std::vector<double>>> m_hidden;
        std::vector<std::vector<std::vector<double>>> v_hidden;

        std::vector<std::vector<std::vector<double>>> v_hidden_hat;

        std::vector<double> m_output;
        std::vector<double> v_output;

        std::vector<double> v_output_hat;
        while(true){
            learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
            for(int i = 0; i < n_mini_batch; i++){
                std::vector<double> y_hat = modelSetTest(inputMiniBatches[i]);
                cost_prev = Cost(y_hat, outputMiniBatches[i]);

                auto [cumulativeHiddenLayerWGrad, outputWGrad] = computeGradients(y_hat, outputMiniBatches[i]);
                if(!network.empty() && m_hidden.empty() && v_hidden.empty()){ // Initing our tensor
                    m_hidden = alg.resize(m_hidden, cumulativeHiddenLayerWGrad);
                    v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
                    v_hidden_hat = alg.resize(v_hidden_hat, cumulativeHiddenLayerWGrad);
                }

                if(m_output.empty() && v_output.empty()){
                    m_output.resize(outputWGrad.size());
                    v_output.resize(outputWGrad.size());
                    v_output_hat.resize(outputWGrad.size());
                }

                m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulativeHiddenLayerWGrad));
                v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulativeHiddenLayerWGrad, 2)));

                m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, outputWGrad));
                v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(outputWGrad, 2)));

                v_hidden_hat = alg.max(v_hidden_hat, v_hidden);

                v_output_hat = alg.max(v_output_hat, v_output);

                std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(m_hidden, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
                std::vector<double> outputLayerUpdation = alg.scalarMultiply(learning_rate/n, alg.elementWiseDivision(m_output, alg.scalarAdd(e, alg.sqrt(v_output_hat))));


                updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
                y_hat = modelSetTest(inputMiniBatches[i]);

                if(UI) { ANN::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]); }
            }
            epoch++;
            if(epoch > max_epoch) { break; }
        }
        forwardPass();
    }

    double ANN::score(){
        Utilities util;
        forwardPass();
        return util.performance(y_hat, outputSet);
    }

    void ANN::save(std::string fileName){
        Utilities util;
        if(!network.empty()){
            util.saveParameters(fileName, network[0].weights, network[0].bias, 0, 1);
            for(int i = 1; i < network.size(); i++){
                util.saveParameters(fileName, network[i].weights, network[i].bias, 1, i + 1); 
            }
            util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, 1, network.size() + 1);
        }
        else{
            util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, 0, network.size() + 1);
        }
     }

     void ANN::setLearningRateScheduler(std::string type, double decayConstant){
        lrScheduler = type;
        ANN::decayConstant = decayConstant;
     }

     void ANN::setLearningRateScheduler(std::string type, double decayConstant, double dropRate){
         lrScheduler = type; 
         ANN::decayConstant = decayConstant;
         ANN::dropRate = dropRate;
     }

    // https://en.wikipedia.org/wiki/Learning_rate
    // Learning Rate Decay (C2W2L09) - Andrew Ng - Deep Learning Specialization
     double ANN::applyLearningRateScheduler(double learningRate, double decayConstant, double epoch, double dropRate){
         if(lrScheduler == "Time"){
             return learningRate / (1 + decayConstant * epoch);
         }
         else if(lrScheduler == "Epoch"){
             return learningRate * (decayConstant / std::sqrt(epoch));
         }
         else if(lrScheduler == "Step"){
            return learningRate * std::pow(decayConstant, int((1 + epoch)/dropRate)); // Utilizing an explicit int conversion implicitly takes the floor.
         }
        else if(lrScheduler == "Exponential"){
             return learningRate * std::exp(-decayConstant * epoch);
         }
         return learningRate;
     }

    void ANN::addLayer(int n_hidden, std::string activation, std::string weightInit, std::string reg, double lambda, double alpha){
        if(network.empty()){
            network.push_back(HiddenLayer(n_hidden, activation, inputSet, weightInit, reg, lambda, alpha));
            network[0].forwardPass();
        }
        else{
            network.push_back(HiddenLayer(n_hidden, activation, network[network.size() - 1].a, weightInit, reg, lambda, alpha));
            network[network.size() - 1].forwardPass();
        }
    }
    
    void ANN::addOutputLayer(std::string activation, std::string loss, std::string weightInit, std::string reg, double lambda, double alpha){
        LinAlg alg;
        if(!network.empty()){
            outputLayer = new OutputLayer(network[network.size() - 1].n_hidden, activation, loss, network[network.size() - 1].a, weightInit, reg, lambda, alpha);
        }
        else{
            outputLayer = new OutputLayer(k, activation, loss, inputSet, weightInit, reg, lambda, alpha);
        }
    }

    double ANN::Cost(std::vector<double> y_hat, std::vector<double> y){
        Reg regularization;
        class Cost cost;
        double totalRegTerm = 0;

        auto cost_function = outputLayer->cost_map[outputLayer->cost];
        if(!network.empty()){
            for(int i = 0; i < network.size() - 1; i++){
                totalRegTerm += regularization.regTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg);
            }
        }
        return (cost.*cost_function)(y_hat, y) + totalRegTerm + regularization.regTerm(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg);
    }

    void ANN::forwardPass(){
        if(!network.empty()){
            network[0].input = inputSet;
            network[0].forwardPass();

            for(int i = 1; i < network.size(); i++){
                network[i].input = network[i - 1].a;
                network[i].forwardPass();
            }
            outputLayer->input = network[network.size() - 1].a;
        }
        else{
            outputLayer->input = inputSet;
        }
        outputLayer->forwardPass();
        y_hat = outputLayer->a;
    }

    void ANN::updateParameters(std::vector<std::vector<std::vector<double>>> hiddenLayerUpdations, std::vector<double> outputLayerUpdation, double learning_rate){
        LinAlg alg;

        outputLayer->weights = alg.subtraction(outputLayer->weights, outputLayerUpdation);
        outputLayer->bias -= learning_rate * alg.sum_elements(outputLayer->delta) / n;

        if(!network.empty()){
                
            network[network.size() - 1].weights = alg.subtraction(network[network.size() - 1].weights, hiddenLayerUpdations[0]);
            network[network.size() - 1].bias = alg.subtractMatrixRows(network[network.size() - 1].bias, alg.scalarMultiply(learning_rate/n, network[network.size() - 1].delta));

            for(int i = network.size() - 2; i >= 0; i--){
                network[i].weights = alg.subtraction(network[i].weights, hiddenLayerUpdations[(network.size() - 2) - i + 1]);
                network[i].bias = alg.subtractMatrixRows(network[i].bias, alg.scalarMultiply(learning_rate/n, network[i].delta));
            }
        }
    }
    
    std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<double>> ANN::computeGradients(std::vector<double> y_hat, std::vector<double> outputSet){
       // std::cout << "BEGIN" << std::endl;
        class Cost cost; 
        Activation avn;
        LinAlg alg;
        Reg regularization;

        std::vector<std::vector<std::vector<double>>> cumulativeHiddenLayerWGrad; // Tensor containing ALL hidden grads. 

        auto costDeriv = outputLayer->costDeriv_map[outputLayer->cost];
        auto outputAvn = outputLayer->activation_map[outputLayer->activation];
        outputLayer->delta = alg.hadamard_product((cost.*costDeriv)(y_hat, outputSet), (avn.*outputAvn)(outputLayer->z, 1));
        std::vector<double> outputWGrad = alg.mat_vec_mult(alg.transpose(outputLayer->input), outputLayer->delta);
        outputWGrad = alg.addition(outputWGrad, regularization.regDerivTerm(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg));

        if(!network.empty()){
            auto hiddenLayerAvn = network[network.size() - 1].activation_map[network[network.size() - 1].activation];
            network[network.size() - 1].delta = alg.hadamard_product(alg.outerProduct(outputLayer->delta, outputLayer->weights), (avn.*hiddenLayerAvn)(network[network.size() - 1].z, 1));
            std::vector<std::vector<double>> hiddenLayerWGrad = alg.matmult(alg.transpose(network[network.size() - 1].input), network[network.size() - 1].delta);

            cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(network[network.size() - 1].weights, network[network.size() - 1].lambda, network[network.size() - 1].alpha, network[network.size() - 1].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well. 

            for(int i = network.size() - 2; i >= 0; i--){
                auto hiddenLayerAvn = network[i].activation_map[network[i].activation];
                network[i].delta = alg.hadamard_product(alg.matmult(network[i + 1].delta, alg.transpose(network[i + 1].weights)), (avn.*hiddenLayerAvn)(network[i].z, 1));
                std::vector<std::vector<double>> hiddenLayerWGrad = alg.matmult(alg.transpose(network[i].input), network[i].delta);
                cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

            }
        }
        return {cumulativeHiddenLayerWGrad, outputWGrad};
    }

    void ANN::UI(int epoch, double cost_prev, std::vector<double> y_hat, std::vector<double> outputSet){
        Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
        std::cout << "Layer " << network.size() + 1 << ": " << std::endl;
        Utilities::UI(outputLayer->weights, outputLayer->bias); 
        if(!network.empty()){ 
            for(int i = network.size() - 1; i >= 0; i--){
                std::cout << "Layer " << i + 1 << ": " << std::endl;
                Utilities::UI(network[i].weights, network[i].bias); 
            }
        }
    }
}