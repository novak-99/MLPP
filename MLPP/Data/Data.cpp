//
//  Data.cpp
//  MLP
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "Data.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Stat/Stat.hpp"
#include "SoftmaxNet/SoftmaxNet.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


namespace MLPP{
    // MULTIVARIATE SUPERVISED

    void Data::setData(int k, std::string fileName, std::vector<std::vector<double>>& inputSet, std::vector<double>& outputSet){
        LinAlg alg;
        std::string inputTemp;
        std::string outputTemp;
        
        inputSet.resize(k);
        
        std::ifstream dataFile(fileName);
        if(!dataFile.is_open()){
            std::cout << fileName << " failed to open." << std::endl;
        }
        
        std::string line;
        while(std::getline(dataFile, line)){
            std::stringstream ss(line);
       
            for(int i = 0; i < k; i++){
                std::getline(ss, inputTemp, ',');
                inputSet[i].push_back(std::stod(inputTemp));
                
            }
            
            std::getline(ss, outputTemp, ',');
            outputSet.push_back(std::stod(outputTemp));
        }
        inputSet = alg.transpose(inputSet);
        dataFile.close();
    }

    void Data::printData(std::vector <std::string> inputName, std::string outputName, std::vector<std::vector<double>> inputSet, std::vector<double> outputSet){
        LinAlg alg;
        inputSet = alg.transpose(inputSet);
        for(int i = 0; i < inputSet.size(); i++){
            std::cout << inputName[i] << std::endl;
            for(int j = 0; j < inputSet[i].size(); j++){
                std::cout << inputSet[i][j] << std::endl;
            }
        }
        
        std::cout << outputName << std::endl;
        for(int i = 0; i < outputSet.size(); i++){
            std::cout << outputSet[i] << std::endl;
        }
    }

    // UNSUPERVISED

    void Data::setData(int k, std::string fileName, std::vector<std::vector<double>>& inputSet){
        LinAlg alg;
        std::string inputTemp;
        
        inputSet.resize(k);
        
        std::ifstream dataFile(fileName);
        if(!dataFile.is_open()){
            std::cout << fileName << " failed to open." << std::endl;
        }
        
        std::string line;
        while(std::getline(dataFile, line)){
            std::stringstream ss(line);
       
            for(int i = 0; i < k; i++){
                std::getline(ss, inputTemp, ',');
                inputSet[i].push_back(std::stod(inputTemp));
                
            }
        }
        inputSet = alg.transpose(inputSet);
        dataFile.close();
    }

    void Data::printData(std::vector <std::string> inputName, std::vector<std::vector<double>> inputSet){
        LinAlg alg;
        inputSet = alg.transpose(inputSet);
        for(int i = 0; i < inputSet.size(); i++){
            std::cout << inputName[i] << std::endl;
            for(int j = 0; j < inputSet[i].size(); j++){
                std::cout << inputSet[i][j] << std::endl;
            }
        }
    }

    // SIMPLE

    void Data::setData(std::string fileName, std::vector <double>& inputSet, std::vector <double>& outputSet){
        std::string inputTemp, outputTemp;
        
        std::ifstream dataFile(fileName);
        if(!dataFile.is_open()){
            std::cout << "The file failed to open." << std::endl;
        }
        
        std::string line;
        
        
        while(std::getline(dataFile, line)){
            std::stringstream ss(line);

            std::getline(ss, inputTemp, ',');
            std::getline(ss, outputTemp, ',');
            
            inputSet.push_back(std::stod(inputTemp));
            outputSet.push_back(std::stod(outputTemp));
        }
            
        dataFile.close();
    }

    void Data::printData(std::string& inputName, std::string& outputName, std::vector <double>& inputSet, std::vector <double>& outputSet){
        std::cout << inputName << std::endl;
        for(int i = 0; i < inputSet.size(); i++){
            std::cout << inputSet[i] << std::endl;
        }
        
        std::cout << outputName << std::endl;
        for(int i = 0; i < inputSet.size(); i++){
            std::cout << outputSet[i] << std::endl;
        }
    }

    // Images
    
    // TEXT-BASED & NLP
    std::string Data::toLower(std::string text){
        for(int i = 0; i < text.size(); i++){
            text[i] = tolower(text[i]);
        }
        return text;
    }

    std::vector<char> Data::split(std::string text){
        std::vector<char> split_data;
        for(int i = 0; i < text.size(); i++){
            split_data.push_back(text[i]);
        }
        return split_data;
    }

    std::vector<std::string> Data::splitSentences(std::string data){
        std::vector<std::string> sentences;
        std::string currentStr = "";

        for(int i = 0; i < data.length(); i++){
            currentStr.push_back(data[i]); 
            if(data[i] == '.' && data[i + 1] != '.'){
                sentences.push_back(currentStr);
                currentStr = "";
                i++;
            }
        }
        return sentences;
    }

    std::vector<std::string> Data::removeSpaces(std::vector<std::string> data){
        for(int i = 0; i < data.size(); i++){
            auto it = data[i].begin();
            for(int j = 0; j < data[i].length(); j++){
                if(data[i][j] == ' '){
                    data[i].erase(it);
                }
                it++;
            }
        }
        return data; 
    }

    std::vector<std::string> Data::removeNullByte(std::vector<std::string> data){
        for(int i = 0; i < data.size(); i++){
            if(data[i] == "\0"){
                data.erase(data.begin() + i);
            }
        }
        return data; 
    }

    std::vector<std::string> Data::segment(std::string text){
        std::vector<std::string> segmented_data;
        int prev_delim = 0;
        for(int i = 0; i < text.length(); i++){
            if(text[i] == ' '){
                segmented_data.push_back(text.substr(prev_delim, i - prev_delim)); 
                prev_delim = i + 1;  
            }
            else if(text[i] == ',' || text[i] == '!' || text[i] == '.' || text[i] == '-'){
                segmented_data.push_back(text.substr(prev_delim, i - prev_delim)); 
                std::string punc;
                punc.push_back(text[i]);
                segmented_data.push_back(punc);
                prev_delim = i + 2; 
                i++;
            }
            else if(i == text.length() - 1){
                segmented_data.push_back(text.substr(prev_delim, text.length() - prev_delim)); // hehe oops- forgot this
            }
        }

        return segmented_data;
    }

    std::vector<double> Data::tokenize(std::string text){
        int max_num = 0;
        bool new_num = true;
        std::vector<std::string> segmented_data = segment(text);
        std::vector<double> tokenized_data; 
        tokenized_data.resize(segmented_data.size());
        for(int i = 0; i < segmented_data.size(); i++){
            for(int j = i - 1; j >= 0; j--){
                if(segmented_data[i] == segmented_data[j]){
                    tokenized_data[i] = tokenized_data[j];
                    new_num = false;
                }
            }
            if(!new_num){
                new_num = true;
            }
            else{ 
                max_num++;
                tokenized_data[i] = max_num;
            }
        }
        return tokenized_data;
    }

    std::vector<std::string> Data::removeStopWords(std::string text){
        std::vector<std::string> stopWords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}; 
        std::vector<std::string> segmented_data = removeSpaces(segment(toLower(text))); 

        for(int i = 0; i < stopWords.size(); i++){
            for(int j = 0; j < segmented_data.size(); j++){
                if(segmented_data[j] == stopWords[i]){
                    segmented_data.erase(segmented_data.begin() + j);
                }
            }
        }
        return segmented_data;
    }

    std::vector<std::string> Data::removeStopWords(std::vector<std::string> segmented_data){
        std::vector<std::string> stopWords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}; 
        for(int i = 0; i < segmented_data.size(); i++){
            for(int j = 0; j < stopWords.size(); j++){
                if(segmented_data[i] == stopWords[j]){
                    segmented_data.erase(segmented_data.begin() + i);
                }
            }
        }
        return segmented_data;
    }

    std::string Data::stemming(std::string text){

        // Our list of suffixes which we use to compare against
        std::vector<std::string> suffixes = {"eer", "er", "ion", "ity", "ment", "ness", "or", "sion", "ship", "th", "able", "ible", "al", "ant", "ary", "ful", "ic", "ious", "ous", "ive", "less", "y", "ed", "en", "ing", "ize", "ise", "ly", "ward", "wise"};
        int padding_size = 4; 
        char padding = ' '; // our padding

        for(int i = 0; i < padding_size; i++){
            text[text.length() + i] = padding; // ' ' will be our padding value
        }
        

        for(int i = 0; i < text.size(); i++){
            for(int j = 0; j < suffixes.size(); j++){
                if(text.substr(i, suffixes[j].length()) == suffixes[j] && (text[i + suffixes[j].length()] == ' ' || text[i + suffixes[j].length()] == ',' || text[i + suffixes[j].length()] == '-' || text[i + suffixes[j].length()] == '.' || text[i + suffixes[j].length()] == '!')){
                    text.erase(i, suffixes[j].length());
                }
            }
        }

        return text; 
    }

    std::vector<std::vector<double>> Data::BOW(std::vector<std::string> sentences, std::string type){
        /* 
        STEPS OF BOW: 
            1) To lowercase (done by removeStopWords function by def)
            2) Removing stop words
            3) Obtain a list of the used words
            4) Create a one hot encoded vector of the words and sentences
            5) Sentence.size() x list.size() matrix
        */

        std::vector<std::string> wordList = removeNullByte(removeStopWords(createWordList(sentences)));

        std::vector<std::vector<std::string>> segmented_sentences; 
        segmented_sentences.resize(sentences.size());

        for(int i = 0; i < sentences.size(); i++){
            segmented_sentences[i] = removeStopWords(sentences[i]);
        }

        std::vector<std::vector<double>> bow; 

        bow.resize(sentences.size());
        for(int i = 0; i < bow.size(); i++){
            bow[i].resize(wordList.size());
        }


        for(int i = 0; i < segmented_sentences.size(); i++){
            for(int j = 0; j < segmented_sentences[i].size(); j++){
                for(int k = 0; k < wordList.size(); k++){ 
                    if(segmented_sentences[i][j] == wordList[k]){
                        if(type == "Binary"){
                            bow[i][k] = 1;
                        }
                        else{
                            bow[i][k]++;
                        }
                    }
                }
            }
        }
        return bow; 
    }

    std::vector<std::vector<double>> Data::TFIDF(std::vector<std::string> sentences){
        LinAlg alg;
        std::vector<std::string> wordList = removeNullByte(removeStopWords(createWordList(sentences)));

        std::vector<std::vector<std::string>> segmented_sentences; 
        segmented_sentences.resize(sentences.size());

        for(int i = 0; i < sentences.size(); i++){
            segmented_sentences[i] = removeStopWords(sentences[i]);
        }

        std::vector<std::vector<double>> TF; 
        std::vector<int> frequency;
        frequency.resize(wordList.size());
        TF.resize(segmented_sentences.size());
        for(int i = 0; i < TF.size(); i++){
            TF[i].resize(wordList.size());
        }
        for(int i = 0; i < segmented_sentences.size(); i++){
            std::vector<bool> present(wordList.size(), 0);
            for(int j = 0; j < segmented_sentences[i].size(); j++){
                for(int k = 0; k < wordList.size(); k++){
                    if(segmented_sentences[i][j] == wordList[k]){
                        TF[i][k]++;
                        if(!present[k]){
                            frequency[k]++;
                            present[k] = true;   
                        }
                    }
                }
            }
            TF[i] = alg.scalarMultiply(double(1) / double(segmented_sentences[i].size()), TF[i]);
        }

        std::vector<double> IDF; 
        IDF.resize(frequency.size());

        for(int i = 0; i < IDF.size(); i++){
            IDF[i] = log((double)segmented_sentences.size() / (double)frequency[i]);
        }

        std::vector<std::vector<double>> TFIDF;
        TFIDF.resize(segmented_sentences.size());
        for(int i = 0; i < TFIDF.size(); i++){
            TFIDF[i].resize(wordList.size());
        }

        for(int i = 0; i < TFIDF.size(); i++){
            for(int j = 0; j < TFIDF[i].size(); j++){
                TFIDF[i][j] = TF[i][j] * IDF[j];
            }
        }

        return TFIDF;
    }

    std::tuple<std::vector<std::vector<double>>, std::vector<std::string>> Data::word2Vec(std::vector<std::string> sentences, std::string type, int windowSize, int dimension, double learning_rate, int max_epoch){
        std::vector<std::string> wordList = removeNullByte(removeStopWords(createWordList(sentences)));

        std::vector<std::vector<std::string>> segmented_sentences; 
        segmented_sentences.resize(sentences.size());

        for(int i = 0; i < sentences.size(); i++){
            segmented_sentences[i] = removeStopWords(sentences[i]);
        }

        std::vector<std::string> inputStrings; 
        std::vector<std::string> outputStrings; 

        for(int i = 0; i < segmented_sentences.size(); i++){
            for(int j = 0; j < segmented_sentences[i].size(); j++){
                for(int k = windowSize; k > 0; k--){
                    if(j - k >= 0){
                        inputStrings.push_back(segmented_sentences[i][j]);

                        outputStrings.push_back(segmented_sentences[i][j - k]);
                    }
                    if(j + k <= segmented_sentences[i].size() - 1){
                        inputStrings.push_back(segmented_sentences[i][j]);
                        outputStrings.push_back(segmented_sentences[i][j + k]);
                    }
                }
            }
        }
        
        int inputSize = inputStrings.size();

        inputStrings.insert(inputStrings.end(), outputStrings.begin(), outputStrings.end());

        std::vector<std::vector<double>> BOW = Data::BOW(inputStrings, "Binary");

        std::vector<std::vector<double>> inputSet;
        std::vector<std::vector<double>> outputSet; 
        
        for(int i = 0; i < inputSize; i++){
            inputSet.push_back(BOW[i]);
        }

        for(int i = inputSize; i < BOW.size(); i++){
            outputSet.push_back(BOW[i]);
        }
        LinAlg alg;
        SoftmaxNet* model;
        if(type == "Skipgram"){
            model = new SoftmaxNet(outputSet, inputSet, dimension);
        }
        else { // else = CBOW. We maintain it is a default, however. 
            model = new SoftmaxNet(inputSet, outputSet, dimension);
        }
        model->gradientDescent(learning_rate, max_epoch, 1);

        std::vector<std::vector<double>> wordEmbeddings = model->getEmbeddings();
        delete model;
        return {wordEmbeddings, wordList};
    }

    std::vector<std::string> Data::createWordList(std::vector<std::string> sentences){
        std::string combinedText = "";
        for(int i = 0; i < sentences.size(); i++){
            if(i != 0){ combinedText += " "; }
            combinedText += sentences[i];
        }
        
        return removeSpaces(vecToSet(removeStopWords(combinedText)));
    }

    // EXTRA 
    void Data::setInputNames(std::string fileName, std::vector<std::string>& inputNames){
        std::string inputNameTemp;
        std::ifstream dataFile(fileName);
        if(!dataFile.is_open()){
            std::cout << fileName << " failed to open." << std::endl;
        }
        
        while (std::getline(dataFile, inputNameTemp))
        {
            inputNames.push_back(inputNameTemp);
        }
        
        dataFile.close();
    }
    
    std::vector<std::vector<double>> Data::featureScaling(std::vector<std::vector<double>> X){
        LinAlg alg;
        X = alg.transpose(X);
        std::vector<double> max_elements, min_elements;
        max_elements.resize(X.size());
        min_elements.resize(X.size());

        for(int i = 0; i < X.size(); i++){
            max_elements[i] = alg.max(X[i]);
            min_elements[i] = alg.min(X[i]);
        }

        for(int i = 0; i < X.size(); i++){
            for(int j = 0; j < X[i].size(); j++){
                X[i][j] = (X[i][j] - min_elements[i]) / (max_elements[i] - min_elements[i]);
            }
        }
        return alg.transpose(X);
    }


    std::vector<std::vector<double>> Data::meanNormalization(std::vector<std::vector<double>> X){
        LinAlg alg;
        Stat stat; 
        // (X_j - mu_j) / std_j, for every j

        X = meanCentering(X);
        for(int i = 0; i < X.size(); i++){
            X[i] = alg.scalarMultiply(1/stat.standardDeviation(X[i]), X[i]);
        }
        return X;
    }

    std::vector<std::vector<double>> Data::meanCentering(std::vector<std::vector<double>> X){
        LinAlg alg;
        Stat stat; 
        for(int i = 0; i < X.size(); i++){
            double mean_i = stat.mean(X[i]);
            for(int j = 0; j < X[i].size(); j++){
                X[i][j] -= mean_i; 
            }
        }
        return X; 
    }
    
    std::vector<std::vector<double>> Data::oneHotRep(std::vector<double> tempOutputSet, int n_class){
        std::vector<std::vector<double>> outputSet;
        outputSet.resize(tempOutputSet.size());
        for(int i = 0; i < tempOutputSet.size(); i++){
            for(int j = 0; j <= n_class - 1; j++){
                if(tempOutputSet[i] == j){
                    outputSet[i].push_back(1);
                }
                else{
                    outputSet[i].push_back(0);
                }
            }
        }
        return outputSet;
    }

    std::vector<double> Data::reverseOneHot(std::vector<std::vector<double>> tempOutputSet){
        std::vector<double> outputSet;
        int n_class = tempOutputSet[0].size();
        for(int i = 0; i < tempOutputSet.size(); i++){
            int current_class = 1;
            for(int j = 0; j < tempOutputSet[i].size(); j++){
                if(tempOutputSet[i][j] == 1){
                    break;
                } 
                else{
                    current_class++;
                }
            }
            outputSet.push_back(current_class);
        }

        return outputSet;
    }
}
