//
//  Data.hpp
//  MLP
//
//  Created by Marc Melikyan on 11/4/20.
//

#ifndef Data_hpp
#define Data_hpp

#include <vector>
#include <tuple>
#include <string>


namespace MLPP{
class Data{
        public:
        // Load Datasets
            std::tuple<std::vector<std::vector<double>>, std::vector<double>> loadBreastCancer();
            std::tuple<std::vector<std::vector<double>>, std::vector<double>> loadBreastCancerSVC();
            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> loadIris();
            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> loadWine();
            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> loadMnistTrain();
            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> loadMnistTest();
            std::tuple<std::vector<std::vector<double>>, std::vector<double>> loadCaliforniaHousing();
            std::tuple<std::vector<double>, std::vector<double>> loadFiresAndCrime();

            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> trainTestSplit(std::vector<std::vector<double>> inputSet, std::vector<std::vector<double>> outputSet, double testSize);

        // Supervised
        void setData(int k, std::string fileName, std::vector<std::vector<double>>& inputSet, std::vector<double>& outputSet);
        void printData(std::vector <std::string> inputName, std::string outputName, std::vector<std::vector<double>> inputSet, std::vector<double> outputSet);
        
        // Unsupervised
        void setData(int k, std::string fileName, std::vector<std::vector<double>>& inputSet);
        void printData(std::vector <std::string> inputName, std::vector<std::vector<double>> inputSet);
        
        // Simple
        void setData(std::string fileName, std::vector <double>& inputSet, std::vector <double>& outputSet);
        void printData(std::string& inputName, std::string& outputName, std::vector <double>& inputSet, std::vector <double>& outputSet);

        // Images
        std::vector<std::vector<double>> rgb2gray(std::vector<std::vector<std::vector<double>>> input);
        std::vector<std::vector<std::vector<double>>> rgb2ycbcr(std::vector<std::vector<std::vector<double>>> input);
        std::vector<std::vector<std::vector<double>>> rgb2hsv(std::vector<std::vector<std::vector<double>>> input);
        std::vector<std::vector<std::vector<double>>> rgb2xyz(std::vector<std::vector<std::vector<double>>> input);
        std::vector<std::vector<std::vector<double>>> xyz2rgb(std::vector<std::vector<std::vector<double>>> input);

        // Text-Based & NLP
        std::string toLower(std::string text);
        std::vector<char> split(std::string text);
        std::vector<std::string> splitSentences(std::string data);
        std::vector<std::string> removeSpaces(std::vector<std::string> data);
        std::vector<std::string> removeNullByte(std::vector<std::string> data);
        std::vector<std::string> segment(std::string text);
        std::vector<double> tokenize(std::string text);
        std::vector<std::string> removeStopWords(std::string text);
        std::vector<std::string> removeStopWords(std::vector<std::string> segmented_data);
        
        std::string stemming(std::string text);
        
        std::vector<std::vector<double>> BOW(std::vector<std::string> sentences, std::string = "Default"); 
        std::vector<std::vector<double>> TFIDF(std::vector<std::string> sentences);
        std::tuple<std::vector<std::vector<double>>, std::vector<std::string>> word2Vec(std::vector<std::string> sentences, std::string type, int windowSize, int dimension, double learning_rate, int max_epoch);
        std::vector<std::vector<double>> LSA(std::vector<std::string> sentences, int dim);

        std::vector<std::string> createWordList(std::vector<std::string> sentences);

        // Extra
        void setInputNames(std::string fileName, std::vector<std::string>& inputNames);
        std::vector<std::vector<double>> featureScaling(std::vector<std::vector<double>> X);
        std::vector<std::vector<double>> meanNormalization(std::vector<std::vector<double>> X);
        std::vector<std::vector<double>> meanCentering(std::vector<std::vector<double>> X);
        std::vector<std::vector<double>> oneHotRep (std::vector<double> tempOutputSet, int n_class); 
        std::vector<double> reverseOneHot(std::vector<std::vector<double>> tempOutputSet); 

        template <class T>
        std::vector<T> vecToSet(std::vector<T> inputSet){
            std::vector<T> setInputSet;
            for(int i = 0; i < inputSet.size(); i++){
                bool new_element = true; 
                for(int j = 0; j < setInputSet.size(); j++){
                    if(setInputSet[j] == inputSet[i]){
                        new_element = false;
                    }
                }
                if(new_element){
                    setInputSet.push_back(inputSet[i]);
                }
            }
            return setInputSet;
        }

        private:
    };
}

#endif /* Data_hpp */
