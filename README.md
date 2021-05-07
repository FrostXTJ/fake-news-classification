# Fake News Classification
## Overview
The goal of this project is to leverage machine learning models to make fast and accurate fake news detection. I implement two ML approaches to classify the fake news: one is Random Forrest model, the classical classification method, and the other one is a dense neural network model using BERT, the state-of-the-art NLP method brought by Google scientists. A complete introduction of the model design and build-up can be found in my [Medium article](https://xutianjian.medium.com/fake-news-classification-use-machine-learning-to-fight-against-fake-news-e0da735e55e0). The fake news dataset used for model training are originally from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=True.csv).

## Random Forest Model
I use TF-IDF algorithm for text vectorization. Here is the performance of the Random Forest model:
```
Training Time (in seconds) = 0.366178 
Testing Time (in seconds) = 0.029793 
=============Evaluation Result============== 
Random forrest classifier accuracy: 0.882684 
Random forrest classifier precision: 0.872125 
Random forrest classifier recall: 0.884019 
Random forrest classifier f1_score: 0.878032
```
![image](https://user-images.githubusercontent.com/41350819/117400040-b04c3f80-aeb6-11eb-982a-0f10fb48d163.png)

## BERT + DNN Model
I use BertTokenizer for word tokenization. The design of the Dense Neural Network looks like:
![image](https://user-images.githubusercontent.com/41350819/117400115-d8d43980-aeb6-11eb-9c9a-ddaef869ce7b.png)

Here is the performance of the BERT + DNN Model:
```
Training Time (in minutes) = 16.584202
Testing Time (in minutes) = 10.621744
===============Evaluation End=============== 
BERT classifier accuracy: 0.947272 
BERT classifier precision: 0.983650 
BERT classifier recall: 0.904651 
BERT classifier f1_core: 0.942498
```
![image](https://user-images.githubusercontent.com/41350819/117400172-f1dcea80-aeb6-11eb-8d40-d4b37c97d875.png)
