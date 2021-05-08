# Fake News Classification
## Overview
The goal of this project is to leverage machine learning models to make fast and accurate fake news detection. I implement two ML approaches to classify the fake news: one is Random Forrest model, the classical classification method, and the other one is a dense neural network model using BERT, the state-of-the-art NLP method brought by Google scientists. A complete introduction of the model design and build-up can be found in my [Medium article](https://xutianjian.medium.com/fake-news-classification-use-machine-learning-to-fight-against-fake-news-e0da735e55e0). The fake news dataset used for model training are originally from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=True.csv).

## Random Forest Model
I use TF-IDF algorithm for text vectorization. Here is the performance of the Random Forest model:
```
Training Time (in seconds) = 0.371672 
Testing Time (in seconds) = 0.049181 
=============Evaluation Result============== 
Random forrest classifier accuracy: 0.899332 
Random forrest classifier precision: 0.893516 
Random forrest classifier recall: 0.896427 
Random forrest classifier f1_score: 0.894969
```
![image](https://user-images.githubusercontent.com/41350819/117528419-07214a00-af87-11eb-9e9f-51f42d50a94e.png)


## BERT + DNN Model
I use BertTokenizer for word tokenization. The design of the Dense Neural Network looks like:
![image](https://user-images.githubusercontent.com/41350819/117400115-d8d43980-aeb6-11eb-9c9a-ddaef869ce7b.png)

Here is the performance of the BERT + DNN Model:
```
Training Time (in minutes) = 41.600925
Testing Time (in minutes) = 11.135320
===============Evaluation End===============
BERT classifier accuracy: 0.962918 
BERT classifier precision: 0.958473 
BERT classifier recall: 0.964273 
BERT classifier f1_score: 0.961364
```
![image](https://user-images.githubusercontent.com/41350819/117528435-1dc7a100-af87-11eb-89c1-5f9e1b2e6423.png)

## Model Comparison
![image](https://user-images.githubusercontent.com/41350819/117528447-2a4bf980-af87-11eb-8bb1-04bc09f9277f.png)

