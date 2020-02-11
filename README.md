# Prediction-of-Car-Insurance-Claims
Based on the researches on the subject of car insurance, constructed machine learning models to classify customers by characteristics for insurance customers and predicted claim amount.  
This project aimed to provide more information to the car insurance market and make transactions more viable and efficient.

Team Members: 

Xi Zhang, Lianjie Shan
![image](https://github.com/f0000000x/Prediction-of-Car-Insurance-Claims/blob/master/Images/carInsur.jpg)

# Background
As the automobile gradually becomes the necessity of every family, the car insurance also becomes more and more prosperous for auto insurance companies, to maximize revenue, they must sell corresponding insurance plans to different customers. However, because the types of customers are so diverse and the correlation between the characteristics is not obvious, the use of simple statistics cannot enable insurance companies to make accurate judgments about customers. With the advent of machine learning, more models are available to learn data in depth. Thus, more accurate predictions can be achieved.

# Motivation
Machine learning, as the most cutting-edge technology, is able to dig deeper information through known data without the influence of appearances.Insurance market data are so complex that it is difficult to extract macro direction from large amounts of data. In theory, using machine learning as a tool to mine information is very efficient but the current market has little to offer.So we think this project is very valuable.

# Definiation of the Problem
![image](https://github.com/f0000000x/Prediction-of-Car-Insurance-Claims/blob/master/Images/frame.png)

Our initial plan is to use the existing data of the auto insurance company to train the models, so that when new customers come in, we can use these models to predict whether they will claim or not according to their characteristics. With only two types of target data, SVM became our first choice.
After determining whether the customer will claim in the future, we divide the customers who will claim and use the new model to predict the amount.Due to the high tolerance of multilayer pecenptron to data and its better handling of specific values, we decided to use MLP network to predict specific values.
After training the MLP model with existing data, if the evaluation is reliable, we can use the characteristics of new customers to predict their future claims. But before we can implement this process, we need to start with data processing.

# Data Description
The data we chose was released by Kaggle, an open-source data site. The distributor xiaomengsun published it in 2018. It is made up of a record of 10,302 observations and 27 variables. This data can be downloaded from the following websites for study and research:
https://www.kaggle.com/xiaomengsun/car-insurance-claim-data

