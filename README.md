# Prediction-of-Car-Insurance-Claims
Based on the researches on the subject of car insurance, constructed machine learning models to classify customers by characteristics for insurance customers and predicted claim amount.  
This project aimed to provide more information to the car insurance market and make transactions more viable and efficient.

Team Members: 

Xi Zhang, Lianjie Shan
<div align="center">
<img src="https://github.com/f0000000x/Prediction-of-Car-Insurance-Claims/blob/master/Images/carInsur.jpg" align=center />
</div>
# Background
As the automobile gradually becomes the necessity of every family, the car insurance also becomes more and more prosperous for auto insurance companies, to maximize revenue, they must sell corresponding insurance plans to different customers. However, because the types of customers are so diverse and the correlation between the characteristics is not obvious, the use of simple statistics cannot enable insurance companies to make accurate judgments about customers. With the advent of machine learning, more models are available to learn data in depth. Thus, more accurate predictions can be achieved.

# Motivation
Machine learning, as the most cutting-edge technology, is able to dig deeper information through known data without the influence of appearances.Insurance market data are so complex that it is difficult to extract macro direction from large amounts of data. In theory, using machine learning as a tool to mine information is very efficient but the current market has little to offer.So we think this project is very valuable.

# Definiation of the Problem

<img align="left" src="https://github.com/f0000000x/Prediction-of-Car-Insurance-Claims/blob/master/Images/frame.png" width="170" height="250" />  
Our initial plan is to use the existing data of the auto insurance company to train the models, so that when new customers come in, we can use these models to predict whether they will claim or not according to their characteristics. With only two types of target data, SVM became our first choice.<br/>  
After determining whether the customer will claim in the future, we divide the customers who will claim and use the new model to predict the amount.Due to the high tolerance of multilayer pecenptron to data and its better handling of specific values, we decided to use MLP network to predict specific values.<br/>    
After training the MLP model with existing data, if the evaluation is reliable, we can use the characteristics of new customers to predict their future claims. But before we can implement this process, we need to start with data processing.<br/>
<br/>

# Data Description
The data we chose was released by Kaggle, an open-source data site. The distributor xiaomengsun published it in 2018. It is made up of a record of 10,302 observations and 27 variables. This data can be downloaded from the following websites for study and research:
[Kaggle](https://www.kaggle.com/xiaomengsun/car-insurance-claim-data)

# Models

## Support Vector Method (SVM)  

<img align="left" src="https://github.com/f0000000x/Prediction-of-Car-Insurance-Claims/blob/master/Images/svm.png" width="270" height="210" /> </div> 

Support Vector Method (SVM) as a popular machine learning tool is most used for classification and regression. Generally speaking, SVM tries to find a plane that has the maximum margin and the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence. <br/>

    
## Multi-layer Perceptron

![img](https://github.com/f0000000x/Prediction-of-Car-Insurance-Claims/blob/master/Images/mlp.png)

In the part of model selection, we hope to train a neural network to achieve our goal because of the complexity of data and the relatively vague correlation between variables. MLP has a high degree of parallel processing, a high degree of nonlinear global function, good fault tolerance, associative memory function, very strong adaptive, self-learning function, so we finally decided to use MLP multilayer perceptron. 

If you are interested in the performace and results of our models, please move to the [report](https://github.com/f0000000x/Prediction-of-Car-Insurance-Claims/blob/master/Final-Group-Project-Report/FinalReport.pdf) 


# Extension
![](https://github.com/f0000000x/Prediction-of-Car-Insurance-Claims/blob/master/Images/k-means.png)
* K-means
In our research, we want to use the k-means algorithm to find an optimal classification group number, that is to say, the classification group number that can make the value of MSE become the smallest. In this case, we are breaking down the original data into k classes, and within each of those classes we will re-using MLP to build a predictive model. That brings our total number of predictive models to K.
When we use the established K models to make predictions, we first find the cluster to which the input (customer) belongs, and then use the model to predict the value of claim. With the assistance of this method, we can minimize the value of MSE and improve the accuracy of the model.

# References
* Sato, Kaz. “Using Machine Learning for Insurance Pricing Optimization | Google Cloud Blog.” Google, Google Cloud Platform, 19 Mar. 2017, cloud.google.com/blog/big-data/2017/03/using-machine-learning-for-insurance-pricing-optimization.
* Malhotra, Ravi, and Swati Sharma. MACHINE LEARNING IN INSURANCE - Accenture.com. Accenture, 2018, www.accenture.com/t20180822T093440Z__w__/us-en/_acnmedia/PDF-84/Accenture-Machine-Leaning-Insurance.pdf.
* Raschka, Sebastian, and Vahid Mirjalili. Python Machine Learning: Machine Learning and Deep Learning with Python, Scikit-Learn, and TensorFlow. Pack Publishing, 2018.


