#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Copy:60 Modified:7 Own:76 Result = 38.97%
#--------------------------------------------------------------------------------------------
#Copy 6 Modified 0 Own 6
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
plt.style.use('ggplot')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#--------------------------------------------------------------------------------------------
#Data Preparation Copy 54 Modified 0 Own 70
#--------------------------------------------------------------------------------------------
#C0 M0 O5
#Raw data
raw = pd.read_csv("car_insurance_claim.csv")
raw.shape
raw.head()
list(raw)
print(str(raw))

#C0 M0 O5
#Drop repeating and useless columns
df = raw.drop(['ID','BIRTH','OCCUPATION','CAR_TYPE','CLAIM_FLAG'], axis=1)
#Convert all the 'No' ,'Female' ,'Private' and 'Rural' categpries into numberic values(0).
df = df.replace(['No', 'z_No', 'no', 'z_F', 'Private', 'z_Highly Rural/ Rural'], 
                     [0, 0, 0, 0, 0, 0]) 
#Convert all the 'Yes' ,'Male' ,'Commerical' and 'Urban' categpries into numberic values(1).
df = df.replace(['Yes', 'yes', 'M', 'Commercial', 'Highly Urban/ Urban'], 
                     [1, 1, 1, 1, 1]) 
#Convert the education level into numberic values(0-3).
df = df.replace(['z_High School', '<High School', 'Bachelors', 'Masters', 'PhD'], 
                     [0, 0, 1, 2, 3]) 
df.dtypes

#C0 M0 O8
#Convert 'object' and 'float' columns into dtype'int'.
df[df.columns[4]]=df[df.columns[4]].replace('[\$,]', '', regex=True).astype(float)
df[df.columns[6]]=df[df.columns[6]].replace('[\$,]', '', regex=True).astype(float)
df[df.columns[12]]=df[df.columns[12]].replace('[\$,]', '', regex=True).astype(float)
df[df.columns[15]]=df[df.columns[15]].replace('[\$,]', '', regex=True).astype(float)
df[df.columns[19]]=df[df.columns[19]].replace('[\$,]', '', regex=True).astype(float)
df[df.columns[0:23]]=df[df.columns[0:23]].astype(float)
df.shape
df.info()
#--------------------------------------------------------------------------------------------
#C0 M0 O12
#Define a structure function for showing mean, median, min, max and percentile.
def structure(x):
    
    print("Mean                   :", x.mean())
    print("Median                 :", x.median())
    print("Minimum                :", x.min())
    print("Maximum                :", x.max())
    print("25th percentile of arr :", 
       np.percentile(x, 25)) 
    print("50th percentile of arr :",  
       np.percentile(x, 50)) 
    print("75th percentile of arr :", 
       np.percentile(x, 75))

#C0 M0 O4
#Structure of Claim Amount Data
clmamt = df.loc[:,('CLM_AMT')]
structure(clmamt)
plt.boxplot(clmamt)
plt.show()
#--------------------------------------------------------------------------------------------
#C0 M0 O6
#Distribution of the claim amount
clmamt.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Distribution of Claim Amount')
plt.xlabel('Claim Amount')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)
plt.show()
#--------------------------------------------------------------------------------------------
#C0 M0 O3
#Remove outliers
df1w = df[df.CLM_AMT<10000]
df1w.to_csv('df1w.csv')
df1w.info()
#--------------------------------------------------------------------------------------------
#C0 M0 O7
#Distribution of the claim amount(after removing outliers)
df1w.loc[:,('CLM_AMT')].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Distribution of Claim Amount(without outliers)')
plt.xlabel('Claim Amount')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)

#--------------------------------------------------------------------------------------------
#C5 M2 O4
#Correlaton plot
#X = df.loc[:, ('KIDSDRIV','AGE','HOMEKIDS','YOJ','INCOME','PARENT1','HOME_VAL','MSTATUS','GENDER','EDUCATION',
#               'TRAVTIME','CAR_USE','BLUEBOOK','RED_CAR','OLDCLAIM','CLM_FREQ','REVOKED','MVR_PTS',
#               'CAR_AGE','CLAIM_FLAG','URBANICITY','CLM_AMT')]  #independent columns
def corrplt(df,col):
    X = df.loc[:, (list(df1w))]  #independent columns
    y = df.loc[:,(col)]    #target column
    #get correlations of each features in dataset
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    plt.savefig('Corr.png')
corrplt(df1w.dropna(),'CLM_AMT')
#--------------------------------------------------------------------------------------------
#Feature Selection
#C12 M5 O3
def decisiontree(df,col):
    X = df.loc[:, ('KIDSDRIV','AGE','HOMEKIDS','YOJ','INCOME','PARENT1','HOME_VAL','MSTATUS','GENDER','EDUCATION',
                   'TRAVTIME','CAR_USE','BLUEBOOK','RED_CAR','OLDCLAIM','CLM_FREQ','REVOKED','MVR_PTS','CAR_AGE',
                   'URBANICITY')]  #independent columns
    y = df.loc[:,(col)]    #target column
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.savefig('DT.png')
    plt.show()
decisiontree(df1w.dropna(),'CLM_AMT')
#--------------------------------------------------------------------------------------------
#C0 M0 O2
#Select the top5 important features
top5 = df1w.loc[:,('BLUEBOOK','TRAVTIME','INCOME','MVR_PTS','AGE','CLM_AMT')]
#top5.info()
top5.dropna().info()
top5.dropna().head()
#--------------------------------------------------------------------------------------------
# # Encode CLM_AMT and split Dataset
#C0 M0 O11
#CLM10 = top5dropna.loc[(top5dropna.CLM_AMT >= 0) , ['BLUEBOOK','TRAVTIME','INCOME','MVR_PTS','AGE','CLM_AMT']]
CLM10 = top5.dropna().loc[(top5.dropna().CLM_AMT >= 0) , ['BLUEBOOK','TRAVTIME','INCOME','MVR_PTS','AGE','CLM_AMT']]
CLM10.CLM_AMT[CLM10.CLM_AMT>0] = 1 
CLM10.head(10)

#The data of clients without claim.
CLM0 = CLM10.loc[(CLM10.CLM_AMT == 0) , ['BLUEBOOK','TRAVTIME','INCOME','MVR_PTS','AGE','CLM_AMT']]
CLM0.head()

#The data of clients with claim.
CLM1 = CLM10.loc[(CLM10.CLM_AMT > 0) , ['BLUEBOOK','TRAVTIME','INCOME','MVR_PTS','AGE','CLM_AMT']]
CLM1.head()

#The amount of clients with specific claim amount.
CLM1value = top5.dropna().loc[(top5.dropna().CLM_AMT>0), ['BLUEBOOK','TRAVTIME','INCOME','MVR_PTS','AGE','CLM_AMT']]
#Save the csv document for the following research
CLM1value.to_csv('CLM1value.csv')
CLM1value.info()
CLM1value.head(10)
#--------------------------------------------------------------------------------------------
#SVM
#--------------------------------------------------------------------------------------------
#C33 M4 O0
# encoding the features using get dummies
from sklearn.preprocessing import LabelEncoder
X_data = pd.get_dummies(CLM10.iloc[:,:-1])
X = X_data.values
# encoding the class with sklearn's LabelEncoder
Y_data = CLM10.values[:, -1]
class_le = LabelEncoder()
# fit and transform the class
y = class_le.fit_transform(Y_data)
# Spliting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# perform training
# creating the classifier object
clf = SVC(kernel="linear")
X_train
y_train
# performing training
clf.fit(X_train, y_train)
# make predictions
# predicton on test
y_pred = clf.predict(X_test)
# calculate metrics
print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

# function to display feature importance of the classifier
# here we will display top 20 features (top 10 max positive and negative coefficient values)
def coef_values(coef, names):
    imp = coef
    print(imp)
    imp,names = zip(*sorted(zip(imp.ravel(),names)))
    imp_pos_10 = imp[:]
    names_pos_10 = names[:]
    imp_neg_10 = imp[:]
    names_neg_10 = names[:]
    imp_top_20 = imp_neg_10+imp_pos_10
    names_top_20 =  names_neg_10+names_pos_10
    plt.barh(range(len(names_top_20)), imp_top_20, align='center')
    plt.yticks(range(len(names_top_20)), names_top_20)
    plt.show()
    
# get the column names
features_names = X_data.columns
# call the function
coef_values(clf.coef_, features_names)

