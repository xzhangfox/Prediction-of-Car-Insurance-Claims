#Machine Learning I - Final Project Codes
#Copy:137 Modified:22 Own:129 Result = 43.233%
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

#--------------------------------------------------------------------------------------------
#MLP Copy:77 Modified:15 Own:53
#--------------------------------------------------------------------------------------------
#Data Preparation
#import data
CLM1value=pd.read_csv("Clm1value.csv",index_col=[0])
CLM1value.info()

#Dataframe Standerdization
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
dfTest = CLM1value
min_max_scaler = preprocessing.MinMaxScaler()
def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(dfTest[col])),columns=[col])
    return df
#Standerdize 'BLUEBOOK','TRAVTIME','INCOME','MVR_PTS'.
clmstd = scaleColumns(CLM1value,list(CLM1value))
clmstd.head()
X = clmstd.dropna().iloc[:,:-1]
Y= clmstd.dropna().iloc[:, -1]

#indexing features and target
x1 = CLM1value.dropna().iloc[:,0:5]
y1 = CLM1value.dropna().iloc[:,[-1]]
print (x1.head())
print (y1.head())

from sklearn.model_selection import train_test_split
import random
random.seed(1234) #set seed for repeatable data
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3)
#x_test1, x_val1, y_test1, y_val1 = train_test_split(x_test1, y_test1, test_size=0.5)
print(len(y_test.values))

#--------------------------------------------------------------------------------------------
#MLP Model Settle
from numpy import exp, array, random, dot
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment
    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2
    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (3 neurons, each with 5 inputs):")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (1 neuron, with 3 inputs):")
        print(self.layer2.synaptic_weights)
if __name__ == "__main__":
    #Seed the random number generator
    random.seed(1)
    # Create layer 1 (3 neurons, each with 5 inputs)
    layer1 = NeuronLayer(3, 5)
    # Create layer 2 (a single neuron with 3 inputs)
    layer2 = NeuronLayer(1, 3)
    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)
    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()
    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = x_train.values
    training_set_outputs = y_train.values
    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 5)
    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()
    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [0.5482, 0.1739, 0.4240, 0., 0.7333] -> ?: ")
    hidden_state, output = neural_network.think(array([0.5482, 0.1739, 0.4240, 0., 0.7333]))
    print(output)

hidden_state1,predictions = neural_network.think(x_test.values)
neural_network.think(x_test.values)
hidden_state1
predictions
mse = metrics.mean_squared_error(y_test, predictions)
print ('MSE of Claim Amount: {:.3f}'.format(mse))

# # MSE Plots
def MSE(iterations,NNeurons):
    if __name__ == "__main__":
        #Seed the random number generator
        random.seed(1)
        # Create layer 1 (4 neurons, each with 3 inputs)
        layer1 = NeuronLayer(NNeurons, 5)
        # Create layer 2 (a single neuron with 4 inputs)
        layer2 = NeuronLayer(1, NNeurons)
        # Combine the layers to create a neural network
        neural_network = NeuralNetwork(layer1, layer2)
        #print("Stage 1) Random starting synaptic weights: ")
        neural_network.print_weights()
        # The training set. We have 7 examples, each consisting of 3 input values
        # and 1 output value.
        training_set_inputs = x_train.values
        training_set_outputs = y_train.values
        neural_network.train(training_set_inputs, training_set_outputs, iterations)
        neural_network.print_weights()
    hidden_state1,predictions = neural_network.think(x_test.values)
    mse = metrics.mean_squared_error(y_test, predictions)
    return mse
#--------------------------------------------------------------------------------------------
#MSE&Number of Neuron Plot with Fixed Epoch
def NNPLOT(min_,max_,steps,iterations):
    N=[]
    I=[]
    for i in range(min_,max_,steps):
        N.append(MSE(iterations,i))
        I.append(i)
    plt.figure(figsize=(12,8))
    plt.scatter(I,N)
    plt.plot(I,N)
    plt.title("Relationship between MES and Number of Neurons", fontsize=16)
    plt.xlabel("Number of Neurons", fontsize=14)
    plt.ylabel("MSE", fontsize=12)
    plt.xticks(rotation=50, fontsize=12)
    #Save the plots to folder
    plt.savefig("NNPLOT.png", bbox_inches='tight')
    plt.show()
NNPLOT(1, 10, 1, 5)
#--------------------------------------------------------------------------------------------
# MSE&NEpoch Plot with Fixed Number of Neuron

def EpochPLOT(min_,max_,steps,NNeurons):
    M=[]
    I=[]
    for i in range(min_,max_,steps):
        M.append(MSE(i,NNeurons))
        I.append(i)
    plt.figure(figsize=(12,8))
    plt.scatter(I,M)
    plt.plot(I,M)
    plt.title("Relationship between MES and Epoch", fontsize=16)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.ylabel("MSE", fontsize=12)
    plt.xticks(rotation=50, fontsize=12)
    #Save the plots to folder
    plt.savefig("EpochPLOT.png", bbox_inches='tight')
    plt.show()
EpochPLOT(1,30,1,3)
#--------------------------------------------------------------------------------------------
#Inverse Standardization
origin_data = min_max_scaler.inverse_transform(predictions)
print(origin_data)


