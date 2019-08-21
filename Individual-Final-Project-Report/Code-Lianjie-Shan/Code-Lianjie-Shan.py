#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#--------------------------------------------------------------------------------------------
#MLP Copy:77 Modified:15 Own:53 result = 47.69
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

