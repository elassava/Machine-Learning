import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

x, y = make_classification(n_classes=2, random_state = 42) # load binary dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42) #split the dataset into train and test

x_train = x_train.T #reshape the dataset 
y_train = y_train.reshape(1, x_train.shape[1])

x_test = x_test.T
y_test = y_test.reshape(1, x_test.shape[1])

#define epochs and required mathematical functions
epochs = 1000

def sigmoid(x): #sigmoid function
    return 1/(1 + np.exp(-x)) 

def dot_product(W, x_train): #function to calculate dot product
    result = 0
    for i in range(len(W)):
        result += W[i] * x_train[i]
    return result

#logistic regression model created using numpy functions (vectorized) 
def logistic_regression_vectorized(x_train, y_train, learning_rate, epochs):
    start_time = time.time()
    observations = x_train.shape[1] # number of observations
    features = x_train.shape[0] #number of features
    
    cost_list = []
    
    W = np.zeros((features,1)) #initialize weight and bias as 0
    b = 0
    
    for _ in range(epochs):
        linear_prediction = np.dot(W.T, x_train) + b
        prediction = sigmoid(linear_prediction)  
        
        cost =  -(1/observations)*np.sum(y_train*np.log(prediction) + (1-y_train)*np.log(1-prediction)) #log loss funciton
        
        dW = (1/observations)*np.dot(prediction-y_train, x_train.T) #derivative of weight and bias
        dB = (1/observations)*np.sum(prediction - y_train)
        
        W = W - learning_rate*dW.T #apply gradient descent
        b = b - learning_rate*dB
        
        cost_list.append(cost) #will use to plot the cost convergence
    
    end_time = time.time()
    time_taken = end_time - start_time
    return W, b, cost_list, time_taken

#logistic regression model using loops instead of numpy functions
def logistic_regression_loop(x_train, y_train, learning_rate, epochs):
    start_time = time.time()
    observations = len(x_train[0]) #no. of observations
    features = len(x_train) #no. of features
    
    cost_list = []

    W = [0] * features #initialize weight and bias
    b = 0
    
    for _ in range(epochs):
        cost = 0 #initialize variables cost and derivatives
        dW = [0] * features
        dB = 0
        
        for i in range(observations):
            linear_prediction = dot_product(W, x_train[:, i]) + b 
            prediction = sigmoid(linear_prediction)
            
            cost += -(y_train[0][i] * math.log(prediction) + (1 - y_train[0][i]) * math.log(1 - prediction))
            
            for j in range(features):
                dW[j] += (prediction - y_train[0][i]) * x_train[j, i]
            
            dB += prediction - y_train[0][i]
        
        cost /= observations
        cost_list.append(cost)  # will use to plot the cost convergence
        
        for j in range(features):
            dW[j] /= observations
            W[j] -= learning_rate * dW[j]
        
        b -= learning_rate * (dB / observations)
    
    end_time = time.time()
    time_taken = end_time - start_time
    return W, b, cost_list, time_taken


def predict(W, b, x_test): #predict function to test model
    W = np.array(W)
    linear_prediction = np.dot(W.T, x_test) + b
    prediction = sigmoid(linear_prediction)
    return np.where(prediction <= 0.5, 0, 1).flatten()


#train both models
weight_vec, bias_vec, cost_list_vec, time_taken_vec = logistic_regression_vectorized(x_train,y_train, 0.0075, epochs)
weight_loop, bias_loop, cost_list_loop, time_taken_loop = logistic_regression_loop(x_train,y_train, 0.0075, epochs)


#Show the time taken by both models
print(f"Time taken by Vectorization Style: {time_taken_vec} seconds")
print(f"Time taken by Loop Style: {time_taken_loop} seconds\n")


#draw cost convergence
plt.plot(range(epochs), cost_list_loop, label="Loop Style")
plt.plot(range(epochs), cost_list_vec, label='Vectorization Style', linestyle="--")
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Convergence')
plt.legend()
plt.show()


#apply model on testing data
predictions_vec = predict(weight_vec, bias_vec, x_test)
predictions_loop = predict(weight_loop, bias_loop, x_test)


#show confusion matrix for vector style
print("Confusion Matrix for Vectorization Style:")
confusion_matrix_vectorized = confusion_matrix(y_test.flatten(), predictions_vec)
print(pd.DataFrame(confusion_matrix_vectorized, columns=["Pred True", "Pred False"], index=["Value True", "Value False"]))
print("\n")

#show confusion matrix for loop style
print("Confusion Matrix for Loop Style:")
confusion_matrix_loop = confusion_matrix(y_test.flatten(), predictions_loop)
print(pd.DataFrame(confusion_matrix_loop, columns=["Pred True", "Pred False"], index=["Value True", "Value False"]))


