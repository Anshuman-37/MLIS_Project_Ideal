# This is the code for the SGD classifier 
## The following is the code for MLIS Course work 2021 - 2022
## Author 
## Anshuman Singh 
## Group members - Anshuman Singh, Alpaslan Erdag, Yixin Fan
## Group number - 1A 

## Paper Reference for the code  
## Linear Classification with Logistic Regression
## Ryan P. Adams
## COS 324 – Elements of Machine Learning Princeton University


## Header Files 
import numpy as np
#import pandas as pd
from tqdm import tqdm

def weights(dim):
    ''' In this function, we will initialize our weights and bias according to the number of parameters that we have'''
    # Here dimenstion refer to the number of the attributes in the data
    w = np.zeros(shape=len(dim))
    b = 0
    return w,b

def sigmoid(z):
    '''This function will compute the sigmoid of the input'''
    #Compute sigmoid(z) and return
    return 1/(1+np.exp(-z))

def loss_fun(y_labels,y_predicticted):
    '''This function will return the log loss of the function'''
    # We are using log in the loss function so that it is easy to 
    # avoid numeric difficulties due to products of small numbers
    loss = -1 * (np.sum((y_labels * np.log10(y_predicticted))+ \
                      ((1-y_labels)*np.log10(1-y_predicticted))))/len(y_labels)
    return loss

## As we are using gradient descent so we reach the global minimum for the convex function 
## We will calculate the gradient of weights and bais

def dw(x,y,w,b,alpha,N):
    '''In this function, we will compute the gardient w.r.t to w '''
    # Calculcating the graindent of weighted vectors and then returning the result 
    return x * (y - sigmoid(np.dot(w, x) + b)) - alpha/N*w

def db(x,y,w,b):
    '''In this function, we will compute gradient w.r.to b '''
    # Calculating the gradient of bais and then returning the result
    return y - sigmoid(np.dot(w, x) + b)


def train_classifier(x_train,y_train,x_test,y_test,epochs,alpha,eta0,p):
    '''This function will apply the logistic regression'''
    # First we intialize the weights 
    w,b = weights(x_train[0])
    same_loss_counter = 0

    ## The number of data points in x_train
    N = len(x_train)

    #Vectors to store our loss for testing and training data
    train_loss , test_loss = [],[]

    #To run code in batches
    part_no = 0
    part_size = 25
    ctr = 0
    n = len(x_train)
    
    # Loop to traveres in epoche
    for i in tqdm(range(0,epochs)):

        # Loop to access data point in the respective part 
        for j in range(part_size):
            
            # Calculating gradient of w and adding it to the existing one    
            w = w + eta0*dw(x_train[(j+part_no)%n], y_train[(j+part_no)%n],w, b, alpha, len(x_train))
            
            #Calculating gradient of b and adding it to the existing one
            b = b + eta0*db(x_train[(j+part_no)%n], y_train[(j+part_no)%n], w, b)
        
        
        part_no = (part_no + part_size)%n # To updtae the new part

        #predicticting the traing data in comparison of the the xtrain
        y_predict_train = np.array([sigmoid(np.dot(w, x)+b) for x in x_train])
        
        #predicticting the test data in comaprison of the xtest
        y_predict_test = np.array([sigmoid(np.dot(w, x)+b) for x in x_test])

        #Calculating the loss on for training data
        loss = loss_fun(y_train,y_predict_train)
        train_loss.append(loss)
        
        #Calculatig the loss onfor testing data
        loss = loss_fun(y_test,y_predict_test)
        test_loss.append(loss)

        ## Printing values
        print('\n-- Epoch no(iteration no) ', i+1,'\n Train data set : ')
        #print('Actual values: ', y_train ,'\n predicticted Values : ', y_predict_train)
        #print('Test data set :') 
        #print('Actual values: ', y_test, '\npredicticated Values : ', y_predict_test)
        print('W intercept: {}, B intercept: {}, Train loss: {}, Test loss: {}'\
              .format(w, b, train_loss[i], test_loss[i]))
    return w,b,train_loss,test_loss


# This function is to return the predicticted values numpy array a

def predict(w,b, X):
    '''This function will return the predicticted value in respect to the given data points '''
    N = len(X)
    predictict = []
    ## The loop to iterate over the range of the values 
    for i in range(N):
        z=np.dot(w,X[i])+b
        # Any thing with value of sigmoid more than 0.5 will be classified as class label 1
        if sigmoid(z) > 0.5: 
            predictict.append(1)
        # Anything wiht value of sifgmoid less than 0.5 will be classified as class label 0
        else:
            predictict.append(0)
    ## Returning the array contating the predicticted values 
    return np.array(predictict)





