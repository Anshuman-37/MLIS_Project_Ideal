# This is the code for the SGD classifier 
## The following is the code for MLIS Course work 2021 - 2022
## Author 
## Anshuman Singh 
## Group members - Anshuman Singh, Alpaslan Erdag, Yixin Fan
## Group number - 4B

## Paper Reference for the code  
## Linear Classification with Logistic Regression
## Ryan P. Adams
## COS 324 – Elements of Machine Learning Princeton University
## https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/logistic-regression.pdf



# Code's Working and Its intution
# So we will consider each of the feature important and contributing for our final output 
# So each attribute should have some cofficient that explains about its impact on the final output 
# So we do this by assigning each attribute a specific weight and there is baise realated to it which will add up and give us our predicition
# So first we intialize our weights for all our attributes and bais to 0 
# Then we have to maximize the likelihood of our values refer equation 4 in the paper mentioned above 
# And to avoid the numeric difficulties we have take log of the equation mentioned above and it is implemented by the name loss_fun in the code
# Now we have the loss function and we have to minimize it and to do that we are using Stochastic Gradient descent and noisy version of Gradient Descent
# We callculate the gradient of w and bais in the terms Refer equation 25 in the paper mentioned 
# We now have to regularize the weights refer to equation 28 in the paper mentioned 
# And according to that we update our weight so that we will reach a value eventually where we have minimized our loss and that will give use best intercepts and bais value 
# Finally we can use the predict function that use these weight and bais and uses sigmoid function to get classify them
# If the value is less that 0.5 we classify it as 0 and if the value is 0.5 or greater than it we classify it as 1 



## Header Files 
import numpy as np
from tqdm import tqdm

def loss_fun(y_labels,y_predicticted):
    '''This function will return the log loss of the function'''
    # We are using log in the loss function so that it is easy to 
    # avoid numeric difficulties due to products of small numbers
    loss = -1 * (np.sum((y_labels * np.log10(y_predicticted))+ \
                      ((1-y_labels)*np.log10(1-y_predicticted))))/len(y_labels)
    return loss

def sig_function(z):
    '''This function will compute the sig_function of the input'''
    #Compute sig_function(z) and return
    return 1/(1+np.exp(-z))

def weights(dim):
    ''' In this function, we will initialize our weights and bias according to the number of parameters that we have'''
    # Here dimenstion refer to the number of the attributes in the data
    w = np.zeros(shape=len(dim))
    b = 0
    return w,b

## As we are using gradient descent so we reach the global minimum for the convex function 
## We will calculate the gradient of weights and bais

def dw(x,y,w,b,alpha_value,N):
    '''In this function, we will compute the gardient w.r.t to w '''
    # Calculcating the graindent of weighted vectors and then returning the result 
    return x * (y - sig_function(np.dot(w, x) + b)) - alpha_value/N*w

def db(x,y,w,b):
    '''In this function, we will compute gradient w.r.to b '''
    # Calculating the gradient of bais and then returning the result
    return y - sig_function(np.dot(w, x) + b)

# This function is to return the predicticted values numpy array a
def predict(w,b, X):
    '''This function will return the predicticted value in respect to the given data points '''
    N = len(X)
    predictict = []
    ## The loop to iterate over the range of the values 
    for i in range(N):
        z=np.dot(w,X[i])+b
        # Any thing with value of sig_function more than 0.5 will be classified as class label 1
        if sig_function(z) >=0.5: 
            predictict.append(1)
        # Anything wiht value of sifgmoid less than 0.5 will be classified as class label 0
        else:
            predictict.append(0)
    ## Returning the array contating the predicticted values 
    return np.array(predictict)


def train_classifier(x_train,y_train,x_test,y_test,epochs,alpha_value,t_rate):
    '''This function will update the weights and return back updated weights and baises'''
    
    # First we intialize the weights 
    weight,b = weights(x_train[0])
    # This is to ensure that the loss is not same for iterations 
    # We want to quit when we reach the critical point 
    same_loss_counter = 0
    ## The number of data points in x_train
    N = len(x_train)
    #Vectors to store our loss for testing and training data
    loss_train, loss_test = [],[]

    #To run code in batches
    # Mini batch approach 
    part_no = 0
    part_size = 25
    ctr = 0
    n = len(x_train)
    
    # Loop to traveres in epoche
    for i in tqdm(range(0,epochs)):

        # Loop to access data point in the respective part 
        for j in range(part_size):
            
            # Calculating gradient of w and adding it to the existing one    
            weight = weight + t_rate*dw(x_train[(j+part_no)%n], y_train[(j+part_no)%n],weight, b, alpha_value, len(x_train))
            
            #Calculating gradient of b and adding it to the existing one
            b = b + t_rate*db(x_train[(j+part_no)%n], y_train[(j+part_no)%n], weight, b)
        
        part_no = (part_no + part_size)%n # To updtae the new part

        #predicticting the traing data in comparison of the the xtrain
        y_predict_train = np.array([sig_function(np.dot(weight, x)+b) for x in x_train])
        
        #predicticting the test data in comaprison of the xtest
        y_predict_test = np.array([sig_function(np.dot(weight, x)+b) for x in x_test])

        #Calculating the loss on for training data
        loss = loss_fun(y_train,y_predict_train)
        loss_train.append(loss)
        
        #Calculatig the loss onfor testing data
        loss = loss_fun(y_test,y_predict_test)
        loss_test.append(loss)

        ## Printing values
        print('\n-- Epoch no(iteration no) ', i+1)
        print('W intercept: {}, B intercept: {}, Train loss: {:.5f}, Test loss: {:.5f}'.format(weight, b, loss_train[i], loss_test[i]))
    
    # Return the weights and baises with teh train and test loss back to the function call 
    return weight,b,loss_train,loss_test







