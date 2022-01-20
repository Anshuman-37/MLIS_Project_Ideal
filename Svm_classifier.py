## Header Files
import numpy as np


class OurSVM:
    def __init__(self, kernel=None, C=10.0,bias=1, gamma=1,sigma=0.01):
        # Kernel functions lambda operations to update kernel trick
        self.kernel = {'rbf'   : lambda x,y: np.exp(-gamma*np.sum((y - x[:,np.newaxis])**2, axis=-1)),
                     'linear': lambda x,y: np.dot(x, y.T)
                    }[kernel]
      
        self.C = C
        self.bias= bias

    # X for data y for labels
    def fit(self, X, y):
        # assigning the given features and the prediction labels
        self.X = X
        self.y = y*2 -1
        # Dimenensions of the given data
        # to reach all samples we`ll use these in for loop
        m , l = X.shape
        
        self.alphas = np.zeros((m))
        # Gram matrix/kernel
        self.K = self.kernel(self.X, self.X).T * self.y[:,np.newaxis] * self.y

        # It(10) defines the number of epochs
        for i in range(10):
            # starting with initializing of alphaA and alphaB
            for alphaA in range(0,m):
                #randomm alpha values selected
              alphaB = np.random.randint(0,m)
              
              #the following terms are obtained by breaking down the SMO objective function 
              #alphaKernel,alphaVector,k_value,labelVector,opt
              
              # This variable is is a kernel matrix with initial alpha values
              alphaKernel = self.K[[[alphaA, alphaA], [alphaB, alphaB]], [[alphaA, alphaB], [alphaA, alphaB]]]
              #initial alpha vector
              alphaVector = self.alphas[[alphaA, alphaB]]

              k_value = 1 - np.sum(self.alphas * self.K[[alphaA, alphaB]], axis=1)
              #labels corresponding to the initial alphas
              labelVector = np.array([-self.y[alphaB], self.y[alphaA]])
              
              #derivative of quadratic function wrt t ---- gives us optimun new alpha values
              opt = np.dot(k_value, labelVector) / (np.dot(np.dot(alphaKernel, labelVector), labelVector) + 1E-3)

           #updated alphas without clip
            self.alphas[[alphaA, alphaB]] = alphaVector + labelVector * self.clip(opt, alphaVector, labelVector)
            idx, = np.nonzero(self.alphas > 1E-3)
            #updated b values
            self.b = np.mean((1.0 - np.sum(self.K[idx] * self.alphas, axis=1)) * self.y[idx])
            support_vectors = X[idx, :]

    # Objective Function : ∑ [y*alpha* (kernelX*T*x)] - b
    def objective(self, X):
        return np.sum(self.alphas * self.y * self.kernel(X, self.X), axis=1) + self.b



    #Predict function : Prediction is wT*x-b.--- w= y*alpha* kernelX
    # May treat -b as one more coefficient in w, may take sign of this value
    def predict(self, X):
        return (np.sign(self.objective(X)) + self.bias) // 2
    
    
# Limitation : This function purpuse is to provide this  0 <alpha< C
#For soft margin SVM we have to “clip” size of any change because 
#of addional constraint that every α must be between 0 and C
    def clip(self, t, alphaVector, labelVector):   
        t = (np.clip(alphaVector + t*labelVector, 0, self.C) - alphaVector)[1]/labelVector[1]
        return (np.clip(alphaVector + t*labelVector, 0, self.C) - alphaVector)[0]/labelVector[0]


    

    