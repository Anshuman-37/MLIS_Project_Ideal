## Header Files
import numpy as np


# ## SVM class
# class SVM:
#   def __init__(self, kernel=None, C=10.0, max_iter=50, gamma=1):
    
#     self.kernel = {'rbf'   : lambda x,y: np.exp(-gamma*np.sum((y - x[:,np.newaxis])**2, axis=-1)),
#                    'linear': lambda x,y: np.dot(x, y.T)}[kernel]
#     self.C = C
#     self.max_iter = max_iter

#   def restrict_to_square(self, t, v0, u):
#     t = (np.clip(v0 + t*u, 0, self.C) - v0)[1]/u[1]
#     return (np.clip(v0 + t*u, 0, self.C) - v0)[0]/u[0]

#   def fit(self, X, y):
#     self.X = X
#     self.y = y * 2 - 1
#     self.alphas = np.zeros_like(self.y, dtype=float)  # ??Weights used for updating initially we use 0 for all entries
#     self.K = self.kernel(self.X, self.X) * self.y[:,np.newaxis] * self.y
    
#     for _ in range(self.max_iter):
#       for idxM in range(len(self.alphas)):
#         idxL = np.random.randint(0, len(self.alphas))
#         Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
#         v0 = self.alphas[[idxM, idxL]]
#         k0 = 1 - np.sum(self.alphas * self.K[[idxM, idxL]], axis=1)
#         u = np.array([-self.y[idxL], self.y[idxM]])
#         t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
#         self.alphas[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)
    
#     idx, = np.nonzero(self.alphas > 1E-15)
#     self.b = np.mean((1.0 - np.sum(self.K[idx] * self.alphas, axis=1)) * self.y[idx])
  
#   def decision_function(self, X):
#     return np.sum(self.kernel(X, self.X) * self.y * self.alphas, axis=1) + self.b

#   def predict(self, X):
#     return (np.sign(self.decision_function(X)) + 1) // 2


# def gaussian_kernel(x, z, sigma):
#     n = x.shape[0]
#     m = z.shape[0]
#     xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
#     zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))     
#     return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / (2 * sigma ** 2))

class OurSVM:
    def __init__(self, kernel=None, C=10.0,bias=1, gamma=1,sigma=0.01):
        self.kernel = {'rbf'   : lambda x,y: np.exp(-gamma*np.sum((y - x[:,np.newaxis])**2, axis=-1)),
                     'gauss' : lambda x,y: np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2)),
                     'linear': lambda x,y: np.dot(x, y.T)}[kernel]
      
        self.C = C
        self.bias= bias

    # X for data y for labels
    def fit(self, X, y):
        self.X = X
        self.y = y*2 -1
        m , l = X.shape
        self.alphas = np.zeros((m))
        self.K = self.kernel(self.X, self.X).T * self.y[:,np.newaxis] * self.y

        for i in range(10):
            for alphaX in range(0,m):

              alphaY = np.random.randint(0,m)

              Q = self.K[[[alphaX, alphaX], [alphaY, alphaY]], [[alphaX, alphaY], [alphaX, alphaY]]]

              v0 = self.alphas[[alphaX, alphaY]]

              k0 = 1 - np.sum(self.alphas * self.K[[alphaX, alphaY]], axis=1)

              u = np.array([-self.y[alphaY], self.y[alphaX]])

              t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-3)

            self.alphas[[alphaX, alphaY]] = v0 + u * self.limitation(t_max, v0, u)
            idx, = np.nonzero(self.alphas > 1E-3)
            self.b = np.mean((1.0 - np.sum(self.K[idx] * self.alphas, axis=1)) * self.y[idx])
            support_vectors = X[idx, :]
       
    def objective(self, X):
        return np.sum(self.alphas * self.y * self.kernel(X, self.X), axis=1) + self.b

    def predict(self, X):
        return (np.sign(self.objective(X)) + self.bias) // 2

    def limitation(self, t, v0, u):   # this function purpuse is to provide this  0 <alpha< C
        t = (np.clip(v0 + t*u, 0, self.C) - v0)[1]/u[1]
        return (np.clip(v0 + t*u, 0, self.C) - v0)[0]/u[0]


    