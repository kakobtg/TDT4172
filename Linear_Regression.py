import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    
    def __init__(self, learningRate=0.001, epochs=1000):
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.losses = []
        
        self.learningRate = learningRate
    
    def fit(self, X, y):
        m = X.shape[0]
        self.weights = 0
        self.bias = 0
        
        for epoch in range(self.epochs):
            y_pred = (X * self.weights) + self.bias
            
            loss = np.mean((y_pred - y) ** 2)
            self.losses.append(loss)
            
            dw = (2/m) * np.dot(X.T, (y_pred - y))
            db = (2/m) * np.sum(y_pred - y)
            
            
            self.weights -= self.learningRate * dw
            self.bias -= self.learningRate * db
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss {loss}')
        
        
        
        
    def predict(self, X):
        print(f'Weights: {self.weights}. Bias: {self.bias}')
        return (X * self.weights) + self.bias

    