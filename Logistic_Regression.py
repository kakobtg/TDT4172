import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    
    def __init__(self, learningRate=0.1, epochs=1000):
        self.weights = None
        self.bias = None
        
        self.learningRate = learningRate
        self.epochs = epochs
        self.losses = []
        self.trainAccuracy = []
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def computeLogLoss(self, y, y_pred):
        m = len(y)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -(1/m) * sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        
    def computeGradients(self, X, y, y_pred):
        m = X.shape[0]
        dw = np.matmul(X.T, (y_pred - y)) / m
        db = np.sum(y_pred - y) / m
        
        return dw, db
    
    def updateParameters(self, gradW, gradB):
        self.weights -= self.learningRate * gradW
        self.bias -= self.learningRate * gradB
        
    def predictProba(self, X):
        linearModel = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linearModel)
        return y_pred
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """

        nFeatures = X.shape[1]
        self.weights = np.zeros(nFeatures)
        self.bias = 0
        
        for epoch in range(self.epochs):
            linearModel = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linearModel)
            
            loss = self.computeLogLoss(y, y_pred)
            self.losses.append(loss)
            
            gradW, gradB = self.computeGradients(X, y, y_pred)
            
            self.updateParameters(gradW, gradB)
            
            #Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: LogLoss {self.losses[epoch]}')
        
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        
        linearModel = np.matmul(X, self.weights) + self.bias
        y_pred = self.sigmoid(linearModel)
        predictions = [1 if i > 0.5 else 0 for i in y_pred]
        return predictions
        
