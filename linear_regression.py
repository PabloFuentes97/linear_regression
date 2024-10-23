import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import joblib

class LinearRegression:
    def __init__(self, learningRate=0.01, iter_n=1000, normalize=False):
        self.iter_n = iter_n
        self.weights = 0
        self.bias = 0
        self.normalize = normalize
        if self.normalize:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        self.learningRate = learningRate
    
    def predict(self, X):
        if self.normalize:
            X = self.scaler_X.transform(X)
            return (np.dot(X, self.weights) + self.bias) * self.scaler_y.scale_ + self.scaler_y.mean_
        else:
            return (np.dot(X, self.weights) + self.bias)
     
    def fit(self, X, y):
        samples_n, features_n = X.shape
        self.weights = np.zeros(features_n)
        self.bias = 0

        if self.normalize:
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y).flatten()
        
        for _ in range(self.iter_n):
            y_pred = np.dot(X, self.weights) + self.bias
             
            dw = (1 / samples_n) * np.dot(X.T, (y_pred - y))
            db = (1 / samples_n) * np.sum(y_pred - y)
        
            self.weights = self.weights - (self.learningRate * dw)
            self.bias = self.bias - (self.learningRate * db)
        
    def mean_squared_error(self, y_test, predictions):
        return np.mean((y_test - predictions) ** 2)
    
    def r2_score(self, y_test, predictions):
        return 1 - (self.mean_squared_error(y_test, predictions) / (np.mean((y_test - np.mean(y_test)) ** 2)))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(1)
    filename = sys.argv[1]

    #LOAD FILEDATA
    try:
        file = np.genfromtxt(filename, delimiter=",")
    except FileNotFoundError:
        print(f"Error: file '{filename}' doesn't exist!")
        exit(1)

    #STRUCTURE DATASET
    dataset = file[1:, :]
    rows, columns = dataset.shape
    X = dataset[:, 0:-1].reshape(rows, 1)
    y = dataset[:, -1].reshape(rows, 1)

    #CREATE AND TRAIN MODEL
    model = LinearRegression(normalize=True)
    model.fit(X, y)

    #SAVE MODEL TO FILE
    joblib.dump(model, "my_model")