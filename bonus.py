import linear_regression as lr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#LOAD FILEDATA
try:
    file = np.genfromtxt("data.csv", delimiter=",")
except FileNotFoundError:
    print(f"Error: file 'data.csv' doesn't exist!")
    exit(1)

dataset = file[1:, :]
rows, columns = dataset.shape
X = dataset[:, 0:-1].reshape(rows, 1)
y = dataset[:, -1].reshape(rows, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

#CREATE AND TRAIN MODEL
my_model = lr.LinearRegression(normalize=True)
my_model.fit(X_train, y_train)

sk_model =  linear_model.LinearRegression()
sk_model.fit(X_train, y_train)

#PREDICTIONS
y_test = y_test.flatten()
print("----------OBSERVED VALUES----------\n", y_test)
my_predictions = my_model.predict(X_test) 
sk_predictions = sk_model.predict(X_test) 
print("----------MY MODEL PREDICTED VALUES----------\n", my_predictions)
print("----------SKLEARN MODEL PREDICTED VALUES----------\n", sk_predictions)

#METRICS
print("MY MODEL MEAN SQUARED ERROR:", my_model.mean_squared_error(y_test, my_predictions))
print("SKLEARN MODEL MEAN SQUARED ERROR:", mean_squared_error(y_test, sk_predictions))
print("MY MODEL R2 SCORE:", my_model.r2_score(y_test, my_predictions))
print("SKLEARN R2 SCORE:", r2_score(y_test, sk_predictions))
print("SKLEARN SCORE:", sk_model.score(X_test, y_test))

#PLOT
min_X = int(min(X)[0])
max_X = int(max(X)[0])

X_points = list(range(min_X, max_X, 50))
my_points = [my_model.predict(([[x]])) for x in range(min_X, max_X, 50)]
sk_points = [sk_model.predict([[x]])[0] for x in range(min_X, max_X, 50)]

#MY MODEL GRAPH
plt.subplot(2, 1, 1)
plt.scatter(X, y, color="black")
plt.ylabel("price")
plt.xlabel("km")
plt.title("My model graph")
plt.plot(X_points, my_points, color="red")

#SKLEARN GRAPH
plt.subplot(2, 1, 2)
plt.ylabel("price")
plt.xlabel("km")
plt.scatter(X, y, color="black")
plt.title("Sklearn model graph")
plt.plot(X_points, sk_points, color="blue")
plt.show()