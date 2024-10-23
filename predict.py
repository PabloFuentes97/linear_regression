from linear_regression import LinearRegression
import numpy as np
import joblib

#LOAD MODEL FROM FILE
model = None
try:
    model = joblib.load("my_model")
except FileNotFoundError:
    model = LinearRegression()

#GET INPUT FROM USER
km_input = input("Introduce your km: ")
while not km_input.isdigit():
    km_input = input("Error: not a number. Introduce your km: ")

#PREDICT
pred = model.predict([[int(km_input)]])
print("Your predicted price:", pred)