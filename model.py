import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

houses = pd.read_csv("./csv/USA_Housing.csv")

X = houses.drop(["Address", "Price"], axis=1)
y = houses["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)

lm = LinearRegression()

lm.fit(X_train, y_train)


pickle.dump(lm, open("model.pkl", "wb"))
