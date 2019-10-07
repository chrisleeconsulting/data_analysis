import csv

import inline as inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as seabornInstance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#%matplotlib inline

dataset = pd.read_csv("data/nyc_taxi.csv")
# print(dataset.shape)
# print(dataset.describe())
dataset.plot.scatter(x="TimeMin", y="PickupCount", color="#a9a9a9", title="Number of Taxi Pickups vs. Time of Day")


plt.figure(figsize=(15,10))
plt.tight_layout()
# seabornInstance.distplot(dataset["PickupCount"])


# code for linear regression
dataset.plot.scatter(x="TimeMin", y="PickupCount", color="#a9a9a9")
X = dataset["TimeMin"].values.reshape(-1, 1)
y = dataset["PickupCount"].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))

# plt.scatter(X_test, y_test, color="gray")
plt.plot(X_test, y_pred, color="#a82626")
plt.title("Number of Taxi Pickups vs. Time of Day")
plt.show()

# scatter plot with sin overlay
a = dataset.plot.scatter(x="TimeMin", y="PickupCount", color="#a9a9a9", title="Number of Taxi Pickups vs. Time of Day")

h = np.arange(0, 1450, 1)
k = 40 + 40 * np.sin(1.0/255*(h-800))
print(k)

a.plot(h, k, color="#a82626")

# code for linear sin regression
temp = max(dataset["TimeMin"])
for i in range(len(dataset["TimeMin"])):
    dataset["TimeMin"][i] = np.sin(dataset["TimeMin"][i]/temp * 2*np.pi)

dataset.plot.scatter(x="TimeMin", y="PickupCount", color="#a9a9a9")

X = dataset["TimeMin"].values.reshape(-1, 1)
y = dataset["PickupCount"].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))

# plt.scatter(X_test, y_test, color="gray")
plt.xlabel("$\sin(\dfrac{TimeMin\cdot 2\pi}{1440})$")
plt.plot(X_test, y_pred, color="#a82626")
plt.show()
