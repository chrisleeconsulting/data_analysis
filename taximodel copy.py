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
"""
data = []
timemin = []
pickupcount = []
with open("data/nyc_taxi.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    firstLine = True
    for row in readCSV:
        if (firstLine):
            firstLine = False
            continue;
        timemin.append(int(float(row[0])))
        pickupcount.append(int(float(row[1])))
    data = [timemin, pickupcount]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(timemin[1:], pickupcount[1:])
plt.xlabel("Time")
plt.ylabel("Number of Pick-ups")
"""

dataset = pd.read_csv("data/nyc_taxi.csv")
# print(dataset.shape)
# print(dataset.describe())
a = dataset.plot(x="TimeMin", y="PickupCount", style="o")

h = np.arange(0, 1450, 1)
k = 40 + 40 * np.sin(1.0/255*(h-800))
print(k)

a.plot(h, k)

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset["PickupCount"])

temp = max(dataset["TimeMin"])

for i in range(len(dataset["TimeMin"])):
    dataset["TimeMin"][i] = np.sin(dataset["TimeMin"][i]/temp * 2*np.pi)

dataset.plot(x="TimeMin", y="PickupCount", style="o")

X = dataset["TimeMin"].values.reshape(-1, 1)
y = dataset["PickupCount"].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

"""
y_pred = []
for i in range(len(dataset["TimeMin"])):
    y_pred.append([40*np.sin(255*(dataset["TimeMin"][i]-800))])
"""

# 40 sin 255(x-800)

print(r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color="gray")
plt.plot(X_test, y_pred, color="Red", linewidth=2)
plt.show()

plt.show()
