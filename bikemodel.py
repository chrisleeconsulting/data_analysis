import numpy as np
import pandas as pd
import random

# dist = np.linalg.norm(a-b)
# dteday,season,hour,holiday,weekday,workingday,weather,temp,atemp,hum,windspeed,casual,registered

# https://en.wikipedia.org/wiki/K-means_clustering


def initiate(data, k, numpoints):
    """Set the initial means for the k-means clustering algorithm
    based on k-means++. Returns a float array of the coordinates
    of the centers."""

    centers = [data[0]]
    iscenter = [True]
    for i in range(1, numpoints):
        iscenter.append(False)

    for i in range(1, k): # get all k clusters
        distances = [[], []]
        for j in range(1, numpoints): # for each data point x = j
            if iscenter[j]:
                continue
            else:
                distances[0].append(j)
                tempmin = np.linalg.norm(data[j][:-2] - data[0][:-2])
                for k in range(1, numpoints): # compute D(x), distance from x to nearest center
                    if iscenter[k]:
                        tempmin = min(tempmin, np.linalg.norm(data[j][:-2] - data[k][:-2]))
                distances[1].append(tempmin)
        probmaxes = [distances[1][0] ** 2]
        for j in range(1, len(distances[1])):
            probmaxes.append(probmaxes[j - 1] + distances[1][j] ** 2)
        randindex = random.uniform(0, probmaxes[len(probmaxes)-1])
        nextcenter = len(probmaxes)-1
        while nextcenter > 0 and randindex < probmaxes[nextcenter-1]:
            nextcenter -= 1
        centers.append(data[distances[0][nextcenter]])
        iscenter[distances[0][nextcenter]] = True

    return centers


def assignment(data, centers, numpoints):
    clusters = []
    for i in range(len(centers)):
        clusters.append([])
    for i in range(numpoints):
        closest = 0
        for j in range(1, len(centers)):
            if np.linalg.norm(data[i][:-2]-centers[j][:-2]) < np.linalg.norm(data[i][:-2]-centers[closest][:-2]):
                closest = j
        clusters[closest].append(data[i])
    return clusters


def update(clusters, centers):
    for i in range(len(centers)):
        out = clusters[i][0]
        for j in range(1, len(clusters[i])):
            out += clusters[i][j]
        out /= len(clusters[i])
        centers[i] = out
    return centers


def closestcenter(point, centers): # returns closest cluster center index
    closest = 0
    for i in range(1, len(centers)):
        if np.linalg.norm(point[:-2]-centers[i][:-2]) < np.linalg.norm(point[:-2]-centers[closest][:-2]):
            closest = i
    return closest


dataset = pd.read_csv("data/BSS_hour_raw copy.csv").drop("dteday", axis=1).copy()

numpoints = len(dataset)

groupmaxes = []

for row, group in dataset.iteritems(): #normalize the groups to be from 0 to 1
    groupmaxes.append(max(group))

tdataset = dataset.transpose()

for i in range(numpoints):
    for j in range(len(tdataset)):
        if groupmaxes[j] != 0:
            tdataset[i][j] /= groupmaxes[j]


centers = initiate(tdataset, 10, numpoints)

notdone = True
while (notdone):
    temp = centers.copy()
    # print(temp)
    clusters = assignment(tdataset, centers, numpoints)
    centers = update(clusters, centers)
    print("hot dog")
    # print(temp)
    if np.linalg.norm(np.asarray(temp)-np.asarray(centers)) == 0:
        notdone = False

casualclusters = []
registeredclusters = []

for cluster in clusters:
    tempsumcasual = 0
    tempsumregistered = 0
    for point in cluster:
        tempsumcasual += point[-2]
        tempsumregistered += point[-1]
    casualclusters.append(tempsumcasual / len(cluster) * groupmaxes[-2])
    registeredclusters.append(tempsumregistered / len(cluster) * groupmaxes[-1])

"""

TESTING

"""


db = pd.read_csv("data/BSS_hour_raw copy 2.csv").drop("dteday", axis=1).copy()
tdb = db.transpose()
x = closestcenter(tdb[1], centers)
print(tdb[1][-2], "\t", tdb[1][-1])
print(casualclusters[x], "\t", registeredclusters[x])
