# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data input
dPoint = {
    'x':[23,47,39,16,75,98,16,50],
    'y':[36,13,53,29,67,34,19,71]
}
df = pd.DataFrame(dPoint)

k = 3

# randomly initialize centroids
dCentroid = {
    i+1 : [np.random.randint(0, 100), np.random.randint(0, 100)]
    for i in range(k)
}
centroids = pd.DataFrame (dCentroid)

# set plot
fig = plt.figure(figsize=(5,5))
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.scatter(df['x'], df['y'], color='k') # set points

colmap = {1:'r', 2:'g', 3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], marker='x', color=colmap[i]) # set centroids
plt.show()

# update points cluster
def assignment(df, centroids):
    for i in centroids.keys():
        df['dis_from_{}'.format(i)] = ( # compute Euclidean distance among each centroid and all points
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2 + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_dis_cols = ['dis_from_{}'.format(i) for i in centroids.keys()] # save numbers of dis_cols with prefix
        # set 'closeset' col values with dis_cols numbers
    df['closest'] = df.loc[:, centroid_dis_cols].idxmin(axis=1) # fill with chars+nums
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('dis_from_'))) # remove chars

    df['color'] = df['closest'].map(lambda x: colmap[x]) # add 'color' column
    return df

df = assignment(df, centroids)

# update centroids
def update(k):
    for i in centroids.keys():
        # compute mean distance within each cluster
        centroids[i][0]= np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)

# finish clustering
while True:
    ole_closest = df['closest'].copy()
    # iteration
    centroids = update(centroids)
    df = assignment(df, centroids)
    if ole_closest.equals(df['closest']):
        break

# plot result
fig = plt.figure(figsize=(5,5))
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.scatter(df['x'], df['y'], color=df['color']) # inherit color

for i in centroids.keys():
    plt.scatter(*centroids[i], marker='x', color=colmap[i])
plt.show()