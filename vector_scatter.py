import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import pickle

subject = 100206

def make_kmedoids(is_plot=False):
    with open(f"./data/train/{subject}/bvecs", "r") as f:
        vecs = np.empty((288, 3))
        for i in range(3):
            vecs[:, i] = list(map(float, f.readline().split()))
    f = open(f'./data/train/{subject}/bvecs', 'r')

    f = open(f'./data/train/{subject}/bvals', 'r')
    line = f.readline().split()
    vals = np.array(list(map(float, line)))

    indexes = np.where((vals >= 980) & (vals <= 1020))
    indexes2 = np.where((vals >= 1980) & (vals <= 2020))
    print(indexes, len(indexes[0]))

    vecs_1000 = vecs[indexes]
    vecs_2000 = vecs[indexes2]

    kmedoids_1000 = KMedoids(n_clusters=18, method='pam', random_state=0).fit(vecs_1000)
    kmedoids_2000 = KMedoids(n_clusters=18, method='pam', random_state=0).fit(vecs_2000)

    labels = kmedoids_1000.labels_
    labels2 = kmedoids_2000.labels_
    indices_1000 = kmedoids_1000.medoid_indices_
    indices_2000 = kmedoids_2000.medoid_indices_

    dif_indexes = []
    dif_indexes.extend(indexes[0][tuple([indices_1000])])
    dif_indexes.extend(indexes2[0][tuple([indices_2000])])

    with open(f"./data/diffusion_indexes_train", "wb") as f:
        pickle.dump(tuple(dif_indexes), f)

    if is_plot == True:
        ax = plt.subplot(projection='3d')
        unique_labels = set(labels)

        for k in unique_labels:
            class_member_mask = labels == k
            xy = vecs_1000[class_member_mask]
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2])

        ax.scatter(kmedoids_1000.cluster_centers_[:, 0], kmedoids_1000.cluster_centers_[:, 1], kmedoids_1000.cluster_centers_[:, 2], color='k')
        ax.scatter(vecs_1000[:, 0], vecs_1000[:, 1], vecs_1000[:, 2])

        plt.show()

if __name__== "__main__":
    make_kmedoids()
