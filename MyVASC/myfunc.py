import numpy as np
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,\
    homogeneity_score, completeness_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from skfuzzy import cmeans


# clustering
def clustering(points, k=2, method='cmeans'):

    if method == 'kmeans':
        # points: N_samples * N_features
        # k: number of clusters
        kmeans = KMeans(n_clusters=k, n_init=100).fit(points)
        if len(np.unique(kmeans.labels_)) > 1:
            si = silhouette_score(points, kmeans.labels_)
        else:
            si = 0
            print("Silhouette:" + str(si))
        return kmeans.labels_, si

    if method == 'spec':
        # points: N_samples * N_features
        # k: number of clusters
        spec = SpectralClustering(n_clusters=k, affinity='cosine').fit(points)
        si = silhouette_score(points, spec.labels_)
        print("Silhouette:" + str(si))
        return spec.labels_, si

    if method == 'cmeans':
        # points_in: N_features * N_samples
        # k: number of clusters
        points_in = points.transpose()
        center, u_matrix, u0_matrix, \
            distance, jm, p, fpc = cmeans(data=points_in, c=k, m=1.5, error=0.01, maxiter=500)
        cluster = np.argmax(u_matrix, axis=0)
        return cluster, center, u_matrix, fpc


# performance assessment
def performance_assessment(predict, ground_truth):
    nmi = normalized_mutual_info_score(ground_truth, predict)
    print("NMI: " + str(nmi))
    rand = adjusted_rand_score(ground_truth, predict)
    print("RAND: " + str(rand))
    homo = homogeneity_score(ground_truth, predict)
    print("HOMOGENEITY: " + str(homo))
    completeness = completeness_score(ground_truth, predict)
    print("COMPLETENESS: " + str(completeness))
    return {'NMI': nmi, 'RAND': rand, 'HOMOGENEITY': homo, 'COMPLETENESS': completeness}


# plot(2 dimensions)
def plot2dimensions(result, label, idmap, title):
    # pos_x = (result[:, 0] - result[:, 0].min()) / (result[:, 0].max() - result[:, 0].min())
    # pos_y = (result[:, 1] - result[:, 1].min()) / (result[:, 1].max() - result[:, 1].min())
    color = ['red', 'pink', 'yellow', 'orange', 'blue', 'green', 'gray', 'brown', 'purple', 'black']
    occured = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print("start visualization:")
    for i in tqdm.trange(len(label)):
        if occured[label[i]] == 0:
            occured[label[i]] = 1
            plt.plot(result[:, 0][i], result[:, 1][i], '.', color=color[label[i]], label=idmap[label[i]])
        else:
            plt.plot(result[:, 0][i], result[:, 1][i], '.', color=color[label[i]])
    plt.title(title)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
