import numpy as np
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,\
    homogeneity_score, completeness_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist


def cmeans_calculate(data, u_old, c, m):
    """
    Single step in fuzzy c-means clustering algorithm.
    Parameters inherited from cmeans()
    Modified from Fuzzy Logic With Engineering Applications, pages 382-383, equations 11.28 - 11.35.
    """
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)
    um = u_old ** m

    # Calculate cluster centers
    center = um.dot(data) / np.ones((data.shape[1], 1)).dot(np.atleast_2d(um.sum(axis=1))).T

    d = cdist(data, center).T
    d = np.fmax(d, np.finfo(np.float64).eps)

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return center, u, d


def fuzzy_cmeans_clustering(data, c, m, error, max_iter=1000, init=None):
    """
    Fuzzy c-means clustering algorithm

    Parameters
    ----------
    data : 2d array, size (N, S), Data to be clustered,
        in which N is the number of data sets; S is the number of features within each sample vector.
    c : int, Desired number of clusters or classes.
    m : float, U_new = u_old ** m.
    error : float, Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    max_iter : int, Maximum number of iterations allowed.
    init : 2d array, size (N, S), Initial fuzzy c-partitioned matrix.
        If none provided, algorithm is randomly initialized.

    Returns
    -------
    center : 2d array, size (C, S), Cluster centers of the `c` requested clusters.
    u : 2d array, (C, N), Final fuzzy c-partitioned matrix.
    d : 2d array, (C, N), Final Euclidian distance matrix.
    fpc : float, Final fuzzy partition coefficient, for clustering, 0 is terrible, 1 is perfect.
    error: float, Final error of fuzzy c-partitioned matrix.
    """

    # initial
    if init is None:
        n = data.shape[0]
        u0 = np.random.rand(c, n)
        u0 /= np.ones((c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u_current = np.fmax(u0, np.finfo(np.float64).eps)
    u_last = u_current.copy()
    center, distance = None, None
    # main loop
    for i in range(max_iter):
        [center, u_current, distance] = cmeans_calculate(data, u_last, c, m)
        u_last = u_current.copy()
        if np.linalg.norm(u_current - u_last) < error:
            break

    # error calculation
    error = np.linalg.norm(u_current - u_last)

    # fuzzy partition coefficient
    fpc = np.trace(u_current.dot(u_current.T)) / float(u_current.shape[1])

    return center, u_current, distance, fpc, error


# clustering
def clustering(points, k=2, method='cmeans'):
    # points: N_samples * N_features
    # k: number of clusters
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
        # skfuzzy.cmeans()
        center, u_matrix, distance, fpc, error = \
            fuzzy_cmeans_clustering(data=points, c=k, m=1.5, error=0.01, max_iter=500)
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
