import numpy as np
import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns
# from pandas import DataFrame

# sklearn
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,\
    homogeneity_score, completeness_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope

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


# clustering


def clustering(points, k=2, name='kmeans'):
    '''
    points: N_samples * N_features
    k: number of clusters
    '''
    if name == 'kmeans':
        kmeans = KMeans(n_clusters=k, n_init=100).fit(points)
        ## print within_variance
        # cluster_distance = kmeans.transform( points )
        # within_variance = sum( np.min(cluster_distance,axis=1) ) / float( points.shape[0] )
        # print("AvgWithinSS:"+str(within_variance))
        if len(np.unique(kmeans.labels_)) > 1:
            si = silhouette_score(points, kmeans.labels_)
            # print("Silhouette:"+str(si))
        else:
            si = 0
            print("Silhouette:" + str(si))
        return kmeans.labels_, si

    if name == 'spec':
        spec = SpectralClustering(n_clusters=k, affinity='cosine').fit(points)
        si = silhouette_score(points, spec.labels_)
        print("Silhouette:" + str(si))
        return spec.labels_, si


# outliers detection


def outliers_detection(expr):
    x = PCA(n_components=2).fit_transform(expr)
    ee = EllipticEnvelope()
    ee.fit(x)
    outliers = ee.predict(x)
    return outliers


def plot_2dimensions(result, label, title):
    # pos_x = (result[:, 0] - result[:, 0].min()) / (result[:, 0].max() - result[:, 0].min())
    # pos_y = (result[:, 1] - result[:, 1].min()) / (result[:, 1].max() - result[:, 1].min())
    color = ['red', 'pink', 'yellow', 'orange', 'blue', 'green', 'gray', 'brown', 'purple', 'black']
    occured = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print("start visualization:")
    for i in tqdm.trange(len(label)):
        if occured[label[i]] == 0:
            occured[label[i]] = 1
            plt.plot(result[:, 0][i], result[:, 1][i], '.', color=color[label[i]], label=str(label[i]))
        else:
            plt.plot(result[:, 0][i], result[:, 1][i], '.', color=color[label[i]])
    plt.title(title)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
