import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,homogeneity_score,completeness_score,silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

# performance assessment
def performance_assessment(predict, ground_truth):
    NMI = normalized_mutual_info_score(ground_truth, predict)
    print("NMI: " + str(NMI))
    RAND = adjusted_rand_score(ground_truth, predict)
    print("RAND: " + str(RAND))
    HOMO = homogeneity_score(ground_truth, predict)
    print("HOMOGENEITY: " + str(HOMO))
    COMPLETENESS = completeness_score(ground_truth, predict)
    print("COMPLETENESS: " + str(COMPLETENESS))
    return {'NMI':NMI,'RAND':RAND,'HOMOGENEITY':HOMO,'COMPLETENESS':COMPLETENESS}

# outliers detection
def outliers_detection(expr):
    x = PCA(n_components=2).fit_transform(expr)
    ee = EllipticEnvelope()
    ee.fit(x)
    outliers = ee.predict(x)
    return outliers