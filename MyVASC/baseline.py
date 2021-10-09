from myfunc import plot2dimensions
from dataset import config, preprocessing_txt, preprocessing_npy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope


# outliers detection
def outliers_detection(expr):
    x = PCA(n_components=2).fit_transform(expr)
    ee = EllipticEnvelope()
    ee.fit(x)
    outliers = ee.predict(x)
    return outliers


expr, id_map, label_int, batch_size = preprocessing_txt('biase', log=config['log'], scale=config['scale'])
# expr, id_map, label_int, batch_size = preprocessing_npy(log=config['log'], scale=config['scale'])
model_PCA = PCA(n_components=2)
res_PCA = model_PCA.fit_transform(expr)
plot2dimensions(res_PCA, label_int, id_map, 'PCA')
model_tSNE = TSNE(n_components=2)
res_tSNE = model_tSNE.fit_transform(expr)
plot2dimensions(res_tSNE, label_int, id_map, 'tSNE')
