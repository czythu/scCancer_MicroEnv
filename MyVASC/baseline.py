from myfunc import *
from dataset import *

# expr, id_map, label_int, batch_size = preprocessing_txt('biase', log=config['log'], scale=config['scale'])
expr, id_map, label_int, batch_size = preprocessing_npy('biase', log=config['log'], scale=config['scale'])
model_PCA = PCA(n_components=2)
res_PCA = model_PCA.fit_transform(expr)
plot_2dimensions(res_PCA, label_int, 'PCA')
model_tSNE = TSNE(n_components=2)
res_tSNE = model_tSNE.fit_transform(expr)
plot_2dimensions(res_tSNE, label_int, 'tSNE')
