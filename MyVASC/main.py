import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from vasc_pytorch import VascPytorch
from myfunc import performance_assessment, clustering, plot2dimensions
from dataset import config, MyDataset, preprocessing_txt, preprocessing_npy

if __name__ == '__main__':
    if torch.cuda.is_available():
        mydevice = 'cuda'
    else:
        mydevice = 'cpu'
    print(mydevice)

    device = torch.device(mydevice)
    # expr, id_map, label_int, batch_size = preprocessing_npy(log=config['log'], scale=config['scale'])
    expr, id_map, label_int, batch_size = preprocessing_txt('biase', log=config['log'], scale=config['scale'])
    train_dataset = MyDataset(expr, label_int)
    loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = VascPytorch(in_dim=expr.shape[1], latent=2, dist=config['latent_dist'],
                        gpu=(mydevice == 'cuda'), var=config['var']).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    all_res, all_train_loss = [], []
    tau = config['tau0']
    prev_loss = np.inf

    for epoch in range(config['epoch']):
        train_loss = 0
        # annealing
        if epoch % 100 == 0 and config['annealing']:
            tau = max(config['tau0'] * np.exp(-config['anneal_rate'] * epoch), config['min_tau'])
            print("tau = %s" % tau)

        model.train()
        all_res = None
        all_labels = None
        for i, (X, label) in enumerate(loader):
            X = Variable(X)
            label = Variable(label)
            tau_in = torch.tensor(np.ones(X.shape, dtype='float32') * tau, dtype=torch.float32)
            if mydevice == "cuda":
                X, label, tau_in = X.cuda(), label.cuda(), tau_in.cuda()
            expr, z, z_mean, z_log_var, out = model([X, tau_in])
            if config['latent_dist'] == 'Normal':
                lossvalue = model.lossfunction_normal(expr, z_mean, z_log_var, out)
            else:   # config['latent_dist'] == 'Negative binomial'
                lossvalue = model.lossfunction_nb(expr, z_mean, z_log_var, out)

            if all_labels is None:
                all_labels = label
            else:
                all_labels = torch.cat([all_labels, label])
            if all_res is None:
                all_res = z
            else:
                all_res = torch.cat([all_res, z])

            if i % 50 == 0:
                print('batch' + str(i) + ': recons_loss = ' +
                      str(lossvalue['Reconstruction_Loss']) + '; kl_loss = ' + str(lossvalue['KLD']))
            optimizer.zero_grad()
            lossvalue['loss'].backward()
            optimizer.step()
            train_loss += float(lossvalue['loss'])

        all_train_loss.append(train_loss / len(loader))
        print("epoch:" + str(epoch) + ": train loss = " + ' ' + str(train_loss / len(loader)))

        if epoch % config['patience'] == 1 or epoch + 1 == config['epoch']:
            result = all_res.cpu().detach().numpy()
            label = all_labels.cpu().detach().numpy()
            k = len(np.unique(label))
            if config['clustering_method'] == 'cmeans':
                cluster, center, u_matrix, fpc = clustering(result, k=k, method=config['clustering_method'])
                print("fpc = %s" % fpc)
            else:  # config['clustering_method'] == 'kmeans' or 'spec'
                cluster, _ = clustering(result, k=k, method=config['clustering_method'])
            dm = performance_assessment(cluster, label)
            delta = abs(all_train_loss[epoch] - all_train_loss[epoch - 1])
            if epoch + 1 == config['epoch'] or (delta < config['threshold'] and epoch > config['min_stop']):
                plot2dimensions(result, label, id_map, 'VASC-2.0')
                break

