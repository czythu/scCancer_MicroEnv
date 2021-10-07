import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from vasc_pytorch import VASC_pytorch
from myfunc import performance_assessment, clustering, plot_2dimensions
from dataset import config, MyDataset, preprocessing

if __name__ == '__main__':
    if torch.cuda.is_available():
        mydevice = "cuda"
    else:
        mydevice = "cpu"
    print(mydevice)

    device = torch.device(mydevice)
    expr, id_map, label_int, batch_size = preprocessing('biase', log=config['log'], scale=config['scale'])
    train_dataset = MyDataset(expr, label_int)
    loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = VASC_pytorch(in_dim=expr.shape[1], latent=2, gpu=(mydevice == "cuda"), var=config['var']).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    all_res, all_train_loss = [], []
    tau = config['tau0']
    prev_loss = np.inf

    for epoch in range(config['epoch']):
        train_loss = 0
        if epoch % 100 == 0 and config['annealing']:
            tau = max(config['tau0'] * np.exp(-config['anneal_rate'] * epoch), config['min_tau'])
            print(tau)
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

            if all_labels is None:
                all_labels = label
            else:
                all_labels = torch.cat([all_labels, label])
            if all_res is None:
                all_res = z
            else:
                all_res = torch.cat([all_res, z])

            lossvalue = model.loss_function(expr, z_mean, z_log_var, out)
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
            cl, _ = clustering(result, k=k)
            dm = performance_assessment(cl, label)
            delta = abs(all_train_loss[epoch] - all_train_loss[epoch - 1])
            if epoch + 1 == config['epoch'] or (delta < config['threshold'] and epoch > config['min_stop']):
                plot_2dimensions(result, label, 'VASC-2.0')
                break

