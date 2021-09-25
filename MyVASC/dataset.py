import numpy as np

config = {
    'epoch':10000,
    'batch_size':256,
    'latent':2,
    'log':False,
    'scale':True,
    'patience':50
}

def preprocessing(dataset, log, scale):
    # DATASET = 'biase', PREFIX = 'biase'
    DATASET = dataset  # sys.argv[1]
    filename = DATASET + '.txt'
    data = open(filename)
    head = data.readline().rstrip().split()

    label_file = open(DATASET + '_label.txt')
    label_dict = {}
    for line in label_file:
        temp = line.rstrip().split()
        label_dict[temp[0]] = temp[1]
    label_file.close()

    label = []
    for c in head:
        if c in label_dict.keys():
            label.append(label_dict[c])
        else:
            print(c)

    label_set = []
    for c in label:
        if c not in label_set:
            label_set.append(c)
    name_map = {value: idx for idx, value in enumerate(label_set)}
    id_map = {idx: value for idx, value in enumerate(label_set)}
    label = np.asarray([name_map[name] for name in label])

    expr = []
    for line in data:
        temp = line.rstrip().split()[1:]
        temp = [float(x) for x in temp]
        expr.append(temp)

    expr = np.asarray(expr).T
    n_cell, _ = expr.shape
    if n_cell > 150:
        batch_size = config['batch_size']
    else:
        batch_size = 32

    expr[expr < 0] = 0.0

    if log:
        expr = np.log2(expr + 1)
    if scale:
        for i in range(expr.shape[0]):
            expr[i, :] = expr[i, :] / np.max(expr[i, :])

    return expr, id_map, label, batch_size