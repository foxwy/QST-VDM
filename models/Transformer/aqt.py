import time
import numpy as np
import torch
from torchsummary import summary

import sys
sys.path.append('../../')

import models.Transformer.ann as A


# Basic parameters
def AQT(data_train, N_samples, K, n_qubits, N_epoch, Nl=2, dmodel=64, Nh=4):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    Nq = n_qubits
    Nep = N_epoch

    # Load data
    data = data_train
    np.random.shuffle(data)

    # Train model
    model = A.InitializeModel(
        Nq, Nlayer=Nl, dmodel=dmodel, Nh=Nh, Na=K, device=device).to(device)

    para = sum([p.nelement() for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para / 10**6))

    t = time.time()
    model, loss = A.TrainModel(
        model, data, device, batch_size=200, lr=1e-3, Nep=Nep)
    print('Took %f minutes' % ((time.time() - t) / 60))

    samples, _ = A.sample_mp(model, N_samples, 10**5, 'cuda')

    return samples
