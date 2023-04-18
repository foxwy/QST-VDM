# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-03-30 11:01:27
# @Last Modified by:   yong
# @Last Modified time: 2023-03-30 17:37:30
# @Paper: Learning Quantum Distributions with Variational Diffusion Models

import argparse
import numpy as np

from Basis.Basic_Function import array_posibility_unique
from models.RNN.rnn import LatentAttention
from models.Transformer.aqt import AQT
from models.DDM.VariationalDiffusionModel import VDM


def Ex_1(args):
    N_q = 8
    N_s = 10**4
    povm = 'Tetra4'
    K = 4
    N_s_t = [1000, 2000, 5000, 8000, 10000, 20000, 50000, 100000, 200000, 500000, 800000, 1000000]

    if args.model == "RNN":
        N_epoch = 150
    elif args.model == "AQT":
        N_epoch = 100
    else:
        N_epoch = 1000
        N_batch = 500

    state_name = args.state
    N_sample = N_s_t[-1]

    if args.model == "RNN":
        trainFileName = 'datasets/data/' + state_name + '_1.0_' + povm + '_train_N_t' + str(N_q) + '.txt'
        data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
        model = LatentAttention(data_train=data_train, n_sample=N_s, K=K, n_qubits=N_q, N_epoch=N_epoch, decoder='TimeDistributed_mol', Nsamples=N_sample)
        samples = model.train()
    elif args.model == "AQT":
        trainFileName = 'datasets/data/' + state_name + '_1.0_' + povm + '_data_N_t' + str(N_q) + '.txt'
        data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
        samples = AQT(data_train=data_train, N_samples=N_sample, K=K, n_qubits=N_q, N_epoch=N_epoch)
    else:
        trainFileName = 'datasets/data/' + state_name + '_1.0_' + povm + '_data_N_t' + str(N_q) + '.txt'
        data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
        samples = VDM(data_train, N_sample, N_epoch, N_batch, n_steps=500)

    for i in N_s_t:
        N_sample = i
        samples_i = samples[:i]
        samples, P = array_posibility_unique(samples_i)
        np.savetxt('results/sample_P/' + args.model + '_' + state_name + '_' + povm + '_N' + str(N_q) + '_' + str(int(N_sample/1000)) + '_sample.txt', samples, '%d')
        np.savetxt('results/sample_P/' + args.model + '_' + state_name + '_' + povm + '_N' + str(N_q) + '_' + str(int(N_sample/1000)) + '_P.txt', P)


def Ex_2(args):
    state_name = args.state
    povm = 'Tetra4'
    K = 4

    if args.model == "RNN":
        N_epoch = 15
    elif args.model == "AQT":
        N_epoch = 10
    else:
        N_epoch = 10
        N_batch = 500

    for i in [2, 4, 6, 8, 10]:
        print('number of qubits:', i)
        N_q = i
        N_s = 10**3 * N_q
        #N_sample = 10**4 * N_q**2
        N_sample = 10**3

        if args.model == "RNN":
            trainFileName = 'datasets/data/' + state_name + '_1.0_' + povm + '_train_N' + str(N_q) + '.txt'
            data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
            model = LatentAttention(data_train=data_train, n_sample=N_s, K=K, n_qubits=N_q, N_epoch=N_epoch, decoder='TimeDistributed_mol', Nsamples=N_sample)
            samples = model.train()
        elif args.model == "AQT":
            trainFileName = 'datasets/data/' + state_name + '_1.0_' + povm + '_data_N' + str(N_q) + '.txt'
            data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
            samples = AQT(data_train=data_train, N_samples=N_sample, K=K, n_qubits=N_q, N_epoch=N_epoch)
        else:
            trainFileName = 'datasets/data/' + state_name + '_1.0_' + povm + '_data_N' + str(N_q) + '.txt'
            data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
            samples = VDM(data_train, N_sample, N_epoch, N_batch, n_steps=500)

        #samples, P = array_posibility_unique(samples)
        #np.savetxt('results/sample_P/' + args.model + '_' + state_name + '_' + povm + '_N' + str(N_q) + '_sample.txt', samples, '%d')
        #np.savetxt('results/sample_P/' + args.model + '_' + state_name + '_' + povm + '_N' + str(N_q) + '_P.txt', P)


if __name__ == '__main__':
    print('-'*20+'set parser'+'-'*20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ex", type=int, default=1, choices=[1, 2], help="number of experiments")
    parser.add_argument("--state", type=str, default="GHZ_P", choices=["GHZ_P", "W_P"], help="number of experiments")
    parser.add_argument("--model", type=str, default="RNN", choices=["RNN", "AQT", "VDM"], help="number of experiments")
    args = parser.parse_args()

    if args.n_ex == 1:
        print('section 4.2 experiments')
        Ex_1(args)
    else:
        print('section 4.3 experiments')
        Ex_2(args)