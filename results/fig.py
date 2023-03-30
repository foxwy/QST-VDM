# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-10-14 09:17:52
# @Last Modified by:   yong
# @Last Modified time: 2023-03-30 15:43:06
# @Paper: Learning Quantum Distributions with Variational Diffusion Models

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import openpyxl

filepath = os.path.abspath(os.path.join(os.getcwd(), '..'))

import sys
sys.path.append('../')

from Basis.Basis_State import State
from evaluation.Fidelity import Fid
from Basis.Basic_Function import array_posibility_unique

plt.style.use(['science', 'no-latex'])
plt.rcParams["font.family"] = 'Arial'

font_size = 24  # 34, 28, 38
font = {'size': font_size, 'weight': 'normal'}


def Plt_set(ax, xlabel, ylabel, savepath, log_flag=0, loc=4, ncol=1, f_size=20):
    ax.tick_params(labelsize=24)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.tick_params(width=4)
    font2 = {'size': f_size, 'weight': 'normal'}
    ax.legend(prop=font2, loc=loc, frameon=True, ncol=ncol, edgecolor='k')
    ax.set_xlabel(xlabel, font)  # fontweight='bold'
    ax.set_ylabel(ylabel, font)

    if log_flag == 1:
        ax.set_xscale('log')
    if log_flag == 2:
        ax.set_yscale('log')
    if log_flag == 3:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    if 'samples' in ylabel:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        tx = ax.yaxis.get_offset_text()
        tx.set_fontsize(22)

    plt.grid(linestyle='--', linewidth=1)

    plt.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)


if __name__ == '__main__':
    colors = ['#ff3efc', '#4b9cc8', '#b2b2b2', '#f94d45']
    N_qubit = [2, 4, 6, 8, 10]
    state_name = 'GHZ_P'
    rho_p = 1.0
    povm = 'Tetra4'

    #-----------------------------------------------------------------------------------------------
    '''
    cF_dm = []
    cF_rnn = []
    cF_attn = []
    cF_o = []
    for i in N_qubit:
        print('-----------------', i)
        N_q = i
        N_s = 10**3 * N_q
        _, rho_star = State().Get_state_rho(state_name, N_q, rho_p)
        Ficalc = Fid(basis=povm, n_qubits=N_q, rho_star=rho_star, torch_flag=0)

        #-----DM-----
        fdm_sample = 'samples_P/VDM_' + state_name + '_' + povm + '_N' + str(i) + '_sample.txt'
        fdm_P = 'DDM/samples/VDM_' + state_name + '_' + povm + '_N' + str(i) + '_P.txt'
        dm_sample = np.loadtxt(fdm_sample).astype(np.uint8)
        dm_P = np.loadtxt(fdm_P)

        cF = Ficalc.cFidelity_S_product(dm_sample, dm_P)
        print('dm cfdelity:', cF)
        cF_dm.append(cF)

        #-----RNN----
        frnn_sample = 'samples_P/RNN_' + state_name + '_' + povm + '_N' + str(i) + '_sample.txt'
        frnn_P = 'samples_P/RNN_' + state_name + '_' + povm + '_N' + str(i) + '_P.txt'
        rnn_sample = np.loadtxt(frnn_sample).astype(np.uint8)
        rnn_P = np.loadtxt(frnn_P)

        cF = Ficalc.cFidelity_S_product(rnn_sample, rnn_P)
        print('rnn cfdelity:', cF)
        cF_rnn.append(cF)

        #-----Transformer----
        attn_sample = 'samples_P/AQT_' + state_name + '_' + povm + '_N' + str(i) + '_sample.txt'
        attn_P = 'samples_P/AQT_' + state_name + '_' + povm + '_N' + str(i) + '_P.txt'
        attn_sample = np.loadtxt(attn_sample).astype(np.uint8)
        attn_P = np.loadtxt(attn_P)

        cF = Ficalc.cFidelity_S_product(attn_sample, attn_P)
        print('attn cfdelity:', cF)
        cF_attn.append(cF)

        #-----original-----
        trainFileName = filepath + '/datasets/data/' + state_name + '_1.0_' + povm + '_data_N' + str(i) + '.txt'
        data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
        samples, P = array_posibility_unique(data_train)
        cF = Ficalc.cFidelity_S_product(samples, P)
        print('original cfdelity:', cF)
        cF_o.append(cF)

    np.save('figure/'+state_name+'_cF_dm.npy', cF_dm)
    np.save('figure/'+state_name+'_cF_rnn.npy', cF_rnn)
    np.save('figure/'+state_name+'_cF_attn.npy', cF_attn)
    np.save('figure/'+state_name+'_cF_o.npy', cF_o)'''

    cF_dm = np.load('figure/'+state_name+'_cF_dm.npy')
    cF_rnn = np.load('figure/'+state_name+'_cF_rnn.npy')
    cF_attn = np.load('figure/'+state_name+'_cF_attn.npy')
    cF_o = np.load('figure/'+state_name+'_cF_o.npy')
    print(cF_dm, cF_rnn, cF_attn, cF_o)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(N_qubit, cF_dm, label='VDM (proposed)', linewidth=4, color=colors[0])
    ax.plot(N_qubit, cF_rnn, label='RNN', linewidth=4, color=colors[1])
    ax.plot(N_qubit, cF_attn, label='Transformer', linewidth=4, color=colors[2])
    ax.plot(N_qubit, cF_o, label='Original', linewidth=4, color=colors[3])
    ax.plot([2, 10], [1, 1], 'k--', linewidth=4)
    ax.set_xticks(N_qubit)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    Plt_set(ax, "Number of qubits", 'Classical Fidelity', 'figure/cF_qubit_'+state_name, loc=3)
    plt.show()


    #-----------------------------------------------------------------------------------------------
    state_name = 'GHZ_P'
    cFs_dm = []
    cFs_rnn = []
    cFs_attn = []
    cFs_o = []
    N_q = 8
    N_s = [1000, 2000, 5000, 8000, 10000, 20000, 50000, 100000, 200000, 500000, 800000, 1000000]
    '''
    for N in N_s:
        print('-----------------', N)
        _, rho_star = State().Get_state_rho(state_name, N_q, rho_p)
        Ficalc = Fid(basis=povm, n_qubits=N_q, rho_star=rho_star, torch_flag=0)

        #-----VDM-----
        fdm_sample = 'samples_P/VDM_' + state_name + '_' + povm + '_N' + str(N_q) + '_' + str(int(N/1000)) + '_sample.txt'
        fdm_P = 'samples_P/VDM_' + state_name + '_' + povm + '_N' + str(N_q) + '_' + str(int(N/1000)) + '_P.txt'
        dm_sample = np.loadtxt(fdm_sample).astype(np.uint8)
        dm_P = np.loadtxt(fdm_P)

        cF = Ficalc.cFidelity_S_product(dm_sample, dm_P)
        print('--------------------')
        print('VDM cfdelity:', cF)
        cFs_dm.append(cF)

        #----RNN-----
        frnn_sample = 'samples_P/RNN_' + state_name + '_' + povm + '_N' + str(N_q) + '_' + str(int(N/1000)) + '_sample.txt'
        frnn_P = 'samples_P/RNN_' + state_name + '_' + povm + '_N' + str(N_q) + '_' + str(int(N/1000)) + '_P.txt'
        rnn_sample = np.loadtxt(frnn_sample).astype(np.uint8)
        rnn_P = np.loadtxt(frnn_P)

        cF = Ficalc.cFidelity_S_product(rnn_sample, rnn_P)
        print('RNN cfdelity:', cF)
        cFs_rnn.append(cF)

        ##----Transformer-----
        fattn_sample = 'samples_P/AQT_' + state_name + '_' + povm + '_N' + str(N_q) + '_' + str(int(N/1000)) + '_sample.txt'
        fattn_P = 'samples_P/AQT_' + state_name + '_' + povm + '_N' + str(N_q) + '_' + str(int(N/1000)) + '_P.txt'
        attn_sample = np.loadtxt(fattn_sample).astype(np.uint8)
        attn_P = np.loadtxt(fattn_P)

        cF = Ficalc.cFidelity_S_product(attn_sample, attn_P)
        print('attn cfdelity:', cF)
        cFs_attn.append(cF)

        #----original----
        trainFileName = filepath + '/datasets/data/' + state_name + '_1.0_' + povm + '_data_N_t' + str(N_q) + '.txt'
        data_train = np.loadtxt(trainFileName)[:1000*N_q].astype(int)
        samples, P = array_posibility_unique(data_train)
        cF = Ficalc.cFidelity_S_product(samples, P)
        print('cfdelity:', cF)
        cFs_o.append(cF)

    np.save('figure/'+state_name+'_cFs_dm.npy', cFs_dm)
    np.save('figure/'+state_name+'_cFs_rnn.npy', cFs_rnn)
    np.save('figure/'+state_name+'_cFs_attn.npy', cFs_attn)
    np.save('figure/'+state_name+'_cFs_o.npy', cFs_o)'''

    cFs_dm = np.load('figure/'+state_name+'_cFs_dm.npy')
    cFs_rnn = np.load('figure/'+state_name+'_cFs_rnn.npy')
    cFs_attn = np.load('figure/'+state_name+'_cFs_attn.npy')
    cFs_o = np.load('figure/'+state_name+'_cFs_o.npy')
    print(cFs_dm, cFs_rnn, cFs_attn, cFs_o)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(N_s, cFs_dm, label='VDM (proposed)', linewidth=4, color=colors[0])
    ax.plot(N_s, cFs_rnn, label='RNN', linewidth=4, color=colors[1])
    ax.plot(N_s, cFs_attn, label='Transformer', linewidth=4, color=colors[2])
    ax.plot(N_s, cFs_o, label='Original', linewidth=4, color=colors[3])
    ax.plot([N_s[0], N_s[-1]], [1, 1], 'k--', linewidth=4)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    Plt_set(ax, "Number of samples", 'Classical Fidelity', 'figure/cF_sample_'+state_name, loc=0, log_flag=1)
    plt.show()