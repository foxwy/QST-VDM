# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-09-03 16:22:11
# @Last Modified by:   yong
# @Last Modified time: 2023-03-30 15:43:43
# @Paper: Learning Quantum Distributions with Variational Diffusion Models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons, make_s_curve, \
    make_swiss_roll


def swiss_roll_data(n_data=10**4, noise=0.1):
    swiss_roll, _ = make_swiss_roll(n_data, noise=noise)
    swiss_roll = swiss_roll[:, [0, 2]] / 10.0

    return swiss_roll


def s_curve_data(n_data=10**4, noise=0.1):
    s_curve, _ = make_s_curve(n_data, noise=noise)
    s_curve = s_curve[:, [0, 2]] / 10.0

    return s_curve


def moons_data(n_data=10**4, noise=0.1):
    moons, _ = make_moons(n_data, noise=noise)

    return moons


def circles_data(n_data=10**4, noise=0.01):
    circles, _ = make_circles(n_data, noise=noise)

    return circles


def blobs_data(n_data=10**4, centers=4, n_features=2):
    blobs, _ = make_blobs(n_data, centers=centers, n_features=n_features)

    return blobs / 10.0


def int2bin(num_array):  # numpy
    (N, L) = num_array.shape
    b = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    return np.reshape(b[num_array], [N, L*2])

def bin2int(num_array):
    (N, L) = num_array.shape
    a = np.reshape(num_array, [N, int(L / 2), 2])
    b = np.array([2, 1])
    return a.dot(b)

# onehot
def onehot(data, k):
    (N, L) = data.shape
    data_onehot = np.reshape(np.eye(k)[data], [N, L*k]).astype(np.uint8)

    return data_onehot


# ati_onehot
def ati_onehot(data_onehot, k):
    (N, L) = data_onehot.shape
    N_q = int(L // k)
    data_onehot_reshape = np.reshape(data_onehot, [N * N_q, k])
    data_ati_onehot = np.argmax(data_onehot_reshape, axis=1)
    data_ati_onehot = np.reshape(data_ati_onehot, [N, N_q])

    return data_ati_onehot

def int_data(n_data=10**2, n_features=2):
    a = [0, 1, 2, 3]
    data = np.random.choice(a, (n_data, n_features), p=[0.1, 0.2, 0.3, 0.4])

    return data


if __name__ == "__main__":
    data = moons_data(10**3, 0.1)
    print(data)
    #values, counts = np.unique(data, axis=0, return_counts=True)
    #print(values, counts)
    #print(bin2int(int2bin(data)))

    plt.scatter(*data.T, alpha=0.5, color='white', edgecolor='gray', s=5)
    plt.show()
