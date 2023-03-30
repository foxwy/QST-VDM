# -*- coding: utf-8 -*-
# @Author: foxwy
# @Function: Quantum state and quantum measurment
# @Paper: Ultrafast quantum state tomography with feed-forward neural networks

import sys
import time
import argparse
import numpy as np
import multiprocessing as mp

sys.path.append('..')

# external libraries
from Basis.Basis_State import Mea_basis
from Basis.Basic_Function import qmt, samples_mp


class PaState(Mea_basis):
    """
    Mimic quantum measurements to generate test samples.

    Examples::
        >>> sampler = PaState(basis='Tetra4', n_qubits=2, State_name='GHZ_P', P_state=1)
        >>> sampler.samples_product(10**5, 'N_2', save_flag=True)
    """
    def __init__(self, basis='Tetra', n_qubits=2, State_name='GHZ_P', P_state=1.0):
        """
        Args:
            basis (str): The name of measurement, as Mea_basis().
            n_qubits (int): The number of qubits.
            State_name (str): The name of state, as State().
            P_state (float): The P of Werner state, pure state when p == 1, identity matrix when p == 0.
        """
        super().__init__(basis)
        self.N = n_qubits
        self.State_name = State_name
        self.p = P_state
        self.M = Mea_basis(basis).M
        _, self.rho = self.Get_state_rho(State_name, n_qubits, P_state)

    def samples_product(self, Ns=1000000, filename='N2', group_N=1000, save_flag=True):
        """
        Faster using product structure and multiprocessing for batch processing.

        Args:
            Ns (int): The number of samples wanted.
            filename (str): The name of saved file.
            group_N (int): The number of samples a core can process at one time for collection, 
                [the proper value will speed up, too much will lead to memory explosion].
            save_flag (bool): If True, the sample data is saved as a file '.txt'.

        Returns:
            array: sample in k decimal.
            array: sample in onehot encoding.
        """
        if save_flag:
            if 'P' in self.State_name:  # mix state
                f_name = 'data/' + self.State_name + '_' + str(self.p) + '_' + self.basis + '_train_' + filename + '.txt'
                f2_name = 'data/' + self.State_name + '_' + str(self.p) + '_' + self.basis + '_data_' + filename + '.txt'
            else:  # pure state
                f_name = 'data/' + self.State_name + '_' + self.basis + '_train_' + filename + '.txt'
                f2_name = 'data/' + self.State_name + '_' + self.basis + '_data_' + filename + '.txt'

        P_all = qmt(self.rho, [self.M] * self.N)  # probs of all operators in product construction

        # Multi-process sampling data
        if Ns < group_N:
            group_N = Ns

        params = [[P_all, group_N, self.K, self.N]] * (Ns // group_N)
        if Ns % group_N != 0:
            params.append([P_all, Ns % group_N, self.K, self.N])

        cpu_counts = mp.cpu_count()
        if len(params) < cpu_counts:
            cpu_counts = len(params)

        time_b = time.perf_counter()  # sample time
        print('---begin multiprocessing---')
        with mp.Pool(cpu_counts) as pool:
            results = pool.map(samples_mp, params)
            pool.close()
            pool.join()
        print('---end multiprocessing---')

        # Merge sampling results
        S_all = results[0][0]
        S_one_hot_all = results[0][1]
        print('num:', group_N)
        for num in range(1, len(results)):
            if Ns % group_N != 0 and num == len(results) - 1:
                print('num:', group_N * num + len(results[num][0]))
            else:
                print('num:', group_N * (num + 1))
            S_all = np.vstack((S_all, results[num][0]))
            S_one_hot_all = np.vstack((S_one_hot_all, results[num][1]))
        print('---finished generating samples---')

        if save_flag:
            print('---begin write data to text---')
            np.savetxt(f_name, S_one_hot_all, '%d')
            np.savetxt(f2_name, S_all, '%d')
            print('---end write data to text---')

        print('sample time:', time.perf_counter() - time_b)

        return S_all, S_one_hot_all


#--------------------main--------------------
if __name__ == '__main__':
    print('-'*20+'set parser'+'-'*20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ex", type=int, default=1, choices=[1, 2], help="number of experiments")
    parser.add_argument("--state", type=str, default="GHZ_P", choices=["GHZ_P", "W_P"], help="number of experiments")
    args = parser.parse_args()

    if args.n_ex == 1:  # section 4.2   
        print('section 4.2 experiments')
        num_qubits = 8
        sample_num = 10**4
        sampler = PaState(basis='Tetra4', n_qubits=num_qubits, State_name=args.state, P_state=1.0)
        sampler.samples_product(sample_num, 'N_t'+str(num_qubits), save_flag=True)
    else:
        print('section 4.3 experiments')
        for i in [2, 4, 6, 8, 10]:
            print('-----n_qubit:', i)
            num_qubits = i
            sample_num = 10**3 * i
            sampler = PaState(basis='Tetra4', n_qubits=num_qubits, State_name=args.state, P_state=1.0)
            sampler.samples_product(sample_num, 'N'+str(num_qubits), save_flag=True)
