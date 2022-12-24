import numpy as np
import pandas as pd
import os
import scipy.sparse as sp


# Read data
def read_data(path):
    data = []
    for line in open(path, 'r'):
        ele = line.strip().split(" ")
        tmp = []
        for e in ele:
            if e != '':
                tmp.append(float(e))
        data.append(tmp)
    return data


# 设定阈值，将相似性转为边0， 1
def edge_(dataSim, threshold):
    for i in range(dataSim.shape[0]):
        for j in range(dataSim.shape[1]):
            if dataSim[i, j] >= threshold:
                dataSim[i, j] = 1
            else:
                dataSim[i, j] = 0
    return dataSim


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def Jaccard_drug(adj_dm):
    adj_dm = np.array(adj_dm)
    adj_d = np.zeros(shape=(len(adj_dm), len(adj_dm)))

    for row in range(len(adj_dm)):
        for col in range(len(adj_dm)):
            pq = np.dot(adj_dm[row, :], adj_dm[col, :])
            p = np.sum(adj_dm[row, :])
            q = np.sum(adj_dm[col, :])
            adj_d[row, col] = pq / (p + q - pq)
    return adj_d


def Jaccard_miRNA(adj_dm):
    adj_dm = np.array(adj_dm)
    adj_m = np.zeros(shape=(adj_dm.shape[1], adj_dm.shape[1]))

    for row in range(len(adj_m)):
        for col in range(len(adj_m)):
            pq = np.dot(adj_dm[:, row], adj_dm[:, col])
            p = np.sum(adj_dm[:, row])
            q = np.sum(adj_dm[:, col])
            adj_m[row, col] = pq / (p + q - pq)
    return adj_m


# 用的时miRNA数据
def adjacency_miRNA():
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data")
    data_d_sim = read_data(data_path+'\\drug228_smiles_sim_matrix_166.txt')
    data_d_sim = np.array(data_d_sim)

    data_m_sim = read_data(data_path+'\\SS-Similarity_Matrix_miRNA_BMA794.txt')
    data_m_sim = np.array(data_m_sim)

    data_dm = read_data(data_path+'\\Matrix_drug_miRNA.txt')
    data_dm = np.array(data_dm)

    pos_cla_d = pd.read_csv(data_path+'\\drug113_pos_in_drug228.txt', header=None)
    pos_cla_d = np.array(pos_cla_d)

    adj_d = sp.csr_matrix(data_d_sim)
    adj_mir = sp.csr_matrix(data_m_sim)

    return adj_mir, adj_d, data_dm, data_d_sim.shape[0], pos_cla_d


if __name__ == '__main__':
    adjacency_miRNA()