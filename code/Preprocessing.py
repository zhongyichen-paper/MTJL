import numpy as np
import os
import scipy.sparse as sp
import tensorflow as tf
from sklearn.model_selection import KFold


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = np.dot(np.dot(degree_mat_inv_sqrt, adj_), degree_mat_inv_sqrt)
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_norm_dm, adj_norm_d, adj_norm_m, adj_link, adj_dm_new, adj_d, adj_m, fea_d, fea_m, dropout, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features_d']: fea_d})
    feed_dict.update({placeholders['features_m']: fea_m})
    feed_dict.update({placeholders['adj_d']: adj_d})
    feed_dict.update({placeholders['adj_m']: adj_m})
    feed_dict.update({placeholders['adj_dm']: adj_dm_new})
    feed_dict.update({placeholders['adj_link']: adj_link})
    feed_dict.update({placeholders['adj_norm_d']: adj_norm_d})
    feed_dict.update({placeholders['adj_norm_m']: adj_norm_m})
    feed_dict.update({placeholders['adj_norm_dm']: adj_norm_dm})
    feed_dict.update({placeholders['dropout']: dropout})

    return feed_dict


def get_edges_K_C(adj_tuple, num_com):
    edges = adj_tuple[0]
    edges_K_C = []
    for i in range(len(edges)):
        if edges[i][0] >= num_com or edges[i][1] < num_com:
            edges_K_C.append(edges[i])
    coords = np.array(edges_K_C)
    values = np.ones(len(coords))
    return coords, values


def split_train_test(adj_dm):
    pos, neg = [], []
    for i in range(adj_dm.shape[0]):
        for j in range(adj_dm.shape[1]):
            if adj_dm[i, j] == 1:
                pos.append([i, j])
            elif adj_dm[i, j] == 0:
                neg.append([i, j])
    neg = np.array(neg)
    pos = np.array(pos)

    edges_neg_idx = list(range(len(neg)))
    np.random.shuffle(edges_neg_idx)
    balanced_edges_neg = neg[edges_neg_idx[:len(pos)]]

    edges_pos_idx = list(range(len(pos)))
    np.random.shuffle(edges_pos_idx)
    pos = pos[edges_pos_idx]

    ite = 5
    num_oneOFfive = int(np.floor(len(pos) / ite))

    Y = []
    for i in range(len(pos)):
        Y.append(i)
    kf = KFold(n_splits=5)
    test_edges_pos_, test_edges_neg_ = [], []
    train_edges_pos_, train_edges_neg_ = [], []
    valid_edges_pos_, valid_edges_neg_ = [], []
    adj_dm_new_, adj_train_ = [], []
    for train_id, test_id in kf.split(Y):
        valid_id, train_id = train_id[:num_oneOFfive], train_id[num_oneOFfive:]

        test_edges_pos, test_edges_neg = pos[test_id], balanced_edges_neg[test_id]
        train_edges_pos, train_edges_neg = pos[train_id], balanced_edges_neg[train_id]
        valid_edges_pos, valid_edges_neg = pos[valid_id], balanced_edges_neg[valid_id]

        adj_dm_new = np.zeros_like(adj_dm)
        for e in train_edges_pos:
            adj_dm_new[e[0], e[1]] = 1

        train_edges_pos_.append(train_edges_pos)
        train_edges_neg_.append(train_edges_neg)
        test_edges_pos_.append(test_edges_pos)
        test_edges_neg_.append(test_edges_neg)
        valid_edges_pos_.append(valid_edges_pos)
        valid_edges_neg_.append(valid_edges_neg)
        adj_dm_new_.append(adj_dm_new)

    return train_edges_pos_, train_edges_neg_, test_edges_pos_, test_edges_neg_, valid_edges_pos_, valid_edges_neg_, adj_dm_new_


def drug_class():
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data")
    path = data_path + '\\drug_113_multilabels.txt'
    f = open(path, encoding='utf-8')
    labels = []
    judge = f.readline()
    while judge:
        labels.append(np.array(judge.split('\n')[0].split(' ')))
        judge = f.readline()
    labels = np.array(labels, dtype=int)
    return tf.cast(labels, dtype=tf.float32)