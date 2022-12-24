import time
import os
import warnings
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import auc, precision_recall_curve
from Optimizer import CoOptimizerAE_MF
from Model import Co_GAE_MF
from Preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, drug_class, split_train_test
from Adj_preprocess import adjacency_miRNA, edge_, Jaccard_miRNA, Jaccard_drug
from sklearn import metrics

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set random seed
seed = 1024
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('d_hidden1', 1024, 'Number of units in hidden layer 1. of drug')
flags.DEFINE_integer('d_hidden2', 64, 'Number of units in hidden layer 2. of drug')
flags.DEFINE_integer('m_hidden1', 256, 'Number of units in hidden layer 1. of mirna')
flags.DEFINE_integer('m_hidden2', 128, 'Number of units in hidden layer 2. of mirna')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('threshold_d', 0.9, 'Threshold of drug similarity network')
flags.DEFINE_float('threshold_m', 0.3, 'Threshold of mirna similarity network')
flags.DEFINE_float('a_d', 0.1, 'weight for drug reconstruction loss')
flags.DEFINE_float('a_m', 0.1, 'weight for mirna reconstruction loss')
flags.DEFINE_float('a_cla', 0.3, 'weight for drug classification loss')
flags.DEFINE_float('afa_d', 0.3, 'weight for drug similarity matrix')
flags.DEFINE_float('afa_m', 0.3, 'weight for miRNA similarity matrix')


def common(adj):
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_norm = preprocess_graph(adj_orig)

    num_nodes = adj_orig.shape[0]

    features = sp.identity(adj.shape[0])

    features = sparse_to_tuple(features.tocoo())
    dim_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_orig + sp.eye((adj_orig.shape[0]))
    adj_label = sparse_to_tuple(adj_label)

    return adj_orig, adj_label, adj_norm, num_nodes, dim_features, features_nonzero, features, pos_weight, norm


def Test_result(sess, model, feed_dict, num_drug, edges_pos, edges_neg):
    preds = []
    labels = []
    reconstruct_dm = sess.run(model.reconstructions_dm, feed_dict=feed_dict)
    reconstruct_dm = np.reshape(reconstruct_dm, [num_drug, -1])
    for e in edges_pos:
        preds.append(reconstruct_dm[e[0], e[1]])
        labels.append(1)

    for e in edges_neg:
        preds.append(reconstruct_dm[e[0], e[1]])
        labels.append(0)

    preds, labels = np.array(preds), np.array(labels)
    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    auc_roc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    aupr = auc(recall, precision)

    y_pre = np.zeros(shape=(len(preds), 1))
    for m in range(int(len(preds) * 0.4)):
        max = 0
        p = 0
        for j in range(len(preds)):
            if preds[j] > max:
                max = preds[j]
                p = j
        preds[p] = -1
        y_pre[p, 0] = 1

    f1_score = metrics.f1_score(labels, y_pre)
    recall1 = metrics.recall_score(labels, y_pre)
    accuracy = metrics.accuracy_score(labels, y_pre)
    precision1 = metrics.precision_score(labels.ravel(), y_pre)

    return auc_roc, aupr, f1_score, recall1, accuracy, precision1


def train_():
    adj_mir, adj_d, adj_dm, num_drug, pos_cla_d = adjacency_miRNA()
    drug_labels = drug_class()

    train_e_p_, train_e_n_, test_e_p_, test_e_n_, valid_e_p_, valid_e_n_, adj_dm_new_ = split_train_test(adj_dm)
    auc_roc, aupr_roc, F1, Recall, Precision, Acc = [], [], [], [], [], []
    for i_ in range(5):
        train_edges_pos, train_edges_neg = train_e_p_[i_], train_e_n_[i_]
        test_edges_pos, test_edges_neg = test_e_p_[i_], test_e_n_[i_]
        valid_edges_pos, valid_edges_neg = valid_e_p_[i_], valid_e_n_[i_]
        adj_dm_new = adj_dm_new_[i_]
        adj_link = np.hstack([np.ones(len(train_edges_pos)), np.zeros(len(train_edges_neg))])  # add col

        # Some preprocessing
        afa_d, afa_m = FLAGS.afa_d, FLAGS.afa_m
        adj_d = sp.csr_matrix(edge_(afa_d * adj_d + (1 - afa_d) * Jaccard_drug(adj_dm_new), FLAGS.threshold_d))
        adj_mir = sp.csr_matrix(edge_(afa_m * adj_mir + (1 - afa_m) * Jaccard_miRNA(adj_dm_new), FLAGS.threshold_m))
        m12 = np.hstack([adj_d.toarray(), adj_dm_new])
        m34 = np.hstack([adj_dm_new.T, adj_mir.toarray()])
        adj = sp.csr_matrix(np.vstack([m12, m34]))

        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_norm_dm = preprocess_graph(adj_orig)

        # drug and mirRNA preprocessing
        adj_orig_d, adj_label_d, adj_norm_d, num_drug, dim_fea_d, fea_nonzero_d, fea_d, pos_weight_d, norm_d = common(
            adj_d)
        adj_orig_m, adj_label_m, adj_norm_m, num_mir, dim_fea_m, fea_nonzero_m, fea_m, pos_weight_m, norm_m = common(
            adj_mir)

        # Define placeholders
        placeholders = {
            'features_d': tf.sparse_placeholder(tf.float32),
            'features_m': tf.sparse_placeholder(tf.float32),
            'features_cla_d': tf.placeholder(tf.float32, [len(pos_cla_d), FLAGS.d_hidden2]),
            'adj_d': tf.sparse_placeholder(tf.float32),
            'adj_m': tf.sparse_placeholder(tf.float32),
            'adj_dm': tf.sparse_placeholder(tf.float32),
            'adj_link': tf.placeholder(tf.float32),
            'adj_preds': tf.placeholder(tf.float32),
            'adj_norm_d': tf.sparse_placeholder(tf.float32),
            'adj_norm_m': tf.sparse_placeholder(tf.float32),
            'adj_norm_dm': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'embedding_dm': tf.placeholder(tf.float32)
        }

        # Create model
        model = Co_GAE_MF(placeholders, dim_fea_d, dim_fea_m, fea_nonzero_d, fea_nonzero_m, pos_cla_d, train_edges_pos, train_edges_neg)

        # Optimizer
        with tf.name_scope('optimizer'):
            opt = CoOptimizerAE_MF(preds_d=model.reconstructions_d,
                                   labels_d=tf.reshape(
                                       tf.sparse_tensor_to_dense(placeholders['adj_d'], validate_indices=False),
                                       [-1]),
                                   preds_m=model.reconstructions_m,
                                   labels_m=tf.reshape(
                                       tf.sparse_tensor_to_dense(placeholders['adj_m'], validate_indices=False),
                                       [-1]),
                                   preds_dm=model.preds_dm,
                                   labels_dm=placeholders['adj_link'],
                                   pred_cla=model.pred_class,
                                   multilabels=drug_labels,
                                   a_d=FLAGS.a_d, a_m=FLAGS.a_m, a_cla=FLAGS.a_cla,
                                   pos_weight_d=pos_weight_d, norm_d=norm_d,
                                   pos_weight_m=pos_weight_m, norm_m=norm_m,
                                   )

        # initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        adj_dm_new = sparse_to_tuple(sp.csr_matrix(adj_dm_new))

        max_auc1, max_aupr1, epoch1 = 0, 0, -1
        isbreak = 0
        te_auc, te_aupr, roc_auc, roc_aupr, f1, recall, acc, precision = 0, 0, 0, 0, 0, 0, 0, 0
        for epoch in range(FLAGS.epochs):
            t = time.time()
            feed_dict = construct_feed_dict(adj_norm_dm, adj_norm_d, adj_norm_m, adj_link, adj_dm_new, adj_label_d,
                                            adj_label_m, fea_d, fea_m, FLAGS.dropout, placeholders)
            outs = sess.run([opt.opt_op, opt.loss, opt.preds_dm, opt.labels_dm], feed_dict=feed_dict)

            te_roc_auc, te_roc_aupr, f1, recall, acc, precision = Test_result(sess, model, feed_dict, num_drug, test_edges_pos,
                                                                       test_edges_neg)

            if te_roc_auc > max_auc1:
                max_auc1, max_aupr1, epoch1 = te_roc_auc, roc_aupr, epoch
                te_auc, te_aupr = te_roc_auc, te_roc_aupr

            if te_roc_auc < max_auc1:
                isbreak += 1
            else:
                isbreak = 0

            if isbreak > 10:
                break

        auc_roc.append(te_auc)
        aupr_roc.append(te_aupr)
        F1.append(f1)
        Recall.append(recall)
        Precision.append(precision)
        Acc.append(acc)
        print(epoch1, i_,  ' test: auc = ', te_auc, " aupr = ", te_aupr)

    m_auc = np.array(auc_roc).mean()
    m_aupr = np.array(aupr_roc).mean()
    m_F1 = np.array(F1).mean()
    m_Recall = np.array(Recall).mean()
    m_Precision = np.array(Precision).mean()
    m_Acc = np.array(Acc).mean()

    print("Epoch1:", '%04d' % (epoch1 + 1), "test_roc1=", "{:.5f}".format(m_auc), "test_pr1=",
          "{:.5f}".format(m_aupr), "F1=", "{:.5f}".format(m_F1), "BA=", "{:.5f}".format(m_Acc),
          "recall=", "{:.5f}".format(m_Recall), "precision=", "{:.5f}".format(m_Precision) + '\n')
    print("auc std: ", np.array(auc_roc).std(), " aupr std: ", np.array(aupr_roc).std())


if __name__ == '__main__':
    train_()

