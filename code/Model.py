from Layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, InnerProductDecoderMF2
import tensorflow as tf

flags = tf.app.flags

FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class Co_GAE_MF(Model):
    def __init__(self, placeholders, dim_features_d, dim_features_m,  fea_nonzero_d, fea_nonzero_m, pos_cla_d, tr_pos, tr_neg, **kwargs):
        super(Co_GAE_MF, self).__init__(**kwargs)

        self.input_d = placeholders['features_d']
        self.input_m = placeholders['features_m']
        self.adj_norm_d = placeholders['adj_norm_d']
        self.adj_norm_m = placeholders['adj_norm_m']
        self.fea_nonzero_d = fea_nonzero_d
        self.fea_nonzero_m = fea_nonzero_m
        self.input_dim_d = dim_features_d
        self.input_dim_m = dim_features_m
        self.pos_cla_d = pos_cla_d
        self.pred_dm = []
        self.tr_e_p = tr_pos
        self.tr_e_n = tr_neg

        self.adj_norm_dm = placeholders['adj_norm_dm']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1_d = GraphConvolutionSparse(input_dim=self.input_dim_d,
                                                output_dim=FLAGS.d_hidden1,
                                                adj_norm=self.adj_norm_d,
                                                features_nonzero=self.fea_nonzero_d,
                                                dropout=self.dropout,
                                                logging=self.logging)(self.input_d)
        self.hidden1_m = GraphConvolutionSparse(input_dim=self.input_dim_m,
                                                output_dim=FLAGS.m_hidden1,
                                                adj_norm=self.adj_norm_m,
                                                features_nonzero=self.fea_nonzero_m,
                                                dropout=self.dropout,
                                                logging=self.logging)(self.input_m)
        self.embeddings_d = GraphConvolution(input_dim=FLAGS.d_hidden1,
                                             output_dim=FLAGS.d_hidden2,
                                             adj_norm=self.adj_norm_d,
                                             act=lambda x: x,
                                             dropout=self.dropout,
                                             logging=self.logging)(self.hidden1_d)
        self.embeddings_m = GraphConvolution(input_dim=FLAGS.m_hidden1,
                                             output_dim=FLAGS.m_hidden2,
                                             adj_norm=self.adj_norm_m,
                                             act=lambda x: x,
                                             dropout=self.dropout,
                                             logging=self.logging)(self.hidden1_m)
        self.fea_cla_d = self.embeddings_d[3, :]
        for i in range(1, self.pos_cla_d.shape[0]):
            self.a = self.embeddings_d[self.pos_cla_d[i, 0], :]
            self.fea_cla_d = tf.concat([self.fea_cla_d, self.a], axis=0)
        self.fea_cla_d = tf.reshape(self.fea_cla_d, [-1, FLAGS.d_hidden2])
        self.pred_class0 = tf.keras.layers.Dense(units=int(FLAGS.d_hidden2 / 4), activation='relu')(self.fea_cla_d)
        self.pred_class = tf.keras.layers.Dense(units=14)(self.pred_class0)
        self.reconstructions_d = InnerProductDecoder(input_dim=FLAGS.d_hidden2,
                                                     act=lambda x: x,
                                                     logging=self.logging)(self.embeddings_d)
        self.reconstructions_m = InnerProductDecoder(input_dim=FLAGS.d_hidden2,
                                                     act=lambda x: x,
                                                     logging=self.logging)(self.embeddings_m)
        self.reconstructions_dm, self.weights_ = InnerProductDecoderMF2(emb_d=self.embeddings_d,
                                                        emb_m=self.embeddings_m,
                                                        act=lambda x: x,
                                                        logging=self.logging)(self.embeddings_m)
        for x1 in range(len(self.tr_e_p)):
            self.pred_dm.append(self.reconstructions_dm[self.tr_e_p[x1][0], self.tr_e_p[x1][1]])
        for x2 in range(len(self.tr_e_n)):
            self.pred_dm.append(self.reconstructions_dm[self.tr_e_n[x2][0], self.tr_e_n[x2][1]])
        self.preds_dm = self.pred_dm

