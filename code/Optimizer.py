import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class CoOptimizerAE_MF(object):
    def __init__(self, preds_d, labels_d, preds_m, labels_m, preds_dm, labels_dm, pred_cla, multilabels,
                 a_d, a_m, a_cla, pos_weight_d, norm_d, pos_weight_m, norm_m):

        self.cost_recon_d = norm_d * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_d, targets=labels_d, pos_weight=pos_weight_d))
        self.cost_recon_m = norm_m * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_m, targets=labels_m, pos_weight=pos_weight_m))
        self.cost_recon_dm = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds_dm, labels=labels_dm))

        self.label_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_cla, labels=multilabels))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.loss = a_d * self.cost_recon_d + a_m * self.cost_recon_m + self.cost_recon_dm + a_cla * self.label_cost

        self.opt_op = self.optimizer.minimize(self.loss)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)
        self.preds_dm = preds_dm
        self.labels_dm = labels_dm


