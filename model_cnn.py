import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

class cnn_model():
    """
        Using bidirectional LSTM to learn sentence embedding 
        for users' questions
    """

    def __init__(self, FLAGS, initializer=
            tf.contrib.layers.xavier_initializer()):
        """
            lstm class initialization
        """
        # configurations
        self.hidden_size = FLAGS.hidden_size
        self.vocab_size = FLAGS.vocab_size
        self.word_emb_size = FLAGS.word_emb_size
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.initializer = initializer
        self.s_cnum = FLAGS.s_cnum
        self.margin = FLAGS.margin
        self.keep_prob = FLAGS.keep_prob
        self.num_routing = FLAGS.num_routing
        self.output_atoms = FLAGS.output_atoms

        # parameters for self attention
        self.n = FLAGS.max_time
        self.d = FLAGS.word_emb_size
        self.d_a = FLAGS.d_a
        self.u = FLAGS.hidden_size
        self.r = FLAGS.r
        self.alpha = FLAGS.alpha

        #parameters for cnn 
        self.filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
        self.num_filters = FLAGS.num_filters

        # input data
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.n])
        self.s_len = tf.placeholder("int64", [None])
        self.IND = tf.placeholder(tf.float32, [None, self.s_cnum])
        
        self.instantiate_weights()
        self.l2_loss = tf.constant(0.0)
        self.inference()
        

        # graph
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        self.loss_val = self.loss()
        self.train_op = self.train()

    def instantiate_weights(self):
        """
            Initializer variable weights
        """
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding_word = tf.get_variable("Embedding",
                    shape=[self.vocab_size, self.word_emb_size],
                    initializer=self.initializer, trainable=False)

    def inference(self):
        """
            self attention
        """
        #shape:[None, sentence_length, embed_size]
        self.input_embed = tf.nn.embedding_lookup(
                self.Embedding_word, self.input_x, max_norm=1)
        self.embedded_chars_expanded = tf.expand_dims(self.input_embed, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.word_emb_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.n - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.s_cnum],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.s_cnum]))
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_pool_flat, W, b)
        

    def _margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        """Penalizes deviations from margin for each logit.
        Each wrong logit costs its distance to margin. For negative logits margin is
        0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
        margin is 0.4 from each side.
        Args:
            labels: tensor, one hot encoding of ground truth.
            raw_logits: tensor, model predictions in range [0, 1]
            margin: scalar, the margin after subtracting 0.5 from raw_logits.
            downweight: scalar, the factor for negative cost.
        Returns:
            A tensor with cost for each data point of shape [batch_size].
        """
        logits = raw_logits - 0.5
        positive_cost = labels * tf.cast(tf.less(logits, margin),
            tf.float32) * tf.pow(logits - margin, 2)
        negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    def loss(self):
        loss_val = self._margin_loss(self.IND, self.logits)
        loss_val = tf.reduce_mean(loss_val)
        return 1000 * loss_val + self.alpha * tf.reduce_mean(self.l2_loss)

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)
        return train_op
