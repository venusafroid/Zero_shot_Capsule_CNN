import os
from random import *
import input_data_ml_ht as input_data
import numpy as np
import tensorflow as tf
import model_cnn
import tool
import math
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import time
from functools import reduce
from operator import mul

a = Random();
a.seed(1)

def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_time = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    s_cnum = np.unique(data['y_tr']).shape[0]
    u_cnum = np.unique(data['y_te']).shape[0]

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_float("keep_prob", 0.8, "embedding dropout keep rate")
    tf.app.flags.DEFINE_integer("hidden_size", 32, "embedding vector size")
    tf.app.flags.DEFINE_integer("batch_size", 64, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("num_epochs", 200, "num of epochs")
    tf.app.flags.DEFINE_integer("vocab_size", vocab_size, "vocab size of word vectors")
    tf.app.flags.DEFINE_integer("max_time", max_time, "max number of words in one sentence")
    tf.app.flags.DEFINE_integer("sample_num", sample_num, "sample number of training data")
    tf.app.flags.DEFINE_integer("test_num", test_num, "number of test data")
    tf.app.flags.DEFINE_integer("s_cnum", s_cnum, "seen class num")
    tf.app.flags.DEFINE_integer("u_cnum", u_cnum, "unseen class num")
    tf.app.flags.DEFINE_integer("word_emb_size", word_emb_size, "embedding size of word vectors")
    tf.app.flags.DEFINE_string("ckpt_dir", './cnn_saved_models/' , "check point dir")
    tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
    tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
    tf.app.flags.DEFINE_float("sim_scale", 4, "sim scale")
    tf.app.flags.DEFINE_float("margin", 1.0, "ranking loss margin")
    tf.app.flags.DEFINE_float("alpha", 0.0001, "coefficient for self attention loss")
    tf.app.flags.DEFINE_integer("num_routing", 2, "capsule routing num")
    tf.app.flags.DEFINE_integer("output_atoms", 10, "capsule output atoms")
    tf.app.flags.DEFINE_boolean("cnn_save_model", False, "save model to disk")
    tf.app.flags.DEFINE_integer("d_a", 20, "self attention weight hidden units number")
    tf.app.flags.DEFINE_integer("r", 3, "self attention weight hops")
    # configs about cnn
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    return FLAGS

def get_sim(data):
    # get unseen and seen categories similarity
    s = normalize(data['sc_vec'])
    u = normalize(data['uc_vec'])
    sim = tool.compute_label_sim(u, s, FLAGS.sim_scale)
    return sim

def evaluate_test(data, FLAGS, sess):
    # zero-shot testing state
    # seen votes shape (110, 2, 34, 10)
    x_te = data['x_te']
    y_te_id = data['y_te']
    u_len = data['u_len']

    # get unseen and seen categories similarity
    # sim shape (8, 34)
    sim_ori = get_sim(data)
    total_unseen_pred = np.array([], dtype=np.int64)

#     batch_size  = FLAGS.test_num
    batch_size  = FLAGS.batch_size
    test_batch = int(math.ceil(FLAGS.test_num / float(batch_size)))
    #test_batch = int(math.ceil(FLAGS.test_num / float(FLAGS.batch_size)))
    for i in range(test_batch):
        begin_index = i * batch_size
        end_index = min((i + 1) * batch_size, FLAGS.test_num)
        batch_te = x_te[begin_index : end_index]
        batch_id = y_te_id[begin_index : end_index]
        batch_len = u_len[begin_index : end_index]

        [seen_logits] = sess.run([lstm.logits],
            feed_dict={lstm.input_x: batch_te, lstm.s_len: batch_len})
        
        
#         sim = tf.expand_dims(sim_ori, [0])
#         print(sim_ori.shape)
#         sim = tf.tile(sim, [seen_logits.shape[0],1,1])

#         seen_logits = tf.expand_dims(seen_logits, [1])
#         unseen_logits = tf.matmul(seen_logits, tf.transpose(sim, perm=[0, 2, 1]))
#         unseen_logits = tf.squeeze(unseen_logits, [1])
#         with tf.Session() as sess:
#             unseen_logits = unseen_logits.eval()
        te_batch_pred = np.argmax(seen_logits, 1)
        total_unseen_pred = np.concatenate((total_unseen_pred, te_batch_pred))

    print "zero-shot intent detection test set performance"
    acc = accuracy_score(y_te_id, total_unseen_pred)
    print classification_report(y_te_id, total_unseen_pred, digits=4)
    return acc


def generate_batch(n, batch_size):
    batch_index = a.sample(xrange(n), batch_size)
    return batch_index

def assign_pretrained_word_embedding(sess, data, textRNN):
    print("using pre-trained word emebedding.begin...")
    embedding = data['embedding']

    word_embedding = tf.constant(embedding, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding_word,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("using pre-trained word emebedding.ended...")

def squash(input_tensor):
    norm = tf.norm(input_tensor, axis=2, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

def update_unseen_routing(votes, FLAGS, num_routing=3):
    votes_t_shape = [3, 0, 1, 2]
    r_t_shape = [1, 2, 3, 0]
    votes_trans = tf.transpose(votes, votes_t_shape)
    num_dims = 4
    input_dim = FLAGS.r
    output_dim = FLAGS.u_cnum
    input_shape = tf.shape(votes)
    logit_shape = tf.stack([input_shape[0], input_dim, output_dim])

    def _body(i, logits, activations, route):
        route = tf.nn.softmax(logits, dim=2)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1)
        activation = squash(preactivate)
        activations = activations.write(i, activation)

        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations, route)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    route = tf.nn.softmax(logits, dim=2)
    _, logits, activations, route = tf.while_loop(
        lambda i, logits, activations, route: i < num_routing,
        _body,
        loop_vars=[i, logits, activations, route],
        swap_memory=True)

    return activations.read(num_routing - 1), route

def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            p = reduce(mul, [dim.value for dim in shape], 1)
            print(variable.name,p)
            num_params += p
        return num_params

if __name__ == "__main__":
    # load data
    data = input_data.read_datasets()
    x_tr = data['x_tr']
    y_tr = data['y_tr']
    y_tr_id = data['y_tr']
    y_te_id = data['y_te']
    y_ind = data['s_label']
    s_len = data['s_len']
    embedding = data['embedding']

    x_te = data['x_te']
    u_len = data['u_len']

    # load settings
    FLAGS = setting(data)

    # start
    tf.reset_default_graph()
    config=tf.ConfigProto()
    with tf.Session(config=config) as sess:
        # Instantiate Model
        lstm = model_cnn.cnn_model(FLAGS)

#         if os.path.exists(FLAGS.ckpt_dir):
#             print("Restoring Variables from Checkpoint for rnn model.")
#             saver = tf.train.Saver()
#             saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
#         else:
        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())
        if FLAGS.use_embedding: #load pre-trained word embedding
            assign_pretrained_word_embedding(sess, data, lstm)

        best_acc = 0.0
        best_epoch = 0
        cur_acc = evaluate_test(data, FLAGS, sess)
        if cur_acc > best_acc:
            best_acc = cur_acc
        var_saver = tf.train.Saver()

        # Training cycle
        batch_num = FLAGS.sample_num / FLAGS.batch_size
        for epoch in range(FLAGS.num_epochs):
            total_y_id = np.array([], dtype=np.int64)
            total_seen_pred = np.array([], dtype=np.int64)
            for batch in range(batch_num):
                batch_index = generate_batch(FLAGS.sample_num, FLAGS.batch_size)
                batch_x = x_tr[batch_index]
                batch_y_id = y_tr_id[batch_index]
                batch_len = s_len[batch_index]
                batch_ind = y_ind[batch_index]

                [_, loss, logits] = sess.run([lstm.train_op, lstm.loss_val, lstm.logits],
                        feed_dict={lstm.input_x: batch_x, lstm.IND: batch_ind, lstm.s_len: batch_len})
                tr_batch_pred = np.argmax(logits, 1)
                total_seen_pred = np.concatenate((total_seen_pred, tr_batch_pred))
                total_y_id = np.concatenate((total_y_id, batch_y_id))
            train_acc = accuracy_score(total_y_id, total_seen_pred)
            print("train_acc", train_acc)

            print "------------------epoch : ", epoch, " Loss: ", loss, "----------------------"
            cur_acc = evaluate_test(data, FLAGS, sess)
            if cur_acc > best_acc:
                # save model
                best_acc = cur_acc
                best_epoch = epoch
                var_saver.save(sess, os.path.join(FLAGS.ckpt_dir, "model.ckpt"), 1)
            print("cur_acc", cur_acc)
            print("best_acc", best_acc, "best_epoch", best_epoch)

