from __future__ import print_function
""" input data preprocess.
"""

import tensorflow
import numpy as np
import tool

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


data_prefix = '../data/nlu_data/'
word2vec_path = data_prefix+'glove.6B.300d.txt'
# word2vec_path = data_prefix+'wiki.en.vec'
# training_data_path = data_prefix + 'train_shuffle.txt'
# test_data_path = data_prefix + 'test.txt'

training_data_path = '/dfsdata2/hugy3_data/lena_moli/lena_train_shuffle_ts.txt'
test_data_path = '/dfsdata2/hugy3_data/lena_moli/moli_train_shuffle_ts.txt'

seen_intent = ['Audio_Issues', 'Wireless Internet Connection Issue', 'Startup Period No Power or No Boot or No Display', 'SIM Card', 'MotherBoard_Issue', 'SD Card', 'Operating_System', 'Operating System', 'BIOS', 'Heat_Fan', 'Networking_Connection', 'Optical Disk Drive', 'Fingerprint', 'Diagnosis Result', 'Camera_Issues', 'Lenovo Vantage', 'Display', 'Wired Network Connection Issue', 'Storage_Hard_Drive', 'Battery', 'AC_Adapter', 'Mouse_TrackPoint_Touchpad', 'Keyboard', 'Appearance', 'Faulty Docking Station', 'Port']
unseen_intent = ['os and apps', 'repair issue', 'activation', 'power and charging', 'connectivity', 'audio and sound', 'call', 'network issue', 'screen and display', 'camera issue', 'parts and accessories']

# seen_intent = ['music', 'search', 'movie', 'weather', 'restaurant']
# unseen_intent = ['playlist', 'book']

# unseen_intent = ['connectivity', 'audio and sound']
# unseen_intent = ['Operating System', 'Port']

# seen_intent = ['Troubleshooting_Technical_Issues','How_to_configuration']
# unseen_intent = ['playlist', 'book']
# unseen_intent = ['Trouble Shooting', 'How to']
# seen_intent = ['playlist', 'book', 'movie', 'weather', 'restaurant']
# seen_intent = ['music', 'search']
# unseen_intent = ['music', 'search']

# seen_intent = ['playlist', 'book', 'movie', 'music', 'search']
# unseen_intent = ['weather', 'restaurant']

# seen_intent = ['playlist', 'book', 'weather', 'restaurant', 'search']
# unseen_intent = ['movie', 'music']

# seen_intent = ['music', 'search', 'movie', 'book', 'restaurant']
# unseen_intent = ['playlist', 'weather']

# seen_intent = ['playlist', 'book', 'movie', 'weather', 'search']
# unseen_intent = ['music', 'restaurant']

# seen_intent = ['playlist', 'book', 'restaurant', 'music', 'search']
# unseen_intent = ['weather', 'movie']

# seen_intent = ['playlist', 'weather', 'restaurant', 'music', 'movie']
# unseen_intent = ['book', 'search']

# seen_intent = ['book', 'weather', 'restaurant', 'music', 'movie']
# unseen_intent = ['playlist', 'search']

# seen_intent = ['playlist', 'weather', 'movie', 'search']
# unseen_intent = ['book', 'restaurant', 'music']

# seen_intent = ['book', 'restaurant', 'music', 'search']
# unseen_intent = ['playlist', 'weather', 'movie']

# seen_intent = ['book', 'restaurant', 'weather', 'search']
# unseen_intent = ['playlist', 'music', 'movie']

def load_w2v(file_name):
    """ load w2v model
        input: model file name
        output: w2v model
    """
#     glove2word2vec(file_name, 'glove.6B.300d.txt.tmp')
    w2v = KeyedVectors.load_word2vec_format(
            'glove.6B.300d.txt.tmp', binary=False, unicode_errors='ignore')
#     w2v = KeyedVectors.load_word2vec_format(
#             file_name, binary=False, unicode_errors='ignore')
    return w2v

def process_label(intents, w2v):
    """ pre process class labels
        input: class label file name, w2v model
        output: class dict and label vectors
    """
    class_dict = {}
    label_vec = []
    class_id = 0
    for line in intents:
        # check whether all the words in w2v dict
        if line == 'Troubleshooting_Technical_Issues':
            line = 'Trouble_shooting_Technical_Issues'
        if '_' in line:
            label = [w.lower().strip() for w in line.split('_')]
        else:
            label = [w.lower().strip() for w in line.split(' ')]
        for w in label:
            if not w2v.vocab.has_key(w.lower()):
                print("not in w2v dict", w)
        # compute label vec
        label_sum = np.sum([w2v[w] for w in label], axis = 0)
        label_vec.append(label_sum)
        # store class names => index
        class_dict[' '.join(label)] = class_id
        class_id = class_id + 1
    return class_dict, np.asarray(label_vec)

def load_vec(file_path, w2v, class_dict, in_max_len):
    """ load input data
        input:
            file_path: input data file
            w2v: word2vec model
            max_len: max length of sentence
        output:
            input_x: input sentence word ids
            input_y: input label ids
            s_len: input sentence length
            max_len: max length of sentence
    """
    input_x = [] # input sentence word ids
    input_y = [] # input label ids

    s_len = [] # input sentence length
    max_len = 0

    for line in open(file_path):
        arr = line.strip().split('\t')
        if len(arr) < 2:
            continue
        if '_' in arr[0]:
            label = [w.lower().strip() for w in arr[0].split('_')]
        else:
            label = [w.lower().strip() for w in arr[0].split(' ')]
         
        question = [w for w in arr[1].split(' ')]
        cname = ' '.join(label) 
        if not class_dict.has_key(cname):
            continue

        # trans words into indexes
        x_arr = []
        for w in question:
            if w2v.vocab.has_key(w):
                x_arr.append(w2v.vocab[w].index)
        s_l = len(x_arr)
        if s_l <= 1:
            continue
        if in_max_len == 0:
            if s_l > max_len:
                max_len = len(x_arr)

        input_x.append(np.asarray(x_arr))
        input_y.append(class_dict[cname])  
        s_len.append(s_l)

    # add paddings
    max_len = max(in_max_len, max_len)
    x_padding = []
    for i in range(len(input_x)):
        if (max_len < s_len[i]):
            x_padding.append(input_x[i][0:max_len])
            continue
        tmp = np.append(input_x[i], np.zeros((max_len - s_len[i],), dtype=np.int64))
        x_padding.append(tmp)
    x_padding = np.asarray(x_padding)
    input_y = np.asarray(input_y)
    s_len = np.asarray(s_len)
    return x_padding, input_y, s_len, max_len

def get_label(data):
    Ybase = data['y_tr']
    sample_num = Ybase.shape[0]
    labels = np.unique(Ybase)
    class_num = labels.shape[0]
    labels = range(class_num)
    # get label index
    ind = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        ind[Ybase == labels[i], i] = 1;
    return ind

def read_datasets():
    print("------------------read datasets begin-------------------")
    data = {}

    # load word2vec model
    print("------------------load word2vec begin-------------------")
    w2v = load_w2v(word2vec_path)
    print("------------------load word2vec end---------------------")

    # load normalized word embeddings
    embedding = w2v.syn0
    data['embedding'] = embedding
    norm_embedding = tool.norm_matrix(embedding)
    data['embedding'] = norm_embedding
    # pre process seen and unseen labels
    sc_dict, sc_vec = process_label(seen_intent, w2v)
    print("sc_dict", sc_dict)
    uc_dict, uc_vec = process_label(unseen_intent, w2v)

    # trans data into embedding vectors
    max_len = 0
    x_tr, y_tr, s_len, max_len = load_vec(
            training_data_path, w2v, sc_dict, max_len)
    x_te, y_te, u_len, max_len = load_vec(
            test_data_path, w2v, uc_dict, max_len)
    data['x_tr'] = x_tr
    data['y_tr'] = y_tr

    data['s_len'] = s_len
    data['sc_vec'] = sc_vec
    data['sc_dict'] = sc_dict

    data['x_te'] = x_te
    data['y_te'] = y_te

    data['u_len'] = u_len
    data['uc_vec'] = uc_vec
    data['uc_dict'] = uc_dict

    data['max_len'] = max_len

    ind = get_label(data)
    data['s_label'] = ind # [0.0, 0.0, ..., 1.0, ..., 0.0]
    print("------------------read datasets end---------------------")
    return data
