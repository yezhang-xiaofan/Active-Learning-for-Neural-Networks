import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    original_text = x_text
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    #positive_labels = [[0, 1] for _ in positive_examples]
    #negative_labels = [[1, 0] for _ in negative_examples]
    positive_labels = [1] * len(positive_examples)
    negative_labels = [0] * len(negative_examples)
    #y = np.concatenate([positive_labels, negative_labels], 0)
    y = positive_labels + negative_labels
    return x_text, y, original_text


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def split_into_train_test(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r").readlines())
    negative_examples = list(open(negative_data_file, "r").readlines())
    num_pos = len(positive_examples)
    num_neg = len(negative_examples)
    shuffle_indices = np.random.permutation(np.arange(num_pos))
    dev_sample_index = -1 * int(0.1 * float(num_pos))
    pos_train_indices = shuffle_indices[:dev_sample_index]
    #pos_test_indices = shuffle_indices[dev_sample_index:]
    pos_train = open('data/rt-polaritydata/train_pos.txt','wb')
    pos_test = open('data/rt-polaritydata/test_pos.txt','wb')
    neg_train = open('data/rt-polaritydata/train_neg.txt','wb')
    neg_test = open('data/rt-polaritydata/test_neg.txt','wb')
    for i in range(num_pos):
        if i in pos_train_indices:
            pos_train.write(positive_examples[i])
        else:
            pos_test.write(positive_examples[i])
    shuffle_indices = np.random.permutation(np.arange(num_neg))
    dev_sample_index = -1 * int(0.1 * float(num_neg))
    neg_train_indices = shuffle_indices[:dev_sample_index]
    for i in range(num_neg):
        if i in neg_train_indices:
            neg_train.write(negative_examples[i])
        else:
            neg_test.write(negative_examples[i])

def split_into_folds(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r").readlines())
    negative_examples = list(open(negative_data_file, "r").readlines())
    folds_pos = []
    folds_neg = []
    for i in range(10):
        folds_pos.append(open("data/folds/fold_pos_"+str(i), 'wb'))
        folds_neg.append(open("data/folds/fold_neg_"+str(i), 'wb'))
    for p in positive_examples:
        cv = np.random.randint(0,10)
        folds_pos[cv].write(p)
    for n in negative_examples:
        cv = np.random.randint(0,10)
        folds_neg[cv].write(n)
    for f in folds_pos:
        f.close()
    for f in folds_neg:
        f.close()

def load_folds(cv):
    train_x, train_y, train_original = [], [], []
    for i in range(10):
        if i == cv:
            pos_file = "data/folds/fold_pos_"+str(i)
            neg_file = "data/folds/fold_neg_"+str(i)
            test_x, test_y, test_original = load_data_and_labels(pos_file, neg_file)
        else:
            pos_file = "data/folds/fold_pos_" + str(i)
            neg_file = "data/folds/fold_neg_" + str(i)
            temp_train_x, temp_train_y, temp_train_original = load_data_and_labels(pos_file, neg_file)
            train_x += temp_train_x
            train_y += temp_train_y
            train_original += temp_train_original
    return train_x, train_y, train_original, test_x, test_y, test_original

def idx_to_word(word_idx_map):
    '''
    :param word_idx_map:  map word to index
    :return: map index to word (not including zero)
    '''
    result = {}
    for word in word_idx_map:
        result[word_idx_map[word]] = word
    return result

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)
        else:
            train.append(sent)
    #train is a list of sent
    #each sent is a list of indices padded with zeroes, followed by the label
    train = np.array(train,dtype='int32')
    test = np.array(test,dtype='int32')
    return [train, test]







