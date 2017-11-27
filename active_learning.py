#! /usr/bin/env python
#This code is modified from https://github.com/dennybritz/cnn-text-classification-tf
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import cPickle
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_string("train_data_file", "train.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/train_pos.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/train_neg.txt", "Data source for the negative data.")
tf.flags.DEFINE_string("test_positive_data_file", "./data/rt-polaritydata/test_pos.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("test_negative_data_file", "./data/rt-polaritydata/test_neg.txt", "Data source for the positive data.")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 25, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("word2vec_path", "/scratch/cluster/yezhang/glove.840B.300d.txt", "word2ec path")
#active learning method
tf.flags.DEFINE_string("AL_method", "EGL", "active learning method. Should be random/entropy/EGL")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

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

x = cPickle.load(open("mr.p","rb"))
revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
vocab_size = W.shape[0]
idx_to_word_map = idx_to_word(word_idx_map)
print "data loaded!"
initW = W
average_accuracy_across_folds = []
for k in range(10):
    print "test on fold: " + str(k)
    datasets = make_idx_data_cv(revs, word_idx_map, k, max_l=56,k=300, filter_h=5)
    img_h = len(datasets[0][0]) - 1
    x_train, y_train = datasets[0][:,:-1], datasets[0][:,-1]
    x_test, y_test = datasets[1][:,:-1], datasets[1][:,-1]


    with tf.Graph().as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
        )
    # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Initialize all variables
        cnn.sess.run(tf.global_variables_initializer())
        cnn.sess.run(cnn.W.assign(initW))


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = cnn.sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, train loss {:g}, train acc {:g}".format(time_str, step, loss, accuracy))

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = cnn.sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, dev loss {:g}, dev acc {:g}".format(time_str, step, loss, accuracy))
            return accuracy

        def entropy_score(train_x, train_index):
            feed_dict = {
                cnn.input_x: train_x[train_index],
                cnn.dropout_keep_prob: 1.0
            }
            step, test_entropies = cnn.sess.run(
                [global_step, cnn.entropy],
                feed_dict)
            return test_entropies

        def EGL_score(train_x, train_index):
            EGL_scores = []
            for j in train_index:
                EGL_norm = np.zeros(vocab_size)
                for k in range(num_classes):
                    feed_dict = {
                        cnn.input_x: np.expand_dims(train_x[j],0),
                        cnn.dropout_keep_prob: 1.0,
                        cnn.input_y: np.array([k])
                    }
                    step, EGL, probs = cnn.sess.run(
                        [global_step, cnn.EGL_norm, cnn.probabilies],
                        feed_dict)
                    EGL_norm += EGL * probs[0][k]
                EGL_max_norm = max(EGL_norm)
                EGL_scores.append(EGL_max_norm)
            return np.array(EGL_scores)


        print "number of training points in fold " + str(k) + ":"+ str(len(y_train))
        print "number of test points in fold " + str(k) + ":" + str(len(y_test))
        indices = np.arange(len(y_train))
        num_classes = len(set(y_train))
        best_dev_accuracy = 0.0
        index_in_labels_pool = []
        index_in_unlabeled_pool = indices
        index_of_new_add_index = np.random.choice(np.arange(len(index_in_unlabeled_pool)), size=FLAGS.batch_size)
        index_in_labels_pool += list(index_in_unlabeled_pool[index_of_new_add_index])
        index_in_unlabeled_pool = np.delete(index_in_unlabeled_pool, index_of_new_add_index)
        cur_train = x_train[np.array(index_in_labels_pool)]
        cur_labels = y_train[np.array(index_in_labels_pool)]
        accuracy_list = []
        init = tf.global_variables_initializer()
        cnn.sess.run(init)
        cnn.sess.run(cnn.W.assign(initW))
        for i in range(20):
            print "current number of labels: ", len(cur_labels)
            print "current positive labels: ", len(np.where(cur_labels == 1)[0])
            print "current negative labels: ", len(np.where(cur_labels == 0)[0])
            print "current number of unlabeled points: ", len(index_in_unlabeled_pool)
            batches = data_helpers.batch_iter(
                    list(zip(cur_train, cur_labels)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                cnn.sess.run(cnn.W[0].assign(np.zeros(FLAGS.embedding_dim)))
            if FLAGS.AL_method == "entropy":
                entropy_scores = entropy_score(x_train, index_in_unlabeled_pool)
                index_of_new_add_index = np.argsort(entropy_scores)[-FLAGS.batch_size:]
            elif FLAGS.AL_method == "random":
                index_of_new_add_index = np.random.choice(np.arange(len(index_in_unlabeled_pool)), size=FLAGS.batch_size)
            elif FLAGS.AL_method == "EGL":
                EGL_scores = EGL_score(x_train, index_in_unlabeled_pool)
                index_of_new_add_index = np.argsort(EGL_scores)[-FLAGS.batch_size:]
            index_in_labels_pool += list(index_in_unlabeled_pool[index_of_new_add_index])
            index_in_unlabeled_pool = np.delete(index_in_unlabeled_pool, index_of_new_add_index)
            cur_train = x_train[np.array(index_in_labels_pool)]
            cur_labels = y_train[np.array(index_in_labels_pool)]
            dev_accuracy = dev_step(x_test, y_test)
            print dev_accuracy
            accuracy_list.append(dev_accuracy)
        print accuracy_list
        average_accuracy_across_folds.append(accuracy_list)
print list(np.average(np.array(average_accuracy_across_folds), axis=0))
