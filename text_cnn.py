import sys
sys.path.append('/scratch/cluster/yezhang/influence-release')
import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_ncg
import os
import time

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, train_dir=None, l2_reg_lambda=0.0, batch_size=100, damping=0.0,
            mini_batch=True, model_name='CNN',session=None):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = batch_size
        self.num_classes = num_classes
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.01, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            self.softmax_W = W
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            self.softmax_b = b
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.probabilies = tf.nn.softmax(self.scores)
            self.log_probabilies = tf.nn.log_softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        self.params = self.get_all_params()
        # CalculateMean cross-entropy loss
        with tf.variable_scope("loss"):
            labels = tf.one_hot(self.input_y, depth=self.num_classes)
            cross_entropy = -tf.reduce_sum(tf.multiply(labels, tf.nn.log_softmax(self.scores)), axis=1)
            self.entropy = -tf.reduce_sum(tf.multiply(self.log_probabilies, self.probabilies), axis=1)
            self.average_entropy = tf.reduce_mean(self.entropy)
            self.indiv_loss_no_reg = cross_entropy
            self.loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)  #average grad loss
            self.loss = self.loss_no_reg + l2_reg_lambda * l2_loss   #average loss
            self.EGL_norm = tf.reshape(tf.norm(tf.gradients(self.loss_no_reg, self.W)[0],axis=-1),[-1]) #batch_size (should be 1) * |V|

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(tf.cast(self.predictions, "int32"), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        config = tf.ConfigProto()
        if session is None:
            self.sess = tf.Session(config=config)
        else:
            self.sess = session
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.damping = damping
        self.grad_total_loss_op = tf.gradients(self.loss, self.params)
        self.mini_batch = mini_batch
        self.train_dir = train_dir
        self.model_name = model_name
        if self.train_dir is not None:
            if not os.path.exists(self.train_dir):
                os.makedirs(self.train_dir)

    def get_all_params(self):
        trainable_vars = tf.trainable_variables()
        trainable_vars = [t for t in trainable_vars if "embedding" not in t.name
                          and 'conv' not in t.name
         ]
        print "params used in Hessian: "
        for t in trainable_vars:
            print (t.name)
            print (t.shape)
        return trainable_vars

    def re_initialize(self, checkpoint_dir):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        print ("checkpoint file: ", checkpoint_file)
        self.saver.restore(self.sess, checkpoint_file)
