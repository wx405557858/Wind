__author__ = 'shawn'

import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


flag = tf.app.flags
FLAGS = flag.FLAGS
flag.DEFINE_integer('batch_size', 32, 'traing batch size')
flag.DEFINE_integer('hidden_size', 64, 'hidden layer size')
flag.DEFINE_integer('length', 7*24*6, 'input timestamp length')
flag.DEFINE_integer('n_iter', 100, 'iteration number')

def weight_variable(shape):
    init = tf.truncated_normal(shape)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def lstm_op(lstm, x, state):
    with tf.variable_scope('lstm') as scope:
        try:
            output, state = lstm(x, state)
        except:
            scope.reuse_variables()
            output, state = lstm(x, state)
    return output, state

def dense(x, shape):
    W = weight_variable(shape)
    b = bias_variable(shape[-1:])
    return tf.matmul(x, W) + b

class LSTMLayer(object):
    def __init__(self, length=None):
        # global configuration
        flag = tf.app.flags.FLAGS

        # update parameter
        if length:
            flag.length = length

        # create network structure
        self.create_network()

        # create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)



    def create_network(self):
        flag = tf.app.flags.FLAGS
        self.x = tf.placeholder(tf.float32, shape=[flag.batch_size, flag.length])
        self.y = tf.placeholder(tf.float32, shape=[flag.batch_size, 1])

        lstm = rnn.BasicLSTMCell(flag.hidden_size)
        # lstm state
        init_state =lstm.zero_state(flag.batch_size)
        state = self.init_state

        for i in range(flag.length):
            hidden, state = lstm_op(lstm, self.x[:, i], state)

        self.y_ = dense(hidden, [flag.hidden_size, 1])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def fit(self, X, Y):
        flag = tf.app.flags.FLAGS
        sess = self.sess

        train_x, test_x = X
        train_y, test_y = Y

        # generate data
        def data_generator(x, y):
            while True:
                for i in range(0, len(x)-flag.batch_size, flag.batch_size):
                    yield (x[i:i+flag.batch_size], y[i:i+flag.batch_size])

        dg = data_generator(train_x, train_y)
        losses = []
        for iter in range(flag.n_iter):
            # training feed_dict
            x, y = next(dg)
            feed_dict = {
                self.x : x,
                self.y : y,
            }

            # training
            loss = sess.run(self.train_step, feed_dict=feed_dict)
            losses.append(loss)

            sys.stdout.write('\b')
            print "iteration %d, train rmse % .6lf" % (iter, np.mean(losses)**0.5)

            # evaluate
            if iter % 1000 == 0:
                losses = []
                loss_eval = self.evaluate(test_x, test_y)
                print "iteration %d, test rmse %.6lf" % (iter, loss_eval**0.5)



    def evaluate(self, test_x, test_y):
        flag = tf.app.flags.FLAGS

        losses = 0
        count = 0

        for i in range(0, len(test_x) - flag.batch_size, flag.batch_size):
            x = test_x[i, i + flag.batch_size]
            y = test_y[i, i + flag.batch_size]
            feed_dict = {
                self.x: x,
                self.y: y
            }
            losses += self.sess.run(self.loss, feed_dict=feed_dict)
            count += 1
        return losses / count