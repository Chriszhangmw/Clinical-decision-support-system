#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

#config of model
rnn_size = 128
num_layers = 2
label_size = 2
seq_length = 700



# 最大下采样操作
def max_pool(name, l_input):
    return tf.nn.max_pool(l_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

class Model():
    def __init__(self,embeddings,istrain = False):
        tf.reset_default_graph()
        self.istrain = istrain
        self.embeddings = embeddings
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size)
        self.cell  = rnn.MultiRNNCell([cell] * num_layers)
        self.input_data = tf.placeholder(tf.int32, [None, seq_length],name = 'inputs')
        print('self_input shape :',self.input_data.get_shape())
        self.targets = tf.placeholder(tf.int64, [None, ])  # target is class label
        print('satrt caluating embedding')
        with tf.variable_scope('embeddingLayer'):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=True,
                                           name="_word_embeddings")
            print('_word_embeddings shape', _word_embeddings.get_shape())
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.input_data,
                                                     name="word_embeddings")
            print('word_embeddings shape',word_embeddings.get_shape())
            x = tf.expand_dims(word_embeddings, 3)
            print('inputs shape:', x.get_shape())
            kernel = tf.Variable(tf.truncated_normal([11, 11, 1, 32], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(x, kernel, [1, 2, 2, 1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias)
            print('conv1 shape:', conv1.get_shape())
            pool1 = max_pool('pool1', conv1)
            print('pool1 shape:', pool1.get_shape())
            norm1 = norm('norm1', pool1, lsize=4)

            kernel2 = tf.Variable(tf.truncated_normal([11, 11, 32, 64], dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            conv2 = tf.nn.conv2d(norm1, kernel2, [1, 2, 2, 1], padding='VALID')
            biases2 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                  trainable=True, name='biases')
            bias2 = tf.nn.bias_add(conv2, biases2)
            conv2 = tf.nn.relu(bias2)
            print('conv2 shape:', conv2.get_shape())
            pool2 = max_pool('pool2', conv2)
            print('pool2 shape:', pool2.get_shape())
            norm2 = norm('norm2', pool2, lsize=4)

            kernel3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            conv3 = tf.nn.conv2d(norm2, kernel3, [1, 2, 2, 1], padding='VALID')
            biases3 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                  trainable=True, name='biases')
            bias3 = tf.nn.bias_add(conv3, biases3)
            conv3 = tf.nn.relu(bias3)
            print('conv3 shape:', conv3.get_shape())
            self.output2 = tf.reshape(conv3, [-1, 18, 128], name='lr_input')
            def cnn_net(x):
                # x = tf.reshape(np.array(x),(-1,150,50,1))
                x = tf.expand_dims(x, 3)
                print('inputs shape:',x.get_shape())
                kernel = tf.Variable(tf.truncated_normal([11, 11, 1, 32], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(x, kernel, [1,2,2,1], padding='VALID')
                biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                     trainable=True, name='biases')
                bias = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(bias)
                print('conv1 shape:',conv1.get_shape())
                pool1 = max_pool('pool1', conv1)
                print('pool1 shape:', pool1.get_shape())
                norm1 = norm('norm1', pool1, lsize=4)

                kernel2 = tf.Variable(tf.truncated_normal([11, 11, 32, 64], dtype=tf.float32,
                                                          stddev=1e-1), name='weights')
                conv2 = tf.nn.conv2d(norm1, kernel2, [1, 2, 2, 1], padding='VALID')
                biases2 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                      trainable=True, name='biases')
                bias2 = tf.nn.bias_add(conv2, biases2)
                conv2 = tf.nn.relu(bias2)
                print('conv2 shape:', conv2.get_shape())
                pool2 = max_pool('pool2', conv2)
                print('pool2 shape:', pool2.get_shape())
                norm2 = norm('norm2', pool2, lsize=4)

                kernel3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], dtype=tf.float32,
                                                          stddev=1e-1), name='weights')
                conv3 = tf.nn.conv2d(norm2, kernel3, [1, 2, 2, 1], padding='VALID')
                biases3 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                      trainable=True, name='biases')
                bias3 = tf.nn.bias_add(conv3, biases3)
                conv3 = tf.nn.relu(bias3)
                print('conv3 shape:', conv3.get_shape())
                conv3 = tf.reshape(conv3, [-1, 18, 128],name='lr_input')

                return conv3
            # word_embeddings = cnn_net(word_embeddings)
            # self.output2 = word_embeddings
            # print('embedded shape:', word_embeddings.get_shape())
            inputs = tf.split(self.output2, 18, 1)
            print('after split:',np.array(inputs).shape)
            print(inputs)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            print('after squeeze:', np.array(inputs).shape)
            print(inputs)


        outputs, last_state = rnn.static_rnn(self.cell, inputs, dtype=tf.float32, scope='rnnLayer')
        print('outputs****',outputs)
        print(len(outputs))
        outputs_ = np.array(outputs[-1])
        print(outputs_)
        self.output3 = tf.reshape(outputs[-1], [-1, 128, 1], name='lr_input2')
        print('outputs shape : ',outputs[-1].shape)
        print('self.output3  shape:',self.output3.get_shape)

        print('finished caluating embedding')

        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [rnn_size, label_size])
            softmax_b = tf.get_variable('b', [label_size])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits,name = 'y')
            print('self.probs:',self.probs.get_shape())

        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.targets))  # Softmax loss
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.targets))  # Softmax loss
        tf.summary.scalar('loss', self.cost)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)  # Adam Optimizer

        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    # def predict_class(self, sess, text):
    #     x = np.array(text)
    #     feed = {self.input_data: x}
    #     probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)
    #     results = np.argmax(probs, 1)
    #     return results
