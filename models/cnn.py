#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午8:32
import tensorflow as tf


class QaCNN(object):
    def __init__(self, q_length, a_length, word_embeddings, filter_sizes, num_filters, margin, l2_reg_lambda):

        self.question = tf.placeholder(tf.int32, [None, q_length], name='question')
        self.answer = tf.placeholder(tf.int32, [None, a_length], name='pos_answer')
        self.label = tf.placeholder(tf.int32, [None], name='label')

        vocab_size, embedding_size = word_embeddings.shape
        num_filters_total = num_filters * len(filter_sizes)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embeddings = tf.get_variable("embeddings",
                                              shape=word_embeddings.shape,
                                              initializer=tf.constant_initializer(word_embeddings),
                                              trainable=False)
            self.embedded_q = tf.nn.embedding_lookup(self.embeddings, self.question)
            self.embedded_a = tf.nn.embedding_lookup(self.embeddings, self.answer)
            self.embedded_q_expanded = tf.expand_dims(self.embedded_q, -1)
            self.embedded_a_expanded = tf.expand_dims(self.embedded_a, -1)

        # conv-pool-drop for question
        pooled_q_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope('ques-conv-pool-{}'.format(filter_size)):
                # convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedded_q_expanded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h, ksize=[1, q_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
                pooled_q_outputs.append(pooled)

        self.q_h_pool = tf.reshape(tf.concat(pooled_q_outputs, 3), [-1, num_filters_total])

        # conv-pool-drop for positive answer
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope('answ-conv-pool-{}'.format(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                # convolution layer
                conv = tf.nn.conv2d(self.embedded_a_expanded, W,
                                        strides=[1, 1, 1, 1], padding='VALID', name='pos-conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='pos-relu')
                # max pooling layer
                pooled = tf.nn.max_pool(h, ksize=[1, a_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1],
                                            padding='VALID', name='pos-pool')
                pooled_outputs.append(pooled)

        self.h_pool = tf.reshape(tf.concat(pooled_outputs, 3), [-1, num_filters_total])

        with tf.name_scope('similarity'):
            normalized_q_h_pool = tf.nn.l2_normalize(self.q_h_pool, axis=1)
            normalized_h_pool = tf.nn.l2_normalize(self.h_pool, axis=1)
            self.similarity = tf.reduce_sum(tf.multiply(normalized_q_h_pool, normalized_h_pool), 1)

        with tf.name_scope('loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.label, tf.float32), logits=self.similarity)
            self.loss = tf.reduce_mean(losses)

        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

