#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午8:32
import tensorflow as tf
from layers import Dropout


class QaCNN(object):
    def __init__(self, q_length, a_length, word_embeddings, filter_sizes, num_filters, margin, l2_reg_lambda):

        self.question = tf.placeholder(tf.int32, [None, q_length], name='question')
        self.pos_answer = tf.placeholder(tf.int32, [None, a_length], name='pos_answer')
        self.neg_answer = tf.placeholder(tf.int32, [None, a_length], name='neg_answer')

        l2_reg_loss = tf.constant(0.0)

        vocab_size, embedding_size = word_embeddings.shape
        num_filters_total = num_filters * len(filter_sizes)

        with tf.name_scope('embedding'):
            self.embeddings = tf.get_variable("embeddings",
                                              shape=word_embeddings.shape,
                                              initializer=tf.constant_initializer(word_embeddings),
                                              trainable=True)
            self.embedded_q = tf.nn.embedding_lookup(self.embeddings, self.question)
            self.embedded_pos_a = tf.nn.embedding_lookup(self.embeddings, self.pos_answer)
            self.embedded_neg_a = tf.nn.embedding_lookup(self.embeddings, self.neg_answer)
            self.embedded_q_expanded = tf.expand_dims(self.embedded_q, -1)
            self.embedded_pos_a_expanded = tf.expand_dims(self.embedded_pos_a, -1)
            self.embedded_neg_a_expanded = tf.expand_dims(self.embedded_neg_a, -1)

        with tf.variable_scope('ai_cnn', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            with tf.variable_scope('encode'):
                Q = Dropout(Q, keep_prob=0.2, is_train=self.is_train)
                C = Dropout(C, keep_prob=0.2, is_train=self.is_train)
                Q_sequence = tf.layers.conv1d(Q, filters=200, kernel_size=3, padding='same')
                C_sequence = tf.layers.conv1d(C, filters=200, kernel_size=3, padding='same')

            with tf.variable_scope('interaction'):
                Q_ = tf.expand_dims(Q_sequence, axis=2)  # (B, L1, 1, dim)
                C_ = tf.expand_dims(C_sequence, axis=1)  # (B, 1, L2, dim)
                hQ = tf.tile(Q_, [1, 1, self.C_maxlen, 1])
                hC = tf.tile(C_, [1, self.Q_maxlen, 1, 1])
                H = tf.concat([hQ, hC], axis=-1)
                A = tf.layers.dense(H, units=200, activation=tf.tanh)  # (B, L1, L2, dim)

                rQ = tf.reduce_max(A, axis=2)
                rC = tf.reduce_max(A, axis=1)

            with tf.variable_scope('attention'):
                # concate
                cate_f_ = tf.expand_dims(self.cate_f, axis=1)
                Q_m = tf.concat([Q_sequence, rQ, tf.tile(cate_f_, [1, self.Q_maxlen, 1])], axis=-1)
                C_m = tf.concat([C_sequence, rC, tf.tile(cate_f_, [1, self.C_maxlen, 1])], axis=-1)

                Q_m = Dropout(Q_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train)
                C_m = Dropout(C_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train)

                Q_m = tf.layers.dense(Q_m, units=300, activation=tf.tanh, name='fw1')
                C_m = tf.layers.dense(C_m, units=300, activation=tf.tanh, name='fw1', reuse=True)

                Q_m = Dropout(Q_m, keep_prob=0.2, is_train=self._is_train)
                C_m = Dropout(C_m, keep_prob=0.2, is_train=self._is_train)

                Q_m = tf.layers.dense(Q_m, units=1, activation=tf.tanh, name='fw2')
                C_m = tf.layers.dense(C_m, units=1, activation=tf.tanh, name='fw2', reuse=True)

                Q_m -= (1 - tf.expand_dims(self.Q_mask, axis=-1)) * 1e30
                C_m -= (1 - tf.expand_dims(self.C_mask, axis=-1)) * 1e30
                Qalpha = tf.nn.softmax(Q_m, axis=1)
                Calpha = tf.nn.softmax(C_m, axis=1)

                Q_vec = tf.reduce_sum(Qalpha * rQ, axis=1)
                C_vec = tf.reduce_sum(Calpha * rC, axis=1)

            info = tf.concat([Q_vec, C_vec], axis=1)
            info = Dropout(info, keep_prob=self.dropout_keep_prob, is_train=self._is_train)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, 3, activation=tf.identity)

            return output

        with tf.name_scope('similarity'):
            normalized_q_h_pool = tf.nn.l2_normalize(self.q_h_pool, axis=1)
            normalized_pos_h_pool = tf.nn.l2_normalize(self.pos_h_pool, axis=1)
            normalized_neg_h_pool = tf.nn.l2_normalize(self.neg_h_pool, axis=1)
            self.pos_similarity = tf.reduce_sum(tf.multiply(normalized_q_h_pool, normalized_pos_h_pool), 1)
            self.neg_similarity = tf.reduce_sum(tf.multiply(normalized_q_h_pool, normalized_neg_h_pool), 1)

        with tf.name_scope('loss'):
            original_loss = tf.reduce_sum(margin - self.pos_similarity + self.neg_similarity)
            self.loss = tf.cond(tf.less(0.0, original_loss), lambda: original_loss, lambda: tf.constant(0.0))

        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

