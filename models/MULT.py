import tensorflow as tf
from layers import Dropout


class MULT(object):
    def __init__(self, q_length, a_length, word_embeddings, filter_sizes, num_filters, margin, l2_reg_lambda):

        self.question = tf.placeholder(tf.int32, [None, q_length], name='question')
        self.answer = tf.placeholder(tf.int32, [None, a_length], name='pos_answer')
        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.is_train = tf.get_variable('is_trian', [], dtype=tf.bool, trainable=False)

        vocab_size, embedding_size = word_embeddings.shape
        num_filters_total = num_filters * len(filter_sizes)

        with tf.variable_scope('build_model', initializer=tf.contrib.layers.xavier_initializer()):
            with tf.device('/cpu:0'), tf.name_scope('embedding'):
                self.embeddings = tf.get_variable("embeddings",
                                                  shape=word_embeddings.shape,
                                                  initializer=tf.constant_initializer(word_embeddings),
                                                  trainable=False)
                self.embedded_q = tf.nn.embedding_lookup(self.embeddings, self.question)
                self.embedded_a = tf.nn.embedding_lookup(self.embeddings, self.answer)

            Q = Dropout(self.embedded_q, keep_prob=0.96, is_train=self.is_train)
            A = Dropout(self.embedded_a, 0.96, self.is_train)

            with tf.variable_scope('preprocessing'):
                gate_q = tf.layers.dense(Q, 150, tf.sigmoid, name='gate')
                output_q = tf.layers.dense(Q, 150, tf.tanh, name='output')
                Q_ = gate_q * output_q

                gate_a = tf.layers.dense(A, 150, tf.sigmoid, name='gate', reuse=True)
                output_a = tf.layers.dense(A, 150, tf.tanh, name='output', reuse=True)
                A_ = gate_a * output_a

            with tf.variable_scope('attention'):
                Q_ = tf.layers.dense(Q_, 150)
                G = tf.nn.softmax(tf.keras.backend.batch_dot(Q_, A_, axes=[2, 2]), axis=1)
                H = tf.keras.backend.batch_dot(G, Q_, axes=[1, 1])

            with tf.variable_scope('comparison'):
                T = A_ * H

            with tf.variable_scope('aggregation'):
                T = tf.expand_dims(T, -1)
                # conv-pool for answer
                pooled_outputs = []
                for filter_size in filter_sizes:
                    with tf.name_scope('conv-pool-{}'.format(filter_size)):
                        # convolution layer
                        filter_shape = [filter_size, 150, 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                        conv = tf.nn.conv2d(T, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                        pooled = tf.nn.max_pool(h, ksize=[1, a_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
                        print(pooled)
                        pooled_outputs.append(pooled)

                pool_concate = tf.concat(pooled_outputs, 3)
                print(pool_concate)
                self.q_h_pool = tf.reshape(pool_concate, [-1, num_filters_total])
                print(self.q_h_pool)
                r = tf.layers.dense(self.q_h_pool, 150, tf.tanh)

        with tf.name_scope('loss'):
            self.score = tf.squeeze(tf.layers.dense(r, 1))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.label, tf.float32), logits=self.score)
            self.loss = tf.reduce_mean(losses)

        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()