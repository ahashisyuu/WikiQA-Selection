#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-12-6 下午2:54

import os
import argparse
import tensorflow as tf
from models.MULT import MULT
from data_helper import DataHelper
from data_helper import get_final_rank
from eval import eval_map_mrr

embedding_file = 'data/embeddings/glove.6B.300d.txt'
train_file = 'data/lemmatized/WikiQA-train.tsv'
dev_file = 'data/lemmatized/WikiQA-dev.tsv'
test_file = 'data/lemmatized/WikiQA-test.tsv'
train_triplets_file = 'data/lemmatized/WikiQA-train-triplets.tsv'


def prepare_helper():
    data_helper = DataHelper()
    data_helper.build(embedding_file, train_file, dev_file, test_file)
    data_helper.save('data/model/data_helper_info.bin')


def train_cnn():
    data_helper = DataHelper()
    data_helper.restore('data/model/data_helper_info.bin')
    data_helper.prepare_train_data('data/lemmatized/WikiQA-train.tsv')
    data_helper.prepare_dev_data('data/lemmatized/WikiQA-dev.tsv')
    data_helper.prepare_test_data('data/lemmatized/WikiQA-test.tsv')
    model = MULT(
        q_length=data_helper.max_q_length,
        a_length=data_helper.max_a_length,
        word_embeddings=data_helper.embeddings,
        filter_sizes=[1, 2, 3, 4, 5],
        num_filters=150,
        margin=0.25,
        l2_reg_lambda=0
    )

    global_step = tf.Variable(0, name='global_step', trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=4e-3)
    train_op = optimizer.minimize(model.loss, global_step=global_step)

    checkpoint_dir = os.path.abspath('data/model/checkpoints')
    checkpoint_model_path = os.path.join(checkpoint_dir, 'model.ckpt')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('data/model/summary', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(30):
            train_loss = 0
            for batch in data_helper.gen_batches_data(batch_size=10):
                sess.run(tf.assign(model.is_train, True))
                q_batch, a_batch, label_batch = zip(*batch)
                _, loss, summaries = sess.run([train_op, model.loss, model.summary_op],
                                              feed_dict={model.question: q_batch,
                                                         model.answer: a_batch,
                                                         model.label: label_batch,
                                                         })
                train_loss += loss
                cur_step = tf.train.global_step(sess, global_step)
                summary_writer.add_summary(summaries, cur_step)
                if cur_step % 150 == 0:
                    # print('Loss: {}'.format(train_loss))
                    # test on dev set
                    sess.run(tf.assign(model.is_train, False))
                    q_dev, ans_dev, label_dev = zip(*data_helper.dev_data)
                    similarity_scores = sess.run(model.score, feed_dict={model.question: q_dev,
                                                                              model.answer: ans_dev,
                                                                              model.label: label_dev
                                                                              })
                    for sample, similarity_score in zip(data_helper.dev_samples, similarity_scores):
                        sample.score = similarity_score
                    with open('data/output/WikiQA-dev.rank'.format(epoch), 'w') as fout:
                        for sample, rank in get_final_rank(data_helper.dev_samples):
                            fout.write('{}\t{}\t{}\n'.format(sample.q_id, sample.a_id, rank))
                    dev_MAP, dev_MRR = eval_map_mrr('data/output/WikiQA-dev.rank'.format(epoch), 'data/raw/WikiQA-dev.tsv')
                    print('Dev MAP: {}, MRR: {}'.format(dev_MAP, dev_MRR))
                    if dev_MAP > 0.72 or dev_MRR > 0.74:
                        saver.save(sess, checkpoint_model_path, global_step=cur_step)
                    train_loss = 0
            print('Saving model for epoch {}'.format(epoch))
            # saver.save(sess, checkpoint_model_path, global_step=epoch)


def gen_rank_for_test(checkpoint_model_path):
    data_helper = DataHelper()
    data_helper.restore('data/model/data_helper_info.bin')
    data_helper.prepare_test_data('data/lemmatized/WikiQA-test.tsv')
    model = QaCNN(
        q_length=data_helper.max_q_length,
        a_length=data_helper.max_a_length,
        word_embeddings=data_helper.embeddings,
        filter_sizes=[1, 2, 3, 5, 7, 9],
        num_filters=128,
        margin=0.25,
        l2_reg_lambda=0
    )
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_model_path)
        # test on test set
        q_test, ans_test, label_test = zip(*data_helper.test_data)
        similarity_scores = sess.run(model.similarity, feed_dict={model.question: q_test,
                                                                  model.answer: ans_test,
                                                                  model.label: label_test
                                                                  })
        for sample, similarity_score in zip(data_helper.test_samples, similarity_scores):
            # print('{}\t{}\t{}'.format(sample.q_id, sample.a_id, similarity_score))
            sample.score = similarity_score
        with open('data/output/WikiQA-test.rank', 'w') as fout:
            for sample, rank in get_final_rank(data_helper.test_samples):
                fout.write('{}\t{}\t{}\n'.format(sample.q_id, sample.a_id, rank))
        test_MAP, test_MRR = eval_map_mrr('data/output/WikiQA-test.rank', 'data/raw/WikiQA-test-gold.tsv')
        print('Test MAP: {}, MRR: {}'.format(test_MAP, test_MRR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='whether to prepare data helper')
    parser.add_argument('--train', action='store_true', help='train a model for answer selection')
    parser.add_argument('--test', action='store_true', help='generate the rank for test file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.prepare:
        prepare_helper()
    if args.train:
        train_cnn()
    if args.test:
        checkpoint_num = 2
        gen_rank_for_test(checkpoint_model_path='data/model/checkpoints/model.ckpt-{}'.format(checkpoint_num))
