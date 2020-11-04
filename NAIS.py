import heapq
import logging
import os
from argparse import ArgumentParser
from time import time

import numpy as np
import tensorflow as tf

from FISM import FISM
from batch import get_train_data, get_batch_train_data, get_batch_test_data
from dataset import DataSet
from evaluate import get_hit_ratio, get_NDCG


def parse_args():
    parser = ArgumentParser(description='Run NAIS.')
    parser.add_argument('--path', nargs='?', default='data',
                        help='Input data path.')
    parser.add_argument('--data_set_name', nargs='?', default='ml-1m',
                        help='Choose a dataset, either ml-1m or pinterest-20.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--topN', type=int, default=10,
                        help='Size of recommendation list.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='whether pretraining or not, 1-pretrain, 0-without pretrain.')

    parser.add_argument('--embedding_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--attention_factor', type=int, default=16,
                        help='Attention factor.')
    parser.add_argument('--algorithm', type=str, default='concat',
                        help='Either concat or prod')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Smoothing exponent of softmax.')
    parser.add_argument('--regs', nargs='?', default='(1e-7, 1e-7, 1e-7, 1e-7, 1e-7)',
                        help='Regularization parameter.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per iteration.')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


class NAIS(tf.Module):

    def __init__(self,
                 num_users,
                 num_items,
                 args):
        self.num_users = num_users
        self.num_items = num_items

        self.embedding_size = args.embedding_size
        self.attention_factor = args.attention_factor
        self.beta = args.beta
        self.algorithm = args.algorithm

        regs = eval(args.regs)
        self.p_reg = regs[0]
        self.q_reg = regs[1]
        self.w_reg = regs[2]
        self.b_reg = regs[3]
        self.h_reg = regs[4]

        self._create_variables()

    @staticmethod
    def init_random(shape):
        return tf.random.truncated_normal(shape=shape, mean=0.0, stddev=0.01)

    def _create_variables(self):
        # Embedding Parameter
        self.P = tf.Variable(NAIS.init_random(shape=(self.num_items, self.embedding_size)),
                             dtype=tf.float32,
                             name='P')
        self.Q = tf.Variable(NAIS.init_random(shape=(self.num_items, self.embedding_size)),
                             dtype=tf.float32,
                             name='Q')
        self.zero = tf.constant(0.0,
                                dtype=tf.float32,
                                shape=(1, self.embedding_size),
                                name='zero')

        # self.bias = tf.Variable(tf.zeros(self.num_items, ), name='bias')

        # Attention Network Parameter
        if self.algorithm == 'concat':
            self.W = tf.Variable(NAIS.init_random(shape=(2 * self.embedding_size, self.attention_factor)),
                                 dtype=tf.float32,
                                 name='W')
        else:
            self.W = tf.Variable(NAIS.init_random(shape=(self.embedding_size, self.attention_factor)),
                                 dtype=tf.float32,
                                 name='W')
        self.b = tf.Variable(NAIS.init_random(shape=(1, self.attention_factor)),
                             dtype=tf.float32,
                             name='b')
        self.h = tf.Variable(tf.ones(shape=(self.attention_factor, 1)),
                             dtype=tf.float32,
                             name='h')

    def _attention(self, inputs, num_idx):
        # (batch,n_u,attention_factor)
        mlp_output = tf.nn.relu(tf.matmul(inputs, self.W) + self.b)
        # (batch,n_u,1)
        weight = tf.matmul(mlp_output, self.h)
        # (batch,n_u)
        weight = tf.reshape(weight, (weight.shape[0], weight.shape[1]))

        exp_a = tf.exp(weight)
        mask_mat = tf.sequence_mask(num_idx, maxlen=inputs.shape[1], dtype=tf.float32)
        exp_att = mask_mat * exp_a

        # (batch,1)
        exp_sum = tf.reduce_sum(exp_att, axis=1, keepdims=True)
        # (batch,n_u,1)
        return tf.expand_dims(exp_att / tf.pow(exp_sum, self.beta), axis=2)

    def predict(self, user_input, item_input, num_idx):
        q_with_mask = tf.concat([self.Q, self.zero], axis=0, name='q_with_mask')
        # (batch,n_u,embedding_size)
        user_embedding = tf.nn.embedding_lookup(q_with_mask, user_input)
        # (batch,1,embedding_size)
        item_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.P, item_input), axis=1)

        if self.algorithm == 'concat':
            # (batch,n_u,2 * embedding_size)
            result = tf.concat([user_embedding,
                                tf.tile(item_embedding,
                                        (1, user_embedding.shape[1], 1))],
                               axis=-1)
        else:
            # (batch,n_u,embedding_size)
            result = user_embedding * item_embedding

        # (batch,embedding_size)
        item_embedding = tf.reduce_sum(item_embedding, axis=1)
        # (batch,n_u,1)
        a_ij = self._attention(result, num_idx)

        return tf.sigmoid(
            tf.reduce_sum(item_embedding * tf.reduce_sum(a_ij * user_embedding, axis=1), axis=1))

    def train(self, user_input, item_input, num_idx, labels, optimizer):
        with tf.GradientTape() as tape:
            predictions = self.predict(user_input, item_input, num_idx)
            total_loss = tf.keras.losses.binary_crossentropy(labels, predictions) \
                         + self.p_reg * tf.reduce_sum(tf.square(self.P)) \
                         + self.q_reg * tf.reduce_sum(tf.square(self.Q)) \
                         + self.w_reg * tf.reduce_sum(tf.square(self.W))

        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss


def evaluate(model, dataset, topN):
    hits, ndcgs = [], []
    for batch_id in range(dataset.num_users):
        user_input, item_input, test_item, n_u = get_batch_test_data(batch_id=batch_id,
                                                                     dataset=dataset)
        predictions = model.predict(user_input=user_input,
                                    item_input=item_input,
                                    num_idx=n_u)
        map_item_score = {}
        for i in range(len(item_input)):
            item = item_input[i]
            map_item_score[item] = predictions[i]

        # Evaluate top rank list
        ranklist = heapq.nlargest(topN, map_item_score, key=map_item_score.get)
        hr = get_hit_ratio(ranklist, test_item)
        ndcg = get_NDCG(ranklist, test_item)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    topN = args.topN
    lr = args.lr
    print(args)

    # whether interrupted or not
    start_epochs = 0

    directory = os.path.join('pretrain', 'NAIS', args.data_set_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # logging setting
    log_dir = os.path.join('log', 'NAIS_%s.log' % args.data_set_name)
    if not os.path.exists('log'):
        os.mkdir('log')
    if logging.root.handlers:
        logging.root.handlers = []
    logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s',
                        level=logging.INFO,
                        filename=log_dir)

    model_out_file = 'NAIS_%d.ckpt' % time()

    dataset = DataSet(path=args.path,
                      data_set_name=args.data_set_name)
    model = NAIS(num_users=dataset.num_users,
                 num_items=dataset.num_items,
                 args=args)
    optimizer = tf.keras.optimizers.Adagrad(lr=lr,
                                            initial_accumulator_value=1e-8)

    checkpoint = tf.train.Checkpoint(model=model,
                                     optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint,
                                         directory=directory,
                                         checkpoint_name=model_out_file,
                                         max_to_keep=1)

    if start_epochs > 0:
        checkpoint.restore(manager.latest_checkpoint)
    elif args.pretrain:
        fism = FISM(num_users=dataset.num_users,
                    num_items=dataset.num_items,
                    args=args)
        ckpt = tf.train.Checkpoint(model=fism)
        ckpt.restore(tf.train.latest_checkpoint('pretrain/FISM/ml-1m_FISM_1573647530.ckpt'))
        model.P = fism.P
        model.Q = fism.Q

    # Check Init performance
    start = time()
    hits, ndcgs = evaluate(model, dataset, topN)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time() - start))
    if start_epochs == 0:
        logging.info('Init: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time() - start))

    # train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(start_epochs, epochs):
        losses = []

        # train step
        start = time()
        train_data = get_train_data(dataset=dataset, num_negatives=args.num_neg)
        if args.verbose:
            print('Epoch %d/%d' % (epoch + 1, epochs))
        for batch_id in range(dataset.num_users):
            start_time = time()
            user_input, num_idx, item_input, labels = get_batch_train_data(batch_id=batch_id,
                                                                           train_data=train_data,
                                                                           train_list=dataset.trainList,
                                                                           num_items=dataset.num_items)
            loss = model.train(user_input=user_input,
                               item_input=item_input,
                               num_idx=num_idx,
                               labels=labels,
                               optimizer=optimizer)
            losses.append(loss)
            if args.verbose:
                print('%d/%d loss=%.4f [%.1f s]' % (batch_id + 1, dataset.num_users, loss, time() - start_time))

        end = time()
        total_loss = np.array(losses).mean()

        # evaluate step
        hits, ndcgs = evaluate(model, dataset, topN)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
              % (epoch + 1, end - start, hr, ndcg, total_loss, time() - end))
        logging.info('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                     % (epoch + 1, end - start, hr, ndcg, total_loss, time() - end))
        if hr > best_hr or (hr == best_hr and ndcg > best_ndcg):
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch + 1
            if args.out:
                manager.save()

    print("Finished.\n Best Iteration %d:  HR = %.4f, NDCG = %.4f. "
          % (best_iter, best_hr, best_ndcg))
    logging.info("Best Iteration %d:  HR = %.4f, NDCG = %.4f. "
                 % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        logging.info("The best NAIS model is saved to %s" % model_out_file)
