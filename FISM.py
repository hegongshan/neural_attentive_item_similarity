import heapq
import logging
import os
from argparse import ArgumentParser
from time import time

import numpy as np
import tensorflow as tf

from dataset import DataSet
from evaluate import get_hit_ratio, get_NDCG
from batch import get_train_data, get_batch_train_data, get_batch_test_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    parser = ArgumentParser(description='Run FISM.')
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

    parser.add_argument('--embedding_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Smoothing exponent of softmax.')
    parser.add_argument('--regs', nargs='?', default='(1e-7, 1e-7, 1e-7)',
                        help='Regularization parameter.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per iteration.')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


class FISM(tf.Module):

    def __init__(self,
                 num_users,
                 num_items,
                 args):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = args.embedding_size
        if hasattr(args,'alpha'):
            self.alpha = args.alpha
        else:
            self.alpha = 0
        self.verbose = args.verbose
        regs = eval(args.regs)
        self.beta = regs[0]
        self.user_bias_reg = regs[1]
        self.item_bias_reg = regs[2]

        self._create_variables()

    @staticmethod
    def init_random(shape):
        return tf.random.truncated_normal(shape=shape, mean=0.0, stddev=0.01)

    def _create_variables(self):
        self.P = tf.Variable(FISM.init_random(shape=(self.num_items, self.embedding_size)),
                             dtype=tf.float32,
                             name='P')

        self.Q = tf.Variable(FISM.init_random(shape=(self.num_items, self.embedding_size)),
                             dtype=tf.float32,
                             name='Q')

        self.zero = tf.constant(0.0,
                                dtype=tf.float32,
                                shape=(1, self.embedding_size),
                                name='zero')

        # self.user_biases = tf.Variable(tf.zeros(shape=(self.num_users,),
        #                                         dtype=tf.float32,
        #                                         name='user_biases'))
        # self.item_biases = tf.Variable(tf.zeros(shape=(self.num_items,)),
        #                                dtype=tf.float32,
        #                                name='item_biases')

    def predict(self, user_input, item_input, user_id, n_u, training=True):

        embedding_p = tf.nn.embedding_lookup(self.P, item_input)

        q_with_mask = tf.concat([self.Q, self.zero], axis=0, name='q_with_mask')
        embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(q_with_mask, user_input), 1)

        # user_bias = tf.nn.embedding_lookup(self.user_biases, user_id)
        # item_bias = tf.nn.embedding_lookup(self.item_biases, item_input)

        if training:
            coefficient = tf.pow(tf.cast(n_u - 1, tf.float32), -tf.constant(self.alpha))
        else:
            coefficient = tf.pow(tf.cast(n_u, tf.float32), -tf.constant(self.alpha))
        return tf.sigmoid(
            coefficient * tf.reduce_sum(embedding_p * embedding_q, axis=1))

    def train_step(self, user_input, item_input, user_id, n_u, labels, optimizer):

        with tf.GradientTape() as tape:
            predictions = self.predict(user_input, item_input, user_id, n_u)
            total_loss = tf.keras.losses.binary_crossentropy(labels, predictions) \
                         + self.beta * (tf.reduce_sum(tf.square(self.P)) + tf.reduce_sum(tf.square(self.Q)))
            # + self.item_bias_reg * tf.reduce_sum(tf.square(self.item_biases)) \
            # + self.user_bias_reg * tf.reduce_sum(tf.square(self.user_biases))

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
                                    n_u=n_u,
                                    user_id=batch_id,
                                    training=False)
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

    if not os.path.exists('pretrain/FISM'):
        os.makedirs('pretrain/FISM')
    if not os.path.exists('log'):
        os.mkdir('log')

    if logging.root.handlers:
        logging.root.handlers = []
    logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s',
                        level=logging.INFO,
                        filename='log/FISM_%s.log' % args.data_set_name)

    directory = 'pretrain/FISM/'
    model_out_file = 'ml-1m_FISM_1573647530.ckpt'  # %s_FISM_%d.ckpt' % (args.data_set_name, time())

    dataset = DataSet(path=args.path,
                      data_set_name=args.data_set_name)
    model = FISM(num_users=dataset.num_users,
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
    # checkpoint.restore(manager.latest_checkpoint)

    # Check Init performance
    start = time()
    hits, ndcgs = evaluate(model, dataset, topN)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time() - start))
    logging.info('Init: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time() - start))

    # train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
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
            loss = model.train_step(user_input=user_input,
                                    item_input=item_input,
                                    n_u=num_idx,
                                    user_id=batch_id,
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
        print("The best FISM model is saved to %s" % model_out_file)
