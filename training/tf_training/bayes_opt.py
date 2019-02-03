#!/usr/bin/env python3
import argparse
import random
import multiprocessing as mp

import GPyOpt
import numpy as np
import tensorflow as tf
from numpy.random import seed

from tfprocess import TFProcess
from utils.chunkparser import ChunkParser
from utuls.logger import DefaultLogger
from model.lrschedule import OneCycleLR
from parse import get_chunks, split_chunks, FileDataSrc


# fix the seed to make the result repeatable
seed(123)


class BayesianOptimizer:
    def __init__(self):
        # hyper-parameters we want to optimize
        self.bounds = [{'name': 'low_lr', 'type': 'continuous', 'domain': (3e-4, 1e-2)},
                       {'name': 'high_lr', 'type': 'continuous', 'domain': (1e-2, 0.15)},
                       {'name': 'up_range', 'type': 'continuous', 'domain': (200, 800)},
                       {'name': 'down_range', 'type': 'continuous', 'domain': (200, 800)},
                       {'name': 'tail_range', 'type': 'continuous', 'domain': (200, 400)},
                       {'name': 'l2_factor', 'type': 'continuous', 'domain': (1e-6, 1e-4)}]

        # previous observations
        self.X_init = np.array([[0.032, 0.32, 400, 400, 200, 1e-4],
                                [0.01, 0.1, 400, 400, 200, 1e-4],
                                [1e-3, 0.1, 300, 800, 400, 1e-4],
                                [0.01, 0.1, 400, 400, 200, 3e-6]])

        self.Y_init = np.array([[3.53],
                                [3.18],
                                [3.13],
                                [2.12]])

        self.experiment_id = 0
        self.logger = DefaultLogger

    def suggest_next_location(self, res=None):
        if res is not None:
            res = np.expand_dims(res, axis=0)
            self.X_init = np.vstack((self.X_init, self.X_next))
            self.Y_init = np.vstack((self.Y_init, res))

        # construct Bayesian optimizer
        bayes_opt = GPyOpt.methods.BayesianOptimization(f=None,
                                                        domain=self.bounds,
                                                        model='GP',
                                                        acquisition_type='EI',
                                                        X=self.X_init,
                                                        Y=self.Y_init)
        self.X_next = bayes_opt.suggest_next_locations()[0]

    def log_info(self, log_base):
        self.logger.info(log_base)
        self.logger.info("low_lr = {}".format(self.X_next[0]))
        self.logger.info("high_lr = {}".format(self.X_next[1]))
        self.logger.info("up_range = {:d}".format(int(self.X_next[2])))
        self.logger.info("down_range = {:d}".format(int(self.X_next[3])))
        self.logger.info("tail_range = {:d}".format(int(self.X_next[4])))
        self.logger.info("l2_factor = {}".format(self.X_next[5]))
        self.logger.info("")

    def get_next_lrs(self, sess, res=None):
        self.suggest_next_location(res)
        self.experiment_id += 1
        lrs = OneCycleLR(sess=sess,
                         low_lr=self.X_next[0],
                         high_lr=self.X_next[1],
                         up_range=int(self.X_next[2]),
                         down_range=int(self.X_next[3]),
                         tail_range=int(self.X_next[4]))
        l2_scale = self.X_next[5]
        log_base = "tflogs/bayes_opt_" + str(self.experiment_id)
        self.log_info(log_base)
        return lrs, l2_scale, log_base


def main():
    parser = argparse.ArgumentParser(description='Train network from game data.')

    parser.add_argument("trainpref",
                        help='Training file prefix', nargs='?', type=str)
    parser.add_argument("restorepref",
                        help='Training snapshot prefix', nargs='?', type=str)
    parser.add_argument("--train", '-t',
                        help="Training file prefix", type=str)
    parser.add_argument("--test", help="Test file prefix", type=str)
    parser.add_argument("--restore", type=str,
                        help="Prefix of tensorflow snapshot to restore from")
    parser.add_argument("--logbase", default='leelalogs', type=str,
                        help="Log file prefix (for tensorboard)")
    parser.add_argument("--sample", default=16, type=int,
                        help="Rate of data down-sampling to use")
    args = parser.parse_args()

    # set logger to log to terminal and file
    logger = DefaultLogger

    train_data_prefix = args.train or args.trainpref
    restore_prefix = args.restore or args.restorepref

    training = get_chunks(train_data_prefix)
    if not args.test:
        # Generate test by taking 10% of the training chunks.
        random.shuffle(training)
        training, test = split_chunks(training, 0.1)
    else:
        test = get_chunks(args.test)

    if not training:
        logger.error("No data to train on!")
        return

    logger.info("Training with {} chunks, validating on {} chunks".format(
        len(training), len(test)))

    train_parser = ChunkParser(FileDataSrc(training),
                               shuffle_size=1 << 20,  # 2.2GB of RAM.
                               sample=args.sample,
                               batch_size=128).parse()

    test_parser = ChunkParser(FileDataSrc(test),
                              shuffle_size=1 << 19,
                              sample=args.sample,
                              batch_size=128).parse()

    bopt = BayesianOptimizer()
    final_total_loss = None
    while True:
        tf.reset_default_graph()
        tfprocess = TFProcess()
        lrs, l2_scale, log_base = bopt.get_next_lrs(tfprocess.session, final_total_loss)
        tfprocess.lrs = lrs
        tfprocess.l2_scale = l2_scale

        tfprocess.init(batch_size=128, logbase=log_base)
        final_total_loss = tfprocess.process(train_parser, test_parser)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
    mp.freeze_support()
