#!/usr/bin/env python3
from collections import deque

import tensorflow as tf

from utils.logger import DefaultLogger


class StepwiseLR:
    '''Stepwise constant learning rate schedule'''

    def __init__(self, sess, min_lr=1e-5, drop_steps=[511, 844, 1071, 1262]):
        self.sess = sess
        self.drop_steps = drop_steps
        self.min_lr = min_lr
        self.global_step = 0

        self.is_end = False

        self.lr = tf.Variable(0.01, dtype=tf.float32, name='lr', trainable=False)

    def step(self, loss):
        self.global_step += 1
        # drop the lr at target steps
        if self.global_step / 1000 in self.drop_steps:
            self.sess.run(tf.assign(self.lr, self.lr * 0.1))
        learning_rate = self.sess.run(self.lr)
        if learning_rate < self.min_lr:
            self.is_end = True
        return learning_rate

    def end(self):
        return self.is_end


class AutoDropLR:
    '''If the loss decrease less than the threshold after target steps, this
    learning rate schedule will drop the learning rate automatically.
    If the learning rate is smaller than the minimal learning rate, we
    will stop the training.'''

    def __init__(self, sess, max_range=800, min_lr=1e-5, threshold=0.01, max_drop_count=5):
        self.min_lr = min_lr
        # we will check the percentage of loss reduction if we are more than
        # max_range steps from last learning rate drop.
        self.max_range = max_range
        # if drop_count exceed max_drop_count, we will drop the learning rate,
        # this is used to avoid some false alarm caused by SGD algorithm.
        self.drop_count = 0
        self.max_drop_count = max_drop_count
        # if the loss reduction in this range is less than the threshold, we
        # will add drop_count by one.
        self.drop_threshold = threshold
        self.loss_record = deque(maxlen=self.max_range)
        self.logger = DefaultLogger
        self.sess = sess
        self.is_end = False

        # initial learning rate
        self.lr = tf.Variable(0.01, dtype=tf.float32, name='lr', trainable=False)

    def step(self, loss):
        self.loss_record.append(loss)
        # keep this for more debug info
        self.logger.debug(self.loss_record)
        if len(self.loss_record) >= self.max_range:
            # calculate the loss drop percentage
            first_loss = self.loss_record[0]
            last_loss = self.loss_record[-1]
            mean_loss = (first_loss + last_loss) / 2
            drop_rate = (first_loss - last_loss) / mean_loss

            if drop_rate < self.drop_threshold:
                self.drop_count += 1
                self.logger.info("Learning rate ready to drop count = {}".format(
                    self.drop_count))

                # drop learning rate only when the loss really has no progress.
                if self.drop_count > self.max_drop_count:
                    self.logger.info("First loss {:g}, last loss {:g}".format(
                        first_loss, last_loss))
                    self.logger.info(
                        "Total loss drop rate {:g} < {:g}, auto drop learning rate.".format(
                            drop_rate, self.drop_threshold))
                    # if no enough progress, drop the learning rate
                    self.sess.run(tf.assign(self.lr, self.lr * 0.1))
                    # reset loss record and drop_count
                    self.loss_record.clear()
                    self.drop_count = 0
        learning_rate = self.sess.run(self.lr)
        if learning_rate < self.min_lr:
            self.is_end = True
        return learning_rate

    def end(self):
        return self.is_end


class LRFinder:
    '''This class is used to find the maximum learning rate that can be used
    to train a specific neural network architecture.'''

    def __init__(self, sess, low_lr=1e-5, high_lr=3, up_range=500):
        # low_lr: the initial learning rate
        # high_lr: the max test learning rate
        # up_range: total test steps, in thousands
        self.sess = sess
        self.is_end = False
        self.logger = DefaultLogger

        self.min_loss = 10
        self.loss_record = []
        self.up_rate = (high_lr - low_lr) / up_range

        self.lr = tf.Variable(low_lr, dtype=tf.float32, name='lr', trainable=False)

    def step(self, loss):
        self.loss_record.append(loss)
        self.min_loss = min(self.min_loss, loss)

        if loss > 1.5 * self.min_loss:
            # if loss explode, we will end the lr finder
            self.is_end = True
            self.logger.info("LRFinder: Loss curve: {}".format(self.loss_record))
        else:
            # otherwise increase the lr (according to the super convergence paper,
            # the LRFinder should increase the lr linearly)
            self.sess.run(tf.assign(self.lr, self.lr + self.up_rate))

        learning_rate = self.sess.run(self.lr)
        self.logger.info("LRFinder: Increase LR to {}".format(learning_rate))
        return learning_rate

    def end(self):
        return self.is_end


class OneCycleLR:
    '''Implementation of 1 Cycle learning rate policy.
    Paper: https://arxiv.org/abs/1708.07120
    A modified version mentioned here: https://www.fast.ai/2018/07/02/adam-weight-decay/'''

    def __init__(self, sess, low_lr=0.01, high_lr=0.1, end_lr=1e-5,
                 up_range=400, down_range=400, tail_range=200):
        # Generally we set up_step = down_step, and tail_step = 1/2 * up_step.
        # high_lr is the highest lr found by LRFinder and low_lr is usually set
        # to 1/10 * high_lr, choose end_lr = 1e-5 is fine.
        self.sess = sess
        self.is_end = False
        self.global_step = 0

        # here we convert the length of each range into total steps, and all
        # steps are in thousands

        # TODO: check the performance of the modified version of 1cycle policy.
        self.up_step = up_range
        self.down_step = up_range + down_range
        self.tail_step = up_range + down_range + tail_range

        self.up_rate = (high_lr - low_lr) / up_range
        self.down_rate = (low_lr - high_lr) / down_range
        self.tail_rate = (end_lr - low_lr) / tail_range

        self.lr_val = low_lr
        self.lr = tf.Variable(low_lr, dtype=tf.float32, name='lr', trainable=False)

    def step(self, loss):
        self.global_step += 1
        self.update_lr_val()
        self.sess.run(tf.assign(self.lr, self.lr_val))
        return self.lr_val

    def update_lr_val(self):
        if self.global_step <= self.up_step:
            self.lr_val += self.up_rate
        elif self.global_step <= self.down_step:
            self.lr_val += self.down_rate
        elif self.global_step <= self.tail_step:
            self.lr_val += self.tail_rate
        else:
            self.is_end = True

    def end(self):
        return self.is_end


class CyclicalLR:
    '''Implementation of Cyclical learning rate policy.
    This is the triangle2 policy mentioned in the paper.
    Paper: https://arxiv.org/abs/1506.01186'''

    def __init__(self, sess, low_lr=3e-4, high_lr=0.1, up_range=100,
                 down_range=100, exp_factor=2):
        self.sess = sess
        self.is_end = False
        self.global_step = 0

        self.low_lr = low_lr
        self.high_lr = high_lr

        self.up_range = up_range
        self.down_range = down_range

        # we will drop the upper learning rate limit after each period.
        self.exp_factor = exp_factor
        # length of learning rate period.
        self.cycle_step = up_range + down_range

        self.lr_val = low_lr
        self.lr = tf.Variable(low_lr, dtype=tf.float32, name='lr', trainable=False)

    def step(self, loss):
        self.global_step += 1
        self.update_lr_val()
        self.sess.run(tf.assign(self.lr, self.lr_val))
        return self.lr_val

    def update_lr_val(self):
        self.global_step += 1
        in_cycle_step = self.global_step % self.cycle_step
        cycle_id = self.global_step // self.cycle_step

        high_lr_lim = (self.high_lr - self.low_lr) / (self.exp_factor ** cycle_id) + self.low_lr
        if in_cycle_step < self.up_range:
            self.lr_val = self.low_lr + (high_lr_lim - self.low_lr) * \
                in_cycle_step / self.up_range
        else:
            self.lr_val = high_lr_lim + (self.low_lr - high_lr_lim) * \
                (in_cycle_step - self.up_range) / self.down_range

    def end(self):
        return self.is_end
