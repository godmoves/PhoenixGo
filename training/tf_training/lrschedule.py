from collections import deque

import tensorflow as tf

from utils import DefaultLogger


class AutoDrop:
    def __init__(self, sess, max_range=800, min_lr=1e-5, threshold=0.01, max_drop_count=5):
        self.min_lr = min_lr
        self.max_range = max_range
        self.drop_count = 0
        self.max_drop_count = max_drop_count
        self.drop_threshold = threshold
        self.loss_record = deque(maxlen=self.max_range)
        self.logger = DefaultLogger
        self.sess = sess
        self.is_end = False

        # initial learning rate
        self.lr = tf.Variable(0.1, dtype=tf.float32, name='lr', trainable=False)

    def step(self, loss):
        self.loss_record.append(loss)
        # keep this for more debug info
        self.logger.debug(self.loss_record)
        if len(self.loss_record) >= self.max_range:
            first_loss = self.loss_record[0]
            last_loss = self.loss_record[-1]
            mean_loss = (first_loss + last_loss) / 2
            drop_rate = (first_loss - last_loss) / mean_loss

            if drop_rate < self.drop_threshold:
                self.drop_count += 1
                self.logger.info("Learning rate ready to drop count = {}".format(
                    self.drop_count))

                # drop lr only when the loss really has no progress.
                # this will avoid some false alarms.
                if self.drop_count > self.max_drop_count:
                    self.logger.info("First loss {:g}, last loss {:g}".format(
                        first_loss, last_loss))
                    self.logger.info("Total loss drop rate {:g} < {:g}, auto drop learning rate.".format(
                        drop_rate, self.drop_threshold))
                    # if no enough progress, drop the learning rate
                    self.sess.run(tf.assign(self.lr, self.lr * 0.1))
                    # reset loss record and count
                    self.loss_record.clear()
                    self.drop_count = 0
        learning_rate = self.sess.run(self.lr)
        if learning_rate < self.min_lr:
            self.is_end = True
        return learning_rate

    def end(self):
        return self.is_end


class LRFinder:
    def __init__(self, sess, initial_lr=1e-5):
        self.loss_record = [1]
        self.sess = sess
        self.initial_lr = initial_lr
        self.logger = DefaultLogger
        self.is_end = False

        self.lr = tf.Variable(self.initial_lr, dtype=tf.float32, name='lr', trainable=False)

    def step(self, loss):
        self.loss_record.append(loss)
        if loss > 10 * self.loss_record[-2]:
            # if loss explode, we will end the lr finder
            self.is_end = True
            self.logger.info("LRFinder: Loss curve: {}".format(self.loss_record[1:]))
        else:
            # otherwise increse the lr
            self.logger.info("LRFinder: Double the LR")
            self.sess.run(tf.assign(self.lr, self.lr * 2))

    def end(self):
        return self.is_end
