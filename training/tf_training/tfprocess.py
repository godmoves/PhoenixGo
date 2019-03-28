#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import os
import unittest

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from model.swa import SWA
from model.resnet import ResNet
from model.senet import SENet
from utils.logger import Timer, Stats, DefaultLogger
from model.lrschedule import AutoDropLR, CyclicalLR, OneCycleLR, StepwiseLR
from model.mixprec import float32_variable_storage_getter, LossScalingOptimizer


# Restore session from checkpoint. It silently ignore mis-matches
# between the checkpoint and the graph. Specifically
# 1. values in the checkpoint for which there is no corresponding variable.
# 2. variables in the graph for which there is no specified value in the
#    checkpoint.
# 3. values where the checkpoint shape differs from the variable shape.
# (variables without a value in the checkpoint are left at their default
# initialized value)
def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
         if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)


class TFProcess:
    def __init__(self):
        # Network structure
        self.RESIDUAL_FILTERS = 256
        self.RESIDUAL_BLOCKS = 40
        # Default 16 planes for move history + 2 for color to move
        self.FEATURE_PLANES = 18

        # l2 regularization scale, default value from alphago paper
        self.l2_scale = 1e-4

        # Output weight file with averaged weights
        self.swa_enabled = False

        # log training info
        self.logger = DefaultLogger

        # Horovod: initialize Horovod.
        hvd.init()

        # GPU number and index
        self.gpu_num = hvd.size()
        self.gpu_rank = hvd.rank()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        self.session = tf.Session(config=config)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # set the learning rate schedule
        # Horovod: adjust learning rate based on number of GPUs.
        self.lrs = StepwiseLR(self.session, init_lr=0.01 * self.gpu_num)

        # TODO: use better scale value, we may need to record the histogram of
        # the gradient for further analysis.
        # scale loss to prevent gradient underflow
        self.loss_scale = 128

        # model architecture
        self.model_dtype = tf.float16
        self.model = SENet(self.RESIDUAL_BLOCKS, self.RESIDUAL_FILTERS, self.FEATURE_PLANES,
                           dtype=self.model_dtype)

        if self.swa_enabled:
            self.model = SWA(self.session, self.model)

        self.logger.info("Model name {}, precision {}".format(
            self.model.name, self.model_dtype))

    def init(self, batch_size, logbase='tflogs'):
        self.batch_size = batch_size
        self.logbase = logbase
        # Input batch placeholders
        self.planes = tf.placeholder(tf.string, name='in_planes')
        self.probs = tf.placeholder(tf.string, name='in_probs')
        self.winner = tf.placeholder(tf.string, name='in_winner')

        # Mini-batches come as raw packed strings. Decode
        # into tensors to feed into network.
        # We use fp16 to store the input data to speed up the
        # training process and save space.
        planes = tf.decode_raw(self.planes, tf.int8)
        planes = tf.cast(planes, self.model_dtype)
        # Learning targets are stored in fp32, this can prevent
        # loss overflow/underflow.
        probs = tf.decode_raw(self.probs, tf.float32)
        winner = tf.decode_raw(self.winner, tf.float32)

        planes = tf.reshape(planes, (batch_size, self.FEATURE_PLANES, 19 * 19))
        probs = tf.reshape(probs, (batch_size, 19 * 19 + 1))
        winner = tf.reshape(winner, (batch_size, 1))

        self.init_net(planes, probs, winner)

    def init_net(self, planes, probs, winner):
        self.x = planes
        self.y_ = probs
        self.z_ = winner

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        opt = tf.train.MomentumOptimizer(
            learning_rate=self.lrs.lr, momentum=0.9, use_nesterov=True)

        # Horovod: add Horovod Distributed Optimizer.
        opt = hvd.DistributedOptimizer(opt)

        # Wrap the optimizer to scale loss to make it in a appropriate range.
        opt = LossScalingOptimizer(opt, scale=self.loss_scale)

        # Construct net here.
        with tf.variable_scope('fp32_storage',
                               # this force trainable variables to be stored as float32.
                               custom_getter=float32_variable_storage_getter):
            self.loss, self.policy_loss, self.mse_loss, self.reg_term, self.y_conv = self.tower_loss(
                self.x, self.y_, self.z_)

        # Do swa after we construct the net
        if self.swa_enabled:
            self.model.init_swa(self.loss, self.planes, self.probs, self.winner)

        # Op to minimize total loss
        self.train_op = opt.minimize(self.loss)

        # Op to increment global step counter
        self.step_op = tf.assign_add(self.global_step, 1)

        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), self.logbase + "/test"), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), self.logbase + "/train"), self.session.graph)

        # Build checkpoint saver
        self.saver = tf.train.Saver()

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())

    def tower_loss(self, x, y_, z_):
        y_conv, z_conv = self.model.construct_net(x)

        # cast the nn result to fp32 to avoid loss overflow/underflow
        if self.model_dtype != tf.float32:
            y_conv = tf.cast(y_conv, tf.float32)
            z_conv = tf.cast(z_conv, tf.float32)

        # Calculate loss on policy head
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_),
                                                                   logits=y_conv)
        policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        mse_loss = tf.reduce_mean(tf.squared_difference(z_, z_conv))

        # Regularizer
        reg_variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
        reg_term = self.l2_scale * tf.add_n(
            [tf.cast(tf.nn.l2_loss(v), tf.float32) for v in reg_variables])

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of mse_loss here.
        loss = 1.0 * policy_loss + 1.0 * mse_loss + reg_term

        return loss, policy_loss, mse_loss, reg_term, y_conv

    def restore(self, file):
        self.logger.info("Restoring from {0}".format(file))
        optimistic_restore(self.session, file)

    def measure_loss(self, batch, training=False):
        # Measure loss over one batch. If training is true, also
        # accumulate the gradient and increment the global step.
        ops = [self.policy_loss, self.mse_loss, self.reg_term, self.accuracy]
        if training:
            ops += [self.train_op, self.step_op],
        r = self.session.run(ops, feed_dict={self.model.training: training,
                                             self.planes: batch[0],
                                             self.probs: batch[1],
                                             self.winner: batch[2]})
        # Google's paper scales mse by 1/4 to a [0,1] range, so we do the same here
        return {'policy': r[0], 'mse': r[1] / 4., 'reg': r[2],
                'accuracy': r[3], 'total': r[0] + r[1] + r[2]}

    def process(self, train_data, test_data):
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        self.session.run(hvd.broadcast_global_variables(0))

        stats = Stats()
        timer = Timer()
        while True:
            batch = next(train_data)

            # Measure losses and compute gradients for this batch.
            losses = self.measure_loss(batch, training=True)
            stats.add(losses)

            # fetch the current global step.
            steps = tf.train.global_step(self.session, self.global_step)

            if steps % 1000 == 0:
                speed = 1000 * self.batch_size / timer.elapsed() * self.gpu_num

                # adjust learning rate according to lr schedule
                learning_rate = self.lrs.step(steps / 1000, stats.mean('total'))

                stats.add({'speed': speed, 'lr': learning_rate})

                # exit when lr is smaller than target.
                if self.lrs.is_end:
                    self.logger.info('learning schedule ended')
                    # we return the final total loss at the end of training
                    return stats.mean('total')

                # write the TF log at the first GPU only
                if self.gpu_rank == 0:
                    self.logger.info("step {}k lr={:g} policy={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".format(
                                     steps / 1000, learning_rate, stats.mean('policy'), stats.mean('mse'), stats.mean('reg'),
                                     stats.mean('total'), stats.mean('speed')))

                    summaries = stats.summaries({'Policy Loss': 'policy',
                                                 'MSE Loss': 'mse',
                                                 'Regularization Term': 'reg',
                                                 'Accuracy': 'accuracy',
                                                 'Total Loss': 'total',
                                                 'Speed': 'speed',
                                                 'Learning Rate': 'lr'})

                    self.train_writer.add_summary(tf.Summary(value=summaries), steps)
                    stats.clear()

            if steps % 16000 == 0 and self.gpu_rank == 0:
                test_stats = Stats()
                test_batches = 800  # reduce sample mean variance by ~28x
                for _ in range(0, test_batches):
                    test_batch = next(test_data)
                    losses = self.measure_loss(test_batch, training=False)
                    test_stats.add(losses)
                summaries = test_stats.summaries({'Policy Loss': 'policy',
                                                  'MSE Loss': 'mse',
                                                  'Accuracy': 'accuracy',
                                                  'Total Loss': 'total'})
                self.test_writer.add_summary(tf.Summary(value=summaries), steps)
                self.logger.info("step {}k policy={:g} training accuracy={:g}%, mse={:g}".format(
                    steps / 1000, test_stats.mean('policy'),
                    test_stats.mean('accuracy') * 100.0,
                    test_stats.mean('mse')))

                # Write out current model and checkpoint
                path = os.path.join(os.getcwd(), "weights/leelaz-model")
                tf_model_path = self.saver.save(self.session, path, global_step=steps)
                self.logger.info("Model saved in file: {}".format(tf_model_path))

                # Deprecated: Actually we do not need to write lz weight every
                # time, because we can generate lz weight from tf model. This
                # can save the time of writing lz weight, which is relatively
                # long for large network.
                # leela_weight_path = path + "-" + str(steps) + ".txt"
                # self.save_leelaz_weights(leela_weight_path)
                # self.logger.info("Leela weights saved to {}".format(leela_weight_path))

                # Things have likely changed enough
                # that stats are no longer valid.
                if self.swa_enabled:
                    self.model.do_swa(steps, path, train_data)
                    tf_model_path = self.saver.save(self.session, path, global_step=steps)
                    self.logger.info("SWA Model saved in file: {}".format(tf_model_path))

                # reset the timer to skip test time.
                timer.reset()

    def assign(self, var, values):
        try:
            self.session.run(tf.assign(var, values))
        except Exception:
            self.logger.error("Failed to assign {}: var shape {}, values shape {}".format(
                var.name, var.shape, values.shape))
            raise

    def save_leelaz_weights(self, filename):
        with open(filename, "w") as file:
            # Version tag
            file.write("1")
            for weights in self.model.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                if weights.name.endswith('/batch_normalization/beta:0'):
                    # Batch norm beta needs to be converted to biases before
                    # the batch norm for backwards compatibility reasons
                    var_key = weights.name.replace('beta', 'moving_variance')
                    var = tf.get_default_graph().get_tensor_by_name(var_key)
                    work_weights = tf.multiply(weights,
                                               tf.sqrt(var + tf.constant(1e-5)))
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # Leela/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def replace_weights(self, new_weights):
        for e, weights in enumerate(self.model.weights):
            if isinstance(weights, str):
                weights = tf.get_default_graph().get_tensor_by_name(weights)
            if weights.name.endswith('/batch_normalization/beta:0'):
                # Batch norm beta is written as bias before the batch
                # normalization in the weight file for backwards
                # compatibility reasons.
                bias = tf.constant(new_weights[e], shape=weights.shape)
                # Weight file order: bias, means, variances
                var = tf.constant(new_weights[e + 2], shape=weights.shape)
                new_beta = tf.divide(bias, tf.sqrt(var + tf.constant(1e-5)))
                self.assign(weights, new_beta)
            elif weights.shape.ndims == 4:
                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [2, 3, 1, 0]))
            elif weights.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_weight)
        # This should result in identical file to the starting one
        # self.save_leelaz_weights('restored.txt')


# Unit tests for TFProcess.
def gen_block(size, f_in, f_out):
    return [[1.1] * size * size * f_in * f_out,  # conv
            [-.1] * f_out,  # bias weights
            [-.2] * f_out,  # batch norm mean
            [-.3] * f_out]  # batch norm var


class TFProcessTest(unittest.TestCase):
    def test_can_replace_weights(self):
        tfprocess = TFProcess()
        tfprocess.init(batch_size=1)
        # use known data to test replace_weights() works.
        data = gen_block(3, 18, tfprocess.RESIDUAL_FILTERS)  # input conv
        for _ in range(tfprocess.RESIDUAL_BLOCKS):
            data.extend(gen_block(
                3,
                tfprocess.RESIDUAL_FILTERS,
                tfprocess.RESIDUAL_FILTERS))
            data.extend(gen_block(
                3,
                tfprocess.RESIDUAL_FILTERS,
                tfprocess.RESIDUAL_FILTERS))

        # policy
        data.extend(gen_block(1, tfprocess.RESIDUAL_FILTERS, 2))
        data.append([0.4] * 2 * 19 * 19 * (19 * 19 + 1))
        data.append([0.5] * (19 * 19 + 1))

        # value
        data.extend(gen_block(1, tfprocess.RESIDUAL_FILTERS, 1))
        data.append([0.6] * 19 * 19 * 256)
        data.append([0.7] * 256)
        data.append([0.8] * 256)
        data.append([0.9] * 1)
        tfprocess.replace_weights(data)


if __name__ == '__main__':
    unittest.main()
