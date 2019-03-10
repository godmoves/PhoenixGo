import numpy as np
import tensorflow as tf


def weight_variable(name, shape, dtype):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    # do not use constant as the initializer, that will make the
    # variable stored in wrong type.
    weights = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(
        stddev=stddev, dtype=dtype), dtype=dtype)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
    return weights


# Bias weights for layers not followed by BatchNorm
# We do not regularize biases, so they are not
# added to the regularizer collection
def bias_variable(name, shape, dtype):
    bias = tf.get_variable(name, shape, initializer=tf.zeros_initializer(), dtype=dtype)
    return bias


def bias_variable_reg(name, shape, dtype):
    weights = tf.get_variable(name, shape, initializer=tf.zeros_initializer(), dtype=dtype)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
    return weights


def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')


class Net:
    def __init__(self):
        self.name = 'net'
        # record weight variables for SWA and leela weight output
        self.weights = []
        # status of the net, determine whether we update batchnorm layer parameters
        self.training = tf.placeholder(tf.bool)

    def construct_net(self, planes):
        # build the net graph, return the outputs of policy and value head
        policy_out = None
        # the value head output should be non-sigmoid, we will do that in loss calculation
        value_out = None
        return policy_out, value_out
