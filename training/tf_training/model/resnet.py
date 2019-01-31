import numpy as np
import tensorflow as tf


def weight_variable(name, shape):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.get_variable(name, initializer=initial)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
    return weights


# Bias weights for layers not followed by BatchNorm
# We do not regularize biases, so they are not
# added to the regularizer collection
def bias_variable(name, shape):
    initial = tf.constant(0.0, shape=shape)
    bias = tf.get_variable(name, initializer=initial)
    return bias


def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')


class ResNet:
    def __init__(self, blocks, filters):
        self.batch_norm_count = 0
        self.reuse_var = None
        self.blocks = blocks
        self.filters = filters

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def reset_batchnorm_key(self):
        self.batch_norm_count = 0
        self.reuse_var = True

    def add_weights(self, variable):
        if self.reuse_var is None:
            self.weights.append(variable)

    def batch_norm(self, net):
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        scope = self.get_batchnorm_key()
        with tf.variable_scope(scope):
            net = tf.layers.batch_normalization(
                net,
                momentum=0.99, axis=1,
                epsilon=1e-5, fused=True,
                center=True, scale=False,
                training=self.training,
                reuse=self.reuse_var)

        for v in ['beta', 'moving_mean', 'moving_variance']:
            name = scope + '/batch_normalization/' + v + ':0'
            var = tf.get_default_graph().get_tensor_by_name(name)
            self.add_weights(var)

        return net

    def conv_block(self, inputs, filter_size, input_channels, output_channels, name):
        W_conv = weight_variable(
            name,
            [filter_size, filter_size, input_channels, output_channels])

        self.add_weights(W_conv)

        net = inputs
        net = conv2d(net, W_conv)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)
        return net

    def residual_block(self, inputs, channels, name):
        net = inputs
        orig = tf.identity(net)

        # First convnet weights
        W_conv_1 = weight_variable(name + "_conv_1", [3, 3, channels, channels])
        self.add_weights(W_conv_1)

        net = conv2d(net, W_conv_1)
        net = self.batch_norm(net)
        net = tf.nn.relu(net)

        # Second convnet weights
        W_conv_2 = weight_variable(name + "_conv_2", [3, 3, channels, channels])
        self.add_weights(W_conv_2)

        net = conv2d(net, W_conv_2)
        net = self.batch_norm(net)
        net = tf.add(net, orig)
        net = tf.nn.relu(net)

        return net

    def construct_net(self, planes):
        # NCHW format
        # batch, 18 channels, 19 x 19
        x_planes = tf.reshape(planes, [-1, 18, 19, 19])

        # Input convolution
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=18,
                               output_channels=self.filters,
                               name="input_conv")
        # Residual tower
        for i in range(0, self.blocks):
            block_name = "res_" + str(i)
            flow = self.residual_block(flow, self.filters,
                                       name=block_name)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.filters,
                                   output_channels=2,
                                   name="policy_head")
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2 * 19 * 19])
        W_fc1 = weight_variable("w_fc_1", [2 * 19 * 19, (19 * 19) + 1])
        b_fc1 = bias_variable("b_fc_1", [(19 * 19) + 1])
        self.add_weights(W_fc1)
        self.add_weights(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.filters,
                                   output_channels=1,
                                   name="value_head")
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19 * 19])
        W_fc2 = weight_variable("w_fc_2", [19 * 19, 256])
        b_fc2 = bias_variable("b_fc_2", [256])
        self.add_weights(W_fc2)
        self.add_weights(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable("w_fc_3", [256, 1])
        b_fc3 = bias_variable("b_fc_3", [1])
        self.add_weights(W_fc3)
        self.add_weights(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        # Reset batchnorm key to 0.
        self.reset_batchnorm_key()

        return h_fc1, h_fc3