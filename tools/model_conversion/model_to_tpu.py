#!/usr/bin/env python3
import argparse

import tensorflow as tf

from tfprocess import TFProcess


def freeze_graph_tpu(model_path, inference_fn):
    """Custom freeze_graph implementation for Cloud TPU."""

    assert model_path
    assert args.tpu_name
    if args.tpu_name.startswith('grpc://'):
        tpu_grpc_url = args.tpu_name
    else:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            args.tpu_name, zone=None, project=None)
        tpu_grpc_url = tpu_cluster_resolver.get_master()
    sess = tf.Session(tpu_grpc_url)

    output_names = []
    with sess.graph.as_default():
        # Replicate the inference function for each TPU core.
        replicated_features = []
        for i in range(args.num_tpu_cores):
            features = tf.placeholder(
                tf.float32, [None, args.features * args.board_size * args.board_size],
                name=args.input_tensor_name + '_%d' % i)
            replicated_features.append((features,))
        outputs = tf.contrib.tpu.replicate(
            inference_fn, replicated_features)

        # The replicate op assigns names like output_0_shard_0 to the output
        # names. Give them human readable names.
        for i, (policy_output, value_output) in enumerate(outputs):
            policy_name = args.policy_tensor_name + '_%d' % i
            value_name = args.value_tensor_name + '_%d' % i
            output_names.extend([policy_name, value_name])
            tf.identity(policy_output, policy_name)
            tf.identity(value_output, value_name)

        tf.train.Saver().restore(sess, model_path)

    # Freeze the graph.
    model_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_names)
    with tf.gfile.GFile(model_path + '_tpu.pb', 'wb') as f:
        f.write(model_def.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model to TPU.')
    parser.add_argument('model_path', type=str,
                        help='Path to txt weight file')
    parser.add_argument('--board_size', type=int, default=19,
                        help='Board size')
    parser.add_argument('--features', type=int, default=18,
                        help='Number of features used in the model')
    parser.add_argument('--input_tensor_name', type=str, default='inputs')
    parser.add_argument('--value_tensor_name', type=str, default='value')
    parser.add_argument('--policy_tensor_name', type=str, default='policy')
    parser.add_argument('--global_step', type=int, default=0,
                        help='global_step of the model')
    parser.add_argument('--tpu_name', type=str, default=None,
                        help='The Cloud TPU to use for conversion. This should be either the name used'
                             'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
    parser.add_argument('--num_tpu_cores', type=int, default=8,
                        help='Number of TPU cores. For a single TPU device, this is 8 because each'
                             ' TPU has 4 chips each with 2 cores.')
    args = parser.parse_args()
    tfp = TFProcess(args, own_session=False)

    def inference_fn(features):
        return tfp.construct_net(features)

    freeze_graph_tpu(args.model_path, inference_fn)
