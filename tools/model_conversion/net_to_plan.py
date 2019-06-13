#!/usr/bin/env python3
import os
import argparse
import subprocess

import uff
import tensorflow as tf

from tfprocess import TFProcess

UFF_TO_PLAN_EXE_PATH = 'uff2plan/build/uff_to_plan'


def graphToPlan(uff_filename, plan_filename, input_tensor_name, input_feature_num, policy_tensor_name,
                value_tensor_name, max_batch_size, max_workspace_size, data_type):

    # convert frozen graph to engine (plan)
    args = [
        uff_filename,
        plan_filename,
        input_tensor_name,
        str(input_feature_num),
        policy_tensor_name,
        value_tensor_name,
        str(max_batch_size),
        str(max_workspace_size),
        data_type  # float / half
    ]
    subprocess.call([UFF_TO_PLAN_EXE_PATH] + args)


def main(args):
    with open(args.model_path, 'r') as f:
        weights = []
        for e, line in enumerate(f):
            if e == 0:
                print("Version", line.strip())
            else:
                weights.append(list(map(float, line.split(' '))))
            if e == 2:
                channels = len(line.split(' '))
                print("Channels", channels)

    blocks = e - (4 + 14)
    if blocks % 8 != 0:
        raise ValueError("Inconsistent number of weights in the file")
    blocks //= 8
    print("Blocks", blocks)

    x = tf.placeholder(tf.float32,
                       [None, args.features * args.board_size * args.board_size],
                       name=args.input_tensor_name)

    tfprocess = TFProcess(args)
    tfprocess.RESIDUAL_BLOCKS = blocks
    tfprocess.RESIDUAL_FILTERS = channels

    tfprocess.construct_net(x)

    tf_model = tfprocess.replace_weights(weights)
    tfprocess.session.run(tf.global_variables_initializer())

    if not args.skip_trt:
        uff_filename = args.output_prefix + '-{}.uff'.format(args.global_step)
        with open(uff_filename + '.step', 'w') as f:
            f.write(str(args.global_step))
        uff.from_tensorflow(
            tf_model,
            output_nodes=[args.policy_tensor_name, args.value_tensor_name],
            input_nodes=[args.input_tensor_name],
            output_filename=uff_filename)

        if args.precision == 'half':
            plan_filename = args.output_prefix + '-{}.FP16.PLAN'.format(args.global_step)
        else:
            plan_filename = args.output_prefix + '-{}.FP32.PLAN'.format(args.global_step)
        with open(plan_filename + '.step', 'w') as f:
            f.write(str(args.global_step))

        graphToPlan(uff_filename, plan_filename, args.input_tensor_name,
                    args.features, args.policy_tensor_name,
                    args.value_tensor_name, args.max_batch_size,
                    args.max_workspace_size, args.precision)

    path = os.path.join(os.getcwd(), args.output_prefix)
    tf.train.Saver().save(tfprocess.session, path, global_step=args.global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert txt weight to TensorRT model.')
    parser.add_argument('model_path', type=str,
                        help='Path to txt weight file')
    parser.add_argument('--precision', type=str, default='float',
                        help='Data type of the weight, float or half')
    parser.add_argument('--features', type=int, default=18,
                        help='Number of features used in the model')
    parser.add_argument('--board_size', type=int, default=19,
                        help='Board size')
    parser.add_argument('--max_workspace_size', type=int, default=1 << 20,
                        help='Max workspace size')
    parser.add_argument('--max_batch_size', type=int, default=4,
                        help='Max batch size')
    parser.add_argument('--output_prefix', type=str, default='tf-model',
                        help='Prefix name of output TensorRT model')
    parser.add_argument('--global_step', type=int, default=0,
                        help='global_step of the model')
    parser.add_argument('--input_tensor_name', type=str, default='inputs')
    parser.add_argument('--value_tensor_name', type=str, default='value')
    parser.add_argument('--policy_tensor_name', type=str, default='policy')
    parser.add_argument('--skip_trt', default=False, action='store_true',
                        help='Skip TRT model conversion, save TF model only')
    args = parser.parse_args()
    main(args)
