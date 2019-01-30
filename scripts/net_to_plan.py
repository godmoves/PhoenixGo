#!/usr/bin/env python3
import tensorflow as tf
import os
import sys
from tfprocess import TFProcess
import uff

import subprocess

UFF_TO_PLAN_EXE_PATH = '../bazel-bin/model/build_tensorrt_model'
UFF_FILENAME = 'leelaz-model-0.uff'


def graphToPlan(uff_model_name, input_name, policy_name,
                value_name, max_batch_size, max_workspace_size, data_type):

    # convert frozen graph to engine (plan)
    args = [
        "--model_path",
        uff_model_name[:-4],
        "--input_name",
        input_name,
        "--policy_name",
        policy_name,
        "--value_name",
        value_name,
        "--max_batch_size",
        str(max_batch_size),
        "--max_workspace_size",
        str(max_workspace_size),
        "--data_type",
        data_type,  # FP16 / FP32
        "--logtostderr"
    ]
    subprocess.call([UFF_TO_PLAN_EXE_PATH] + args)


with open(sys.argv[1], 'r') as f:
    weights = []
    for e, line in enumerate(f):
        if e == 0:
            # Version
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

    x = tf.placeholder(tf.float32, [None, 18 * 19 * 19], name="inputs")

    tfprocess = TFProcess()
    tfprocess.RESIDUAL_BLOCKS = blocks
    tfprocess.RESIDUAL_FILTERS = channels

    tfprocess.construct_net(x)

    tf_model = tfprocess.replace_weights(weights)
    uff_model = uff.from_tensorflow(tf_model, output_nodes=["policy", "value"],
                                    input_nodes=["inputs"], output_filename=UFF_FILENAME)
    graphToPlan(UFF_FILENAME, "inputs", "policy", "value", 8, 1 << 20, "FP32")

    path = os.path.join(os.getcwd(), "leelaz-model")
    save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)
