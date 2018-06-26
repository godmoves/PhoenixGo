#!/usr/bin/env python3
import tensorflow as tf
import os
import sys
from tfprocess import TFProcess
import uff

import subprocess

UFF_TO_PLAN_EXE_PATH = 'u2p/build/uff_to_plan'
TMP_UFF_FILENAME = 'tmp.uff'


def graphToPlan(uff_model, plan_filename, input_name, policy_name,
                value_name, max_batch_size, max_workspace_size, data_type):

    # convert frozen graph to engine (plan)
    args = [
        TMP_UFF_FILENAME,
        plan_filename,
        input_name,
        policy_name,
        value_name,
        str(max_batch_size),
        str(max_workspace_size),
        data_type  # float / half
    ]
    subprocess.call([UFF_TO_PLAN_EXE_PATH] + args)

    # cleanup tmp file
    os.remove(TMP_UFF_FILENAME)


with open(sys.argv[1], 'r') as f:
    weights = []
    for e, line in enumerate(f):
        if e == 0:
            # Version
            print("Version", line.strip())
            if line != '1\n':
                raise ValueError("Unknown version {}".format(line.strip()))
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

    x = tf.placeholder(tf.float32, [None, 18, 19 * 19], name="inputs")

    tfprocess = TFProcess()
    tf_model = tfprocess.construct_net(x)
    uff_model = uff.from_tensorflow(tf_model, output_nodes=["policy", "value"],
                                    input_nodes=["inputs"], output_filename=TMP_UFF_FILENAME)
    graphToPlan(uff_model, "leelaz.PLAN", "inputs", "policy", "value", 1, 1 << 20, "half")
    if tfprocess.RESIDUAL_BLOCKS != blocks:
        raise ValueError("Number of blocks in tensorflow model doesn't match "
                         "number of blocks in input network")
    if tfprocess.RESIDUAL_FILTERS != channels:
        raise ValueError("Number of filters in tensorflow model doesn't match "
                         "number of filters in input network")
    tfprocess.replace_weights(weights)
    path = os.path.join(os.getcwd(), "leelaz-model")
    save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)
