#!/usr/bin/env python3

import tensorflow as tf
import uff

import subprocess

UFF_TO_PLAN_EXE_PATH = '../uff2plan/build/uff_to_plan'


def graphToPlan(uff_model_name, plan_filename, input_name, input_feature, policy_name,
                value_name, max_batch_size, max_workspace_size, data_type):

    # convert frozen graph to engine (plan)
    args = [
        uff_model_name,
        plan_filename,
        input_name,
        str(input_feature),
        policy_name,
        value_name,
        str(max_batch_size),
        str(max_workspace_size),
        data_type  # float / half
    ]
    subprocess.call([UFF_TO_PLAN_EXE_PATH] + args)


with tf.Session() as sess:
    # restore the checkpoint file.
    new_saver = tf.train.import_meta_graph("meta_graph")
    new_saver.restore(sess, "zero.ckpt-20b-v1")

    # save the graph for tensorboard.
    # a warning will be generated due to we don't initialize the global_step,
    # but it doesn't matter due to we only need the structure of the graph.
    summary_writer = tf.summary.FileWriter('./log', sess.graph)

    # freeze the graph and parameters.
    graphdef = tf.get_default_graph().as_graph_def()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                graphdef,
                                                                ["policy", "zero/value_head/Tanh"])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    output_graph = "zero.ckpt-20b-v1.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        print("writing the frozen graph file ...")
        f.write(frozen_graph.SerializeToString())

    # check the property of inputs.
    with tf.gfile.FastGFile(output_graph, "rb") as f:
        graphdef = tf.GraphDef()
        graphdef.ParseFromString(f.read())
    for node in graphdef.node:
        if node.name == "inputs":
            print("\nInputs info:")
            print(node)

    # note that the output of value head should be "zero/value_head/Tanh",
    # because the identify layer is not supported by tensorrt.
    uff_model = uff.from_tensorflow_frozen_model(output_graph,
                                                 output_nodes=["policy", "zero/value_head/Tanh"],
                                                 input_nodes=["inputs"],
                                                 output_filename="zero.ckpt-20b-v1.uff")

    # Cast layer is not supported by tensorrt now.
    # Inputs should be float rather than bool, so we need to skip or rewrite
    # the original graph's inputs part to fix this.
    #
    # Some tools may solve this problem:
    # tf.contrib.graph_editor:
    # https://tensorflow.google.cn/api_guides/python/contrib.graph_editor
    # Graph Transform Tool:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms
    graphToPlan("zero.ckpt-20b-v1.uff", "zero.ckpt-20b-v1.FP16.PLAN", "inputs", 17,
                "policy", "zero/value_head/Tanh", 4, 1 << 20, "half")
