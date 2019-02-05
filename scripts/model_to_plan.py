#!/usr/bin/env python3
import tensorflow as tf
import sys
import uff

import subprocess

from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

UFF_TO_PLAN_EXE_PATH = '../bazel-bin/model/build_tensorrt_model'


def graphToPlan(uff_model_name, input_name, policy_name,
                value_name, max_batch_size, max_workspace_size, data_type):

    # convert frozen graph to engine (plan)
    args = [
        "--model_path",
        uff_model_name,
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


with tf.Session() as sess:
    model_name = sys.argv[1]

    # restore the checkpoint file.
    new_saver = tf.train.import_meta_graph(model_name + ".meta")
    new_saver.restore(sess, model_name)

    graph_def = tf.get_default_graph().as_graph_def()

    # output_graph_def = tf.GraphDef()
    # for node in graphdef.node:
    #     replace_node = tf.NodeDef()
    #     replace_node.CopyFrom(node)
    #     if node.name == "value":
    #         continue
    #     if node.name == "zero/value_head/Reshape_1":
    #         replace_node.name = "value"
    #     if node.name == "inputs":
    #         replace_node.attr["dtype"].CopyFrom(tf.AttrValue(type=tf.float32.as_datatype_enum))
    #     for i, inp in enumerate(node.input):
    #         if inp == "Cast":
    #             replace_node.input[i] = "inputs"
    #             output_graph_def.node.extend([replace_node])

    frozen_graph = optimize_for_inference(graph_def, ["inputs"], ["policy", "value"],
                                          dtypes.float32.as_datatype_enum)
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        frozen_graph,
        ["policy", "value"])

    # # write the frozen graph
    # output_graph = model_name + ".pb"
    # with tf.gfile.GFile(output_graph, "wb") as f:
    #     print("writing the frozen graph file ...")
    #     f.write(frozen_graph.SerializeToString())

    # # check the property of inputs.
    # with tf.gfile.FastGFile(output_graph, "rb") as f:
    #     graphdef = tf.GraphDef()
    #     graphdef.ParseFromString(f.read())
    # for node in graphdef.node:
    #     print(node)

    uff_model = uff.from_tensorflow(frozen_graph,
                                    input_nodes=["inputs"],
                                    output_nodes=["policy", "value"],
                                    output_filename=model_name + '.uff')

    graphToPlan(model_name, "inputs", "policy", "value", 8, 1 << 20, "FP32")
