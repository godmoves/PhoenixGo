#!/use/bin/env pyhton3
import shutil
import os

conf_path = "./etc"

if os.path.exists(conf_path):
    shutil.rmtree(conf_path)
    os.mkdir(conf_path)
else:
    os.mkdir(conf_path)

eval_thread_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
search_thread_list = [0, 8, 12, 16, 20, 24, 28, 32, 36]


def get_model(trt, elf):
    if elf:
        if trt:
            info = """model_config {
    train_dir: "ckpt.elf"
    enable_tensorrt: 1
    tensorrt_model_path: "leelaz-elf-0.PLAN"
}"""
        else:
            info = """model_config {
    train_dir: "ckpt.elf"
    meta_graph_path: "leelaz-elf-0.meta"
}"""
    else:
        if trt:
            info = """model_config {
    train_dir: "ckpt.lz"
    enable_tensorrt: 1
    tensorrt_model_path: "leelaz-model-0.PLAN"
}"""
        else:
            info = """model_config {
    train_dir: "ckpt.lz"
    meta_graph_path: "leelaz-model-0.meta"
}"""

    return info


def get_gpu_list(gpu_num):
    gpu_list = ""
    for i in range(gpu_num):
        gpu_list += str(i) + ","
    return gpu_list


def get_name(gpu_num, trt, elf):
    name = "mcts_" + str(gpu_num) + "gpu"
    if not trt:
        name += "_notrt"
    if elf:
        name += "_elf"
    else:
        name += "_lz"
    name += ".conf"
    return name


def get_conf(gpu_num, trt, elf):
    name = get_name(gpu_num, trt, elf)

    demo = """num_eval_threads: %s
num_search_threads: %s
max_children_per_node: 64
max_search_tree_size: 400000000
timeout_ms_per_step: 30000
max_simulations_per_step: 0
eval_batch_size: 4
eval_wait_batch_timeout_us: 100
%s
gpu_list: "%s"
c_puct: 2.5
virtual_loss: 1.0
enable_resign: 1
v_resign: -0.9
enable_dirichlet_noise: 0
dirichlet_noise_alpha: 0.03
dirichlet_noise_ratio:  0.25
monitor_log_every_ms: 0
get_best_move_mode: 0
enable_background_search: 1
enable_policy_temperature: 0
policy_temperature: 0.67
inherit_default_act: 1
early_stop {
    enable: 1
    check_every_ms: 100
    sims_factor: 1.0
    sims_threshold: 2000
}
unstable_overtime {
    enable: 1
    time_factor: 0.3
}
behind_overtime {
    enable: 1
    act_threshold: 0.0
    time_factor: 0.3
}
time_control {
    enable: 1
    c_denom: 20
    c_maxply: 40
    reserved_time: 1.0
}
"""

    eval_thread = str(eval_thread_list[gpu_num])
    search_thread = str(search_thread_list[gpu_num])
    model = get_model(trt, elf)
    gpu = get_gpu_list(gpu_num)

    conf = demo % (eval_thread, search_thread, model, gpu)

    with open(conf_path + "/" + name, 'w') as f:
        f.write(conf)

    return conf


trt = True
elf = False
for i in range(4):
    get_conf(i + 1, trt, elf)
    trt = not trt
    get_conf(i + 1, trt, elf)
    elf = not elf
    get_conf(i + 1, trt, elf)
    trt = not trt
    get_conf(i + 1, trt, elf)
    elf = not elf
