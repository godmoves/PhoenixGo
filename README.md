# HappyGo

## Compile TensorFlow and TensorRT model

### Dependencies

- Python 3.x  
- Numpy   
- TensorFlow 1.12+  
- [TensorRT 5](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#overview)  
- [UFF](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/index.html#installing-the-uff-toolkit)  
- CMake 3.1+  
- GCC 6/7  

Use `pip` to install what you need. For `TensorRT` and `UFF`, you can
find more info [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#overview).   
You need to install `TensorRT` by **tar package** to get python support. Find more info about how to [download and install](https://developer.nvidia.com/tensorrt). 

### (Optional for TensorRT) Build uff_to_plan

First `git clone` this repository, then execute the commands below:

```
$ mkdir ckpt
$ cd tools/model_conversion/uff2plan
$ mkdir build && cd build
$ cmake ..
$ make
$ cd ../..
```

### Convert the weight file

```
$ python net_to_model.py </path/to/weight/file>
```

If you don't need the TensorRT weight, you can skip that by add `--skip_trt` option.

```
$ python net_to_model.py </path/to/weight/file> --skip_trt
```

You will get the `.uff`, `.PLAN`, `checkpoint` and TensorFlow models. Copy them to `ckpt` folder.
You also need to update the `checkpoint` file to match the new weight path.

## Compile TPU model

HappyGo now support Cloud TPU. If you want to run the engine on TPU, you need to freeze a
TensorFlow model into a GraphDef proto that can be run on Cloud TPU.

This can be done by using the `model_to_tpu.py`:

```
python model_to_tpu.py <path/to/tf/model> --tpu_name=$TPU_NAME --num_tpu_cores=8
```

The TensorFlow model can be either a local file or one on Google Cloud Storage, and $TPU_NAME is
the gRPC name of your TPU, e.g. `grpc://10.240.2.10:8470`. This can be found from the output of
gcloud beta compute tpus list.

This command **must** be run from a Cloud TPU-ready GCE VM.

This invocation to `model_to_tpu.py` will replicate the model 8 times so that it can run on all
eight cores of a Cloud TPU. To take advantage of this parallelism when running inference,
virtual_losses * parallel_games must be at least 8, ideally 128 or higher.

## Build the engine on Linux

### Requirements

* GCC with C++11 support
* Bazel 0.19.2
* CUDA 10 and cuDNN 7 (for GPU support)
* NCCL 2.4 (for GPU support)
* TensorRT (for accelerating computation on GPU, 5.0.2 is known-good) (optional)

### Build TensorFlow library

Download Bazel 0.19.2 form [here](https://github.com/bazelbuild/bazel/releases?after=0.20.0) and install.

Clone the repository and build the TensorFlow library by using:

```
$ git clone https://github.com/godmoves/GoEngine.git
$ cd GoEngine
$ ./configure_tensorflow.sh
```

`./configure_tensorflow.sh` will ask where CUDA and TensorRT have been installed, specify them if need.

If you want to compile for CPU and not GPU, then execute the following instead:

```
TF_NEED_CUDA=0 ./configure_tensorflow.sh
```

This will automatically perform [Installing TensorFlow from Sources](https://www.tensorflow.org/install/source)
but instead of installing the TensorFlow package, it extracts the generated C++ headers into the `model/tensorflow`
subdirectory of the repo. The script then builds the required TensorFlow shared libraries and copies
them to the same directory. The TensorFlow cc library build target in cc/BUILD pulls these header
and library files together into a format the Bazel understands how to link against.

The building process may take a long time.

### Build the MCTS engine

Then build the main MCTS engine with bazel command:

```
$ bazel build --define=grpc_no_ares=true --define=trt=1 //mcts:mcts_main
```

Dependencies will be downloaded and built automatically.

If you don't need TensorRT for inference, you can omit the `--define=trt=1` option.


### Run the engine

Convert the Leela Zero weight as mentioned before, and run
```
$ scripts/start.sh ./etc/{config}
```
or
```
$ bazel-bin/mcts/mcts_main --config_path=./etc/{config} --gtp --logtostderr --v=1
```

If no configure file is provided, `start.sh` will detect the number of GPUs automatically.
See also [configure-guide](#configure-guide).

`--logtostderr` let `mcts_main` log messages to stderr, if you want to log to files,
change `--logtostderr` to `--log_dir={log_dir}` You could modify your config file following
[configure-guide](#configure-guide).

Furthermore, if you want to fully control all the options of `mcts_main` (such as, changing log destination),
you could also run `bazel-bin/mcts/mcts_main` directly. See also [command-line-options](#command-line-options).

The engine supports the GTP protocol, means it could be used with a GUI with GTP capability,
such as [Sabaki](http://sabaki.yichuanshen.de) or [GuGui](https://github.com/godmoves/GoGui).

### Distribute mode

HappyGo support running with distributed workers, if there are GPUs on different machine.

Build the distribute worker:

```
$ bazel build //dist:dist_zero_model_server
```

Run `dist_zero_model_server` on distributed worker, **one for each GPU**.

```
$ CUDA_VISIBLE_DEVICES={gpu} bazel-bin/dist/dist_zero_model_server --server_address="0.0.0.0:{port}" --logtostderr
```

Fill `ip:port` of workers in the config file (`etc/mcts_dist.conf` is an example config for 32 workers),
and run the distributed master:

```
$ scripts/start.sh etc/mcts_dist.conf
```
or
```
$ bazel-bin/mcts/mcts_main --config_path=etc/mcts_dist.conf --gtp --logtostderr --v=1
```

## Configure Guide

Here are some important options in the config file:

* `num_eval_threads`: should equal to the number of GPUs
* `num_search_threads`: should a bit larger than `num_eval_threads * eval_batch_size`
* `timeout_ms_per_step`: how many time will used for each move
* `max_simulations_per_step`: how many simulations will do for each move
* `gpu_list`: use which GPUs, separated by comma
* `model_config -> model_dir`: directory where trained network stored
* `model_config -> checkpoint_path`: use which checkpoint, get from `model_dir/checkpoint` if not set
* `model_config -> enable_tensorrt`: use TensorRT or not
* `model_config -> trt_plan_model_path`: use which TensorRT PLAN model, if `enable_tensorrt`
* `model_config -> trt_uff_model_path`: use which TensorRT uff model, if both `PLAN` and `uff` model path are set, the `uff` model will be used by default
* `model_config -> enable_fp16`: use FP16 precision for TensorRT to speed up the inference process
* `mode_config -> enable_tpu`: use Cloud TPU as inference engine
* `model_config -> tpu_name`: the gRPC name of your TPU, e.g. `grpc://10.240.2.10:8470`
* `model_config -> tpu_model_path`: the path to TPU model file, whether locally or on the Google Cloud
* `max_search_tree_size`: the maximum number of tree nodes, change it depends on memory size
* `max_children_per_node`: the maximum children of each node, change it depends on memory size
* `enable_background_search`: pondering in opponent's time
* `early_stop`: genmove may return before `timeout_ms_per_step`, if the result would not change any more
* `unstable_overtime`: think `timeout_ms_per_step * time_factor` more if the result still unstable
* `behind_overtime`: think `timeout_ms_per_step * time_factor` more if winrate less than `act_threshold`

Options for distribute mode:

* `enable_dist`: enable distribute mode
* `dist_svr_addrs`: `ip:port` of distributed workers, multiple lines, one `ip:port` in each line
* `dist_config -> timeout_ms`: RPC timeout

Options for async distribute mode:

> Async mode is used when there are huge number of distributed workers (more than 200),
> which need too many eval threads and search threads in sync mode.
> `etc/mcts_async_dist.conf` is an example config for 256 workers.

* `enable_async`: enable async mode
* `enable_dist`: enable distribute mode
* `dist_svr_addrs`: multiple lines, comma separated lists of `ip:port` for each line
* `num_eval_threads`: should equal to number of `dist_svr_addrs` lines
* `eval_task_queue_size`: tunning depend on number of distribute workers
* `num_search_threads`: tunning depend on number of distribute workers


Read `mcts/mcts_config.proto` for more config options.

## Command Line Options

`mcts_main` accept options from command line:

* `--config_path`: path of config file
* `--gtp`: run as a GTP engine, if disable, gen next move only
* `--init_moves`: initial moves on the go board
* `--gpu_list`: override `gpu_list` in config file
* `--listen_port`: work with `--gtp`, run gtp engine on port in TCP protocol
* `--allow_ip`: work with `--listen_port`, list of client ip allowed to connect
* `--fork_per_request`: work with `--listen_port`, fork for each request or not

Glog options are also supported:

* `--logtostderr`: log message to stderr
* `--log_dir`: log to files in this directory
* `--minloglevel`: log level, 0 - INFO, 1 - WARNING, 2 - ERROR
* `--v`: verbose log, `--v=1` for turning on some debug log, `--v=0` to turning off

`mcts_main --help` for more command line options.

## FAQ

**1. Where is the win rate?**

Print in the log, something like:

```
I0514 12:51:32.724236 14467 mcts_engine.cc:157] 1th move(b): dp, winrate=44.110905%, N=654, Q=-0.117782, p=0.079232, v=-0.116534, cost 39042.679688ms, sims=7132, height=11, avg_height=5.782244, global_step=639200
```

**2. There are too much log.**

Passing `--v=0` to `mcts_main` will turn off many debug log.
Moreover, `--minloglevel=1` and `--minloglevel=2` could disable INFO log and WARNING log.

Or, if you just don't want to log to stderr, replace `--logtostderr` to `--log_dir={log_dir}`,
then you could read your log from `{log_dir}/mcts_main.INFO`.

**3. How make HappyGo think with constant time per move?**

Modify your config file. `early_stop`, `unstable_overtime`, `behind_overtime` and
`time_control` are options that affect the search time, remove them if exist then
each move will cost constant time/simulations.

**4. GTP command `time_settings` doesn't work.**

Add these lines in your config:

```
time_control {
    enable: 1
    c_denom: 20
    c_maxply: 40
    reserved_time: 1.0
}
```

`c_denom` and `c_maxply` are parameters for deciding how to use the "main time".
`reserved_time` is how many seconds should reserved (for network latency) in "byo-yomi time".
