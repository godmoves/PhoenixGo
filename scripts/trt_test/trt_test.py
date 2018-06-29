import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import subprocess


PLAN_NAME = "test.PLAN"
UFF_TO_PLAN_EXE_PATH = '../u2p/build/uff_to_plan'
UFF_FILENAME = '../leelaz-model-0.uff'


def graphToPlan(uff_model_name, plan_filename, input_name, policy_name,
                value_name, max_batch_size, max_workspace_size, data_type):

    # convert frozen graph to engine (plan)
    args = [
        uff_model_name,
        plan_filename,
        input_name,
        policy_name,
        value_name,
        str(max_batch_size),
        str(max_workspace_size),
        data_type  # float / half
    ]
    subprocess.call([UFF_TO_PLAN_EXE_PATH] + args)


fake_input = np.zeros(19 * 19 * 18, dtype=np.float32)
fake_input[19 * 19 * 16: 19 * 19 * 17] = np.ones(19 * 19, dtype=np.float32)

graphToPlan(UFF_FILENAME, "test.PLAN", "inputs",
            "policy", "value", 4, 1 << 20, "float")
policy = np.empty(362, dtype=np.float32)
value = np.empty(1, dtype=np.float32)

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

engine = trt.utils.load_engine(G_LOGGER, PLAN_NAME)

runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

# alocate device memory
d_input = cuda.mem_alloc(4 * fake_input.size * fake_input.dtype.itemsize)
d_policy = cuda.mem_alloc(4 * policy.size * policy.dtype.itemsize)
d_value = cuda.mem_alloc(4 * value.size * value.dtype.itemsize)

bindings = [int(d_input), int(d_policy), int(d_value)]

stream = cuda.Stream()

# transfer input data to device
cuda.memcpy_htod_async(d_input, fake_input, stream)
# execute model
context.enqueue(1, bindings, stream.handle, None)
# transfer predictions back
cuda.memcpy_dtoh_async(policy, d_policy, stream)
cuda.memcpy_dtoh_async(value, d_value, stream)
# syncronize threads
stream.synchronize()

print(policy[:361].reshape(19, 19))
print(value)

context.destroy()
engine.destroy()
runtime.destroy()
