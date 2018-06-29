import tensorrt as trt
import numpy as np
import pycuda.driver as cuda


PLAN_NAME = "../leelaz-model-0.PLAN"

fake_input = np.zeros(19 * 19 * 18, dtype=np.float32)
fake_input[19 * 19 * 16: 19 * 19 * 17] = np.ones(19 * 19, dtype=np.float32)

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
