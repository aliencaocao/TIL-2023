import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
from collections import namedtuple, OrderedDict

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

class TRTInference:
    def __init__(self, engine_path, output_names_mapping: dict = None, verbose=False):
        cuda.init()
        self.device_ctx = cuda.Device(0).make_context()
        self.engine_path = engine_path
        self.output_names_mapping = output_names_mapping or {}
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = None
        self.load_engine()
        assert self.engine is not None, 'Failed to load TensorRT engine.'

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = self.get_bindings()
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

    def load_engine(self):
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(self.engine):
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                data = np.random.randn(*shape).astype(dtype)
                ptr = cuda.mem_alloc(data.nbytes)
                bindings[name] = Binding(name, dtype, shape, data, ptr)
            else:
                data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                ptr = cuda.mem_alloc(data.nbytes)
                bindings[name] = Binding(name, dtype, shape, data, ptr)

        return bindings

    def __call__(self, blob):
        blob = {n: np.ascontiguousarray(v) for n, v in blob.items()}
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)

        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)

        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            o = self.bindings[n].data
            # reshape to correct output shape
            if o.shape != self.bindings[n].shape:
                o = o.reshape(self.bindings[n].shape)
            outputs[self.output_names_mapping.get(n, n)] = o

        self.stream.synchronize()

        return outputs

    def warmup(self, blob, n=50):
        for _ in range(n):
            self(blob)

    def __del__(self):
        try:
            self.device_ctx.pop()
        except cuda.LogicError:
            pass