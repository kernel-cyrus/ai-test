import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

if __name__ == "__main__":

    sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)

    # Load engine file
    engine_path = './iris_model.engine'
    with open(engine_path, 'rb') as file:
        engine_data = file.read()

    # Create runtime context
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    # Prepare input / output buffer
    src_buf = cuda.mem_alloc(sample.nbytes)
    dst_buf = cuda.mem_alloc(3 * 8)
    cuda.memcpy_htod(src_buf, sample)

    # Run predict
    context.execute_v2([src_buf, dst_buf])

    # Get result
    result = np.empty((1, 3), dtype=np.float32)
    cuda.memcpy_dtoh(result, dst_buf)

    print("Prediction:", result)
    print("Result:", np.argmax(result))
