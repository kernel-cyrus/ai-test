import os
import sys
import platform
import numpy as np
from rknn.api import RKNN

def print_result(sample, result):
    print('Sample:', sample)
    print('Prediction:', result)
    print('Result:', np.argmax(result))

def runtime_pc(onnx_path, rknn_path, sample):

    rknn = RKNN()

    rknn.config(target_platform='rk3588') # Define input shape: "dynamic_input=[[[1, 4]]]"
    
    print('Load onnx model:', onnx_path)
    ret = rknn.load_onnx(onnx_path, inputs=['keras_tensor'], input_size_list=[[1, 4]])
    if ret:
        exit('ERROR: Load onnx failed.')

    ret = rknn.build(do_quantization=False)
    if ret:
        exit('ERROR: Build rknn failed.')

    ret = rknn.export_rknn(rknn_path)
    if ret:
        exit('ERROR: Export rknn failed.')
    print('Save rknn model:', rknn_path)

    print('Init simulation runtime...')
    ret = rknn.init_runtime(target=None)
    if ret:
        exit('ERROR: Runtime init failed.')

    result = rknn.inference(inputs=[sample])
    
    print_result(sample, result[0])

    rknn.release()

def runtime_board(rknn_path, sample):

    rknn = RKNN()

    print('Load rknn model:', rknn_path)
    ret = rknn.load_rknn(rknn_path)
    if ret:
        exit('ERROR: Load rknn failed.')
        
    print('Init rknn runtime.')
    ret = rknn.init_runtime(target='rk3588')
    if ret:
        exit('ERROR: Runtime init failed.')

    result = rknn.inference(inputs=[sample])
    
    print_result(sample, result[0])

    rknn.release()

if __name__ == '__main__':

    onnx_path = './iris_model.onnx'
    rknn_path = './iris_model.rknn'
    
    sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
    
    arch = platform.machine()

    if arch == 'x86_64':

        from rknn.api import RKNN
        runtime_pc(onnx_path, rknn_path, sample)

    elif arch == 'aarch64':

        from rknnlite.api import RKNNLite as RKNN
        runtime_board(rknn_path, sample)

    else:
        exit('ERROR: Unsupport platform.')

