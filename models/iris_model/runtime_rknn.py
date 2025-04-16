import os
import sys
import rknn

if __name__ == '__main__':

    build_model = len(sys.argv) >=2 and sys.argv[1] == '1'

    onnx_path = './iris_model.onnx'
    rknn_path = './iris_model.rknn'

    rknn = rknn.api.RKNN()
    #rknn.config(target_platform='rk3588')

    if os.path.exists(rknn_path) and not build_model:

        ret = rknn.load_rknn(rknn_path)
        if ret:
            exit('ERROR: Load rknn failed.')

    else:

        ret = rknn.load_onnx(onnx_path)
        if ret:
            exit('ERROR: Load onnx failed.')

        ret = rknn.build(do_quantization=False)
        if ret:
            exit('ERROR: Build rknn failed.')

        ret = rknn.export_rknn(rknn_path)
        if ret:
            exit('ERROR: Export rknn failed.')

    rknn.release()
