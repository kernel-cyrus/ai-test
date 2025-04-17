README
============================

## About the model

This model is a simple test to do the iris classification by using neorual network.

It creates a 3-layers neorual network with tensorflow, and trained with iris dataset.

Then you can use the model to do spcieces classification on various platform.

## Platform Supported

The example can be run on various platforms, including:
- x86 PC (CPU Only)
- Nvidia Orin AGX/Nano Board (Cuda GPU)
- Radxa Rock-5A (RK3588, 8TOPS NPU)

## Code Structure

`ires_model.py` - create neorual-network and do the training with tensorflow, then save model and weight to saved model and onnx format.

`runtime_tensorrt.py` - do prediction with tensorrt on a cuda gpu.

`runtime_rknn` - convert onnx to rknn, and do prediction with rknn framework on a rockchip NPU.

`saved_model` - trained model saved in tensorflow "saved model" format, can be loaded in tensorflow.

`iris_model.keras` - trained model saved in keras v3 format, can be loaded in tensorflow.

`iris_model.onnx` - model converted to onnx format, it is a common format for converting to others.

`iris_model.engine` - model format for tensorrt.

`iris_model.rknn` - model format for rockchip platform.

`dataset\iris.data` - iris dataset.

## Installation

1. Download the code

`git clone https://github.com/kernel-cyrus/ai-test.git`
 
2. Install tensorflow

`pip install tensorflow`

3. Install dependecy

`pip install pandas scikit-learn tf2onnx`

4. Individual platforms

For Orin and Rockchip development board, please follow the official guide.

## Run on PC

**1. Direct run**

`python3 iris_model.py`

<img width="993" alt="image" src="https://github.com/user-attachments/assets/4619ca5d-d0e4-43c9-888c-e9c9ed51eb7e" />

Load trained model, and do the prediction test with tensorflow.

On PC, if there's no CUDA GPU, it only use CPU.

**2. Retrain the model**

`python3 iris_model.py 1`

<img width="1018" alt="image" src="https://github.com/user-attachments/assets/6895ceeb-aff0-4491-b25c-93b1d54be707" />
<img width="1012" alt="image" src="https://github.com/user-attachments/assets/d567c6c1-532f-4582-993b-94071c158995" />

Retrain the model with iris dataset, and save trained model and weight to saved model and keras format.

## Run on Orin

**1. Direct run**

The model can run directly on Orin platform, if everything is OK, it will use CUDA GPU to do the prediction.

**2. Retrain the model**

Same as on PC, if everything is OK, it will use CUDA GPU to do the training.

**3. Convert model to TensorRT format**

`trtexec --onnx=iris_model.onnx --saveEngine=iris_model.engine`

**4. Prediction with TensorRT**

`python3 runtime_tensorrt.py`

<img width="756" alt="image" src="https://github.com/user-attachments/assets/46092e70-21c7-4741-9d72-b0947efab338" />

The runtime will load iris_model.engine, and do the prediction with TensorRT and CUDA GPU.

## Run on Rock-5A

**1. Convert model to RKNN**

`python3 runtime_rknn.py`

<img width="820" alt="image" src="https://github.com/user-attachments/assets/2fb20734-1aa7-45bc-a903-91d955c5970f" />

Before run this step, you need install rknn-toolkit

[https://docs.radxa.com/en/rock5/rock5a/app-development/rknn_install]()

The program will compile onnx model to rknn, and run prediction in simulation mode (CPU only).

**2. Prediction on Rock-5A**

`python3 runtime_rknn.py`

<img width="946" alt="image" src="https://github.com/user-attachments/assets/03d33057-6e9f-4671-af00-155719a9e51f" />

It will use NPU to do the prediction.
