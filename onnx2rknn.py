import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = '/home/jtl/playmate_robot/light_face_process/assets/vggface.onnx'
RKNN_MODEL = 'vggface.rknn'

QUANTIZE_ON = False

if __name__=="__main__":
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    #rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588',dynamic_input = [[[1,128,128,3]]])
    rknn.config(target_platform='rk3588')
    
    #rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    #ret = rknn.load_onnx(model=ONNX_MODEL, inputs=['data'],input_size_list=[input_shape])
    #ret = rknn.load_onnx(model=ONNX_MODEL, inputs=[input_shape], input_size_list = [input_shape])
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    #ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    ret = rknn.build(do_quantization=QUANTIZE_ON)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
