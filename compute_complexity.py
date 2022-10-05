import tensorflow as tf
import keras.backend as K
from unet_plusplus import *
import os
from model import Deeplabv3
import sys
sys.path.append('/keras-DR_seg-master/models/keras_flops_master')
from keras_flops_master.keras_flops.flops_calculation import get_flops


def get_model(classes = 5,
                target_size = (420,420),
                img_channel = 3,
                epochs = None,
                learning_rate = 0.0001,
                weight_decay_rate = 0.00005,
                model_name = None,
                supervision = True,
                use_ps = True,
                normalize = 'gn',
              activation='softmax'):
    w, h = target_size[1], target_size[0]
    if model_name == 'unet_plusplus':

        model = res_unet_plusplus(input_size=(w, h, img_channel),
                                  classes=classes,
                                  activation=activation,
                                  supervision=supervision,
                                  weight_fusion=False,
                                  use_fpn=False,
                                  use_ps=use_ps,
                                  ex_supervision=False,
                                  normalize=normalize)
        '''
        model = original_unet_plusplus(input_size = (w,h,img_channel),
                                      classes = classes,
                                      activation = activation,
                                      supervision = True,
                                      weight_fusion = False,                           
                                      use_fpn = False,
                                      use_ps = False,
                                      ex_supervision = False,
                                      normalize = normalize)
        '''
    elif model_name == 'deeplabv3':
        model = Deeplabv3(input_shape=(w, h, img_channel),
                          classes=classes,
                          OS=8)
    elif model_name == 'unet':
        model = unet(input_size=(w, h, img_channel),
                     classes=classes,
                     activation=activation,
                     use_fpn=False,
                     use_ps=False,
                     normalize=normalize)
    elif model_name == 'original_unet_plusplus':
        model = original_unet_plusplus(input_size=(w, h, img_channel),
                                       classes=classes,
                                       activation=activation,
                                       supervision=False,
                                       weight_fusion=False,
                                       use_fpn=False,
                                       use_ps=False,
                                       ex_supervision=False,
                                       normalize=normalize)
    return model
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #tf.compat.v1.disable_eager_execution()

    print(tf.__version__)
    model = get_model(classes = 5,
                target_size = (1376,1376),
                img_channel = 3,
                epochs = None,
                learning_rate = 0.0001,
                weight_decay_rate = 0.00005,
                model_name = 'deeplabv305',
                supervision = False,
                use_ps = False,
                normalize = 'bn',
              activation='softmax')
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 1e9} G")


