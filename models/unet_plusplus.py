#!/usr/bin/env python
# coding: utf-8

# In[ ]:
'''
for tf1.13.1
'''
'''
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from tensorflow.python.keras.models import Model
from keras import backend as K
from tensorflow.python.keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from tensorflow.python.keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.layers import BatchNormalization, Dropout, Flatten, Lambda,Multiply,SeparableConv2D
from tensorflow.python.keras.layers.advanced_activations import ELU, LeakyReLU
#from keras.optimizers import Adam, RMSprop, SGD
from tensorflow.python.keras.layers.core import Reshape
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers.noise import GaussianDropout
from GroupNormalization import GroupNormalization
from pixle_shuffling import*
#from deformable_conv.deform_layer import DeformableConv2D
'''

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from tensorflow.python.keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.layers import BatchNormalization, Dropout, Flatten, Lambda,Multiply,SeparableConv2D
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers.core import Reshape
import sys
sys.path.append('models')
from GroupNormalization import GroupNormalization
from pixle_shuffling import*



'''
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *

from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda,Multiply,SeparableConv2D
from keras.layers.advanced_activations import ELU, LeakyReLU
#from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.core import Reshape
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
from GroupNormalization import GroupNormalization
'''
def relu6(x):
    return relu(x, max_value=6)
def SE_Block(inputs, reduction,step):   
    x = GlobalAveragePooling2D(name='segapool_'+step)(inputs)
    #x = BatchNormalization(momentum=0.99)(x)
    x = Dense(int(inputs.shape[-1]) // reduction, use_bias=False,name='sedense_'+step+'1' ,activation = 'relu')(x)
    #x = BatchNormalization(momentum=0.99,name='sebn_'+step)(x)
    x = Dense(int(inputs.shape[-1]), use_bias=False,name='sedense_'+step+'2' ,activation='sigmoid')(x)
    
    
    x =Reshape((1,1,inputs.shape[-1]))(x)
    
    return Multiply()([inputs,x])

def SE_Block_v2(input,reduction,step=None):
    channels = input.shape.as_list()[-1]
    avg_x = GlobalAveragePooling2D()(input)
    avg_x = Reshape((1,1,channels))(avg_x)
    avg_x = Conv2D(int(channels)//reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu',name='gavg_conv0_'+step)(avg_x)
    avg_x = Conv2D(int(channels),kernel_size=(1,1),strides=(1,1),padding='valid',name='gavg_conv1_'+step)(avg_x)

    max_x = GlobalMaxPooling2D()(input)
    max_x = Reshape((1,1,channels))(max_x)
    max_x = Conv2D(int(channels)//reduction,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu',name='gmax_conv0_'+step)(max_x)
    max_x = Conv2D(int(channels),kernel_size=(1,1),strides=(1,1),padding='valid',name='gmax_conv1_'+step)(max_x)

    cbam_feature = Add()([avg_x,max_x])

    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input,cbam_feature])
        
def scSE_block(prevlayer, prefix):
    def cse_block(prevlayer, prefix):
        #mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer) # H W 求均值
        # K.int_shape() Returns the shape of tensor or variable as a tuple of int or None entries
        mean = GlobalAveragePooling2D(name=prefix + '_gbpool')(prevlayer)
        lin1 = Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + '_cse_lin1', activation='relu')(mean)
        lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + '_cse_lin2', activation='sigmoid')(lin1)
        x = Multiply()([prevlayer, lin2])
        return x
    def sse_block(prevlayer, prefix):
        # Bug? Should be 1 here?
        conv = Conv2D(1, (1, 1), padding="valid", kernel_initializer="he_normal",
                      activation='sigmoid', strides=(1, 1),
                      name=prefix + "_conv")(prevlayer)
        conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
        return conv
    def csse_block(x, prefix):
        cse = cse_block(x, prefix)
        sse = sse_block(x, prefix)
        x = Add(name=prefix + "_csse_mul")([cse, sse])
        return x
    return csse_block(prevlayer, prefix)
def DAC_block(inputs,channel_num,step):
    def atrous_conv(inputs,channel_num,kernel_size,dilation_rate,step):
        atrous_conv = Conv2D(channel_num, kernel_size,
                        padding = 'same',
                        name='atrous_conv_'+step, 
                        dilation_rate=dilation_rate,
                        kernel_initializer = 'he_normal')(inputs)
        atrous_conv = GroupNormalization(name='atrous_conv_'+step+'_GN')(atrous_conv)
        atrous_conv = Activation('relu')(atrous_conv)
        return atrous_conv
    x = Conv2D(channel_num//2, 1,padding = 'same',name='ceneck_conv1_'+step, kernel_initializer = 'he_normal')(inputs)
    x = GroupNormalization(name='ceneck_gn1_'+step)(x)
    x = Activation('relu')(x)
    atrous_conv_11 = atrous_conv(x,channel_num//2,3,1,step+'11')
    atrous_conv_12 = atrous_conv(x,channel_num//2,3,3,step+'12')
    atrous_conv_13 = atrous_conv(x,channel_num//2,3,1,step+'13')
    atrous_conv_14 = atrous_conv(x,channel_num//2,3,1,step+'14')    
    atrous_conv_21 = atrous_conv(atrous_conv_12,channel_num//2,1,1,step+'21')
    atrous_conv_22 = atrous_conv(atrous_conv_13,channel_num//2,3,3,step+'22')
    atrous_conv_23 = atrous_conv(atrous_conv_14,channel_num//2,3,3,step+'23')
    atrous_conv_31 = atrous_conv(atrous_conv_22,channel_num//2,1,1,step+'31')
    atrous_conv_32 = atrous_conv(atrous_conv_23,channel_num//2,3,5,step+'32')
    atrous_conv_41 = atrous_conv(atrous_conv_32,channel_num//2,1,1,step+'41')
    x = concatenate([atrous_conv_11,atrous_conv_21,atrous_conv_31,atrous_conv_41,x],name='atrous_conv_out_'+step, axis = 3)     
    x = Conv2D(channel_num, 1,padding = 'same',name='ceneck_conv2_'+step, kernel_initializer = 'he_normal')(x)
    x = GroupNormalization(name='ceneck_gn2_'+step)(x)
    x = Activation('relu')(x)    
    return x
def RMP_block(inputs,channel_num,step):
    x = Conv2D(channel_num//4, 1,padding = 'same',name='rmpneck_conv1', kernel_initializer = 'he_normal')(inputs)
    x = GroupNormalization(name='rmpneck_gn1')(x)
    x = Activation('relu')(x)  
    pool1 = AveragePooling2D(pool_size=(2, 2),name='rmppool_1')(x)
    pool1 = Lambda(lambda image: tf.compat.v1.image.resize(image, inputs.shape[1:3],method = 0))(pool1)
    pool2 = AveragePooling2D(pool_size=(3, 3),name='rmppool_2')(x)
    pool2 = Lambda(lambda image: tf.compat.v1.image.resize(image, inputs.shape[1:3],method = 0))(pool2)
    pool3 = AveragePooling2D(pool_size=(5, 5),name='rmppool_3')(x)
    pool3 = Lambda(lambda image: tf.compat.v1.image.resize(image, inputs.shape[1:3],method = 0))(pool3)
    pool4 = AveragePooling2D(pool_size=(6, 6),name='rmppool_4')(x)
    pool4 = Lambda(lambda image: tf.compat.v1.image.resize(image, inputs.shape[1:3],method = 0))(pool4)
    x = concatenate([pool1,pool2,pool3,pool4,inputs],name='rmp_out', axis = 3)
    x = Conv2D(channel_num, 1,padding = 'same',name='rmpneck_conv2', kernel_initializer = 'he_normal')(x)
    x = GroupNormalization(momentum=0.99 ,name='rmpneck_gn2')(x)
    x = Activation('relu')(x)
    return x

def conv_bn(inputs,channel_size,kernel_size,strides=1,step = None,active = True,normalize = 'gn',group=32):
    if normalize == None:   #卷积层后有标准化层不需要加偏置
        use_bias = True
    else:
        use_bias = True
    x = Conv2D(channel_size,kernel_size,padding = 'same', strides=strides,
               name='conv_'+step, kernel_initializer = 'he_normal', use_bias = use_bias)(inputs)
    if normalize == 'bn':         #batch-normalization or group-normalization
        x = BatchNormalization(momentum=0.99,name='bn_'+step)(x)
        if active == True:
             x = Activation('relu')(x)
    elif normalize == 'gn':
       # group = group
       # if channel_size <= 64:
       #     group = 8
       # else:
       #     group = 32
        x = GroupNormalization(name='gn_'+step,group = group)(x)
       # x = WeightNormalization(Conv2D(channel_size,kernel_size,padding = 'same', strides=strides,
       #       name='conv_'+step, kernel_initializer = 'he_normal'))(inputs)
        if active == True:
             x = Activation('relu')(x)
    elif normalize == None:
        x = x
        if active == True:
             x = Activation('relu')(x)

    return x

def up_block(inputs,channel_size,input_size,step = None,active = True,linear=True,normalize = 'gn',group=32, ps=False):
    if normalize == None:   #卷积层后有标准化层不需要加偏置
        use_bias = True
    else:
        use_bias = True
    if linear == True:
        x = Lambda(lambda x: tf.compat.v1.image.resize(x,(input_size[0],input_size[1]),method = 0, align_corners=True))(inputs)
    elif ps == True:
        x = SubpixelConv2D(upsampling_factor=2)(inputs)
    else:
        x = UpSampling2D(size = (2,2))(inputs)
    x = Conv2D(channel_size,2,padding = 'same',
               name='upconv_'+step, kernel_initializer = 'he_normal', use_bias = use_bias)(x)
    if normalize == 'bn':  # batch-normalization or group-normalization
        x = BatchNormalization(momentum=0.99, name='bn_' + step)(x)
    elif normalize == 'gn':
        group = group
        x = GroupNormalization(name='gn_' + step, group=group)(x)
    elif normalize == None:
        x = x
    if active == True:
        x = Activation('relu')(x)
    x = Conv2D(channel_size, 2, padding='same',
               name='ex_upconv_' + step, kernel_initializer='he_normal', use_bias = use_bias)(x)
    if normalize == 'bn':  # batch-normalization or group-normalization
        x = BatchNormalization(momentum=0.99, name='ex_bn_' + step)(x)
    elif normalize == 'gn':
        group = group
        x = GroupNormalization(name='ex_gn_' + step, group=group)(x)
    elif normalize == None:
        x = x
    if active == True:
        x = Activation('relu')(x)
    return x

def mini_up_block(inputs,channel_size,input_size,step = None,active = True,linear=True,normalize = 'gn',group=32):
    if normalize == None:   #卷积层后有标准化层不需要加偏置
        use_bias = True
    else:
        use_bias = True
    if linear == True:
        x = Lambda(lambda x: tf.compat.v1.image.resize(x,(input_size[0],input_size[1]),method = 0, align_corners=True))(inputs)
    else:
        x = UpSampling2D(size = (2,2))(inputs)
    
    x = Conv2D(channel_size,2,padding = 'same',
               name='upconv_'+step, kernel_initializer = 'he_normal', use_bias = use_bias)(x)
    if normalize == 'bn':  # batch-normalization or group-normalization
        x = BatchNormalization(momentum=0.99, name='bn_' + step)(x)
    elif normalize == 'gn':
        group = group
        x = GroupNormalization(name='gn_' + step, group=group)(x)
    elif normalize == None:
        x = x
    if active == True:
        x = Activation('relu')(x)
    
    return x

def atrous_conv(inputs,channel_num,kernel_size,dilation_rate,step=None,normalize ='gn'):
    if normalize == None:   #卷积层后有标准化层不需要加偏置
        use_bias = True
    else:
        use_bias = True
    atrous_conv = Conv2D(channel_num, kernel_size,
                    padding = 'same',
                    name='atrous_conv_'+step, 
                    dilation_rate=dilation_rate,
                    kernel_initializer = 'he_normal',
                    use_bias = use_bias)(inputs)
    if normalize == 'bn':  # batch-normalization or group-normalization
        atrous_conv = BatchNormalization(momentum=0.99, name='ex_bn_' + step)(atrous_conv)
    elif normalize == 'gn':
        group = 32
        atrous_conv = GroupNormalization(name='ex_gn_' + step, group=group)(atrous_conv)
    elif normalize == None:
        atrous_conv = atrous_conv
    atrous_conv = Activation('relu')(atrous_conv)
    return atrous_conv
    
def aspp(inputs, channel_num, step, normalize='gn'):
    x0 = GlobalAveragePooling2D()(inputs)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    x0 = Lambda(lambda x: K.expand_dims(x, 1))(x0)
    x0 = Lambda(lambda x: K.expand_dims(x, 1))(x0)
    x0 = conv_bn(x0, channel_num, 1, step='img_pooling', normalize=normalize)
    size_before = tf.keras.backend.int_shape(inputs)
    x0 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method=0, align_corners=True))(x0)
    x1 = conv_bn(inputs, channel_num, 1, step='aspp_conv1_' + step, normalize=normalize)
    x2 = atrous_conv(inputs, channel_num, 3, 1, step=step + '1', normalize=normalize)
    x3 = atrous_conv(inputs, channel_num, 3, 3, step=step + '2', normalize=normalize)
    x4 = atrous_conv(inputs, channel_num, 3, 5, step=step + '3', normalize=normalize)
    x = concatenate([x1, x2, x3, x4, x0], axis=3, name='aspp_conc_' + step)
    x = conv_bn(x, channel_num, 1, step='aspp_conv2_' + step, normalize=normalize)
    return x

def standard_block(inputs,channel_size,kernel_size,step = None,normalize = 'gn',res=False,dense=False,scse=False,group=32):    
    
    '''        
    if deform==True:
        x = DeformableConv2D(channel_size,name='offset_'+step+'1')(inputs)
   ''' 
   
    

    if res == True and step not in ['00x','10','20','30','40']:
        r = conv_bn(inputs, channel_size, 1, step='stres_' + step, active=False, normalize=normalize)
        #x = GroupNormalization(name='stres_gn_' + step, group=group)(inputs)
        #x = Activation('relu')(x)
        #x = conv_bn(inputs,channel_size,kernel_size,step = step+'1', active=True ,normalize = normalize)
        #r = inputs
        #if step in ['00x','10','20','30','40']:
        x = conv_bn(inputs, channel_size, kernel_size, step=step + '1', active=True, normalize=normalize)
        x = conv_bn(x, channel_size, kernel_size, step=step + '2', active=True, normalize=normalize)
        x = conv_bn(x,channel_size,kernel_size,step = step+'3',active = False,normalize = normalize)
        #r = SE_Block(r, 4,step)
        x = add([x,r],name='st_add_'+step)
        
        #x = GroupNormalization(name='dense2_gn_' + step, group=group)(inputs)
        x = Activation('relu')(x)
        #x = TLU()(x)
        #else:
            #x = conv_bn(x,channel_size,kernel_size,step = step+'2',normalize = normalize)
        '''
        if step not in ['00','01','02','03','04']:
            x = conv_bn(x,channel_size,kernel_size,step = step+'2',active = False,normalize = normalize) 

            r = conv_bn(inputs,channel_size,1,step = 'res_'+step,active = False,normalize = normalize)
            r = SE_Block(r, 4,step)
            x = add([x,r],name='add_'+step)
            x = Activation('relu')(x)
       
        else:
      
            x = conv_bn(x,channel_size,kernel_size,step = step+'2',normalize = normalize)   
        '''
    elif dense == True and step not in ['00x','10','20','30','40']:
        x1 = conv_bn(inputs, channel_size, kernel_size, step=step + 'dense1', active=False, normalize=None, group=group)
        c1 = concatenate([inputs,x1], axis=3)
        c1 = GroupNormalization(name='dense1_gn_' + step, group=group)(c1)
        c1 = Activation('relu')(c1)
        x2 = conv_bn(c1, channel_size, kernel_size, step=step + 'dense2', active=False, normalize=None, group=group)
        c2 = concatenate([x2,x1,inputs], axis=3)
        c2 = GroupNormalization(name='dense2_gn_' + step, group=group)(c2)
        c2 = Activation('relu')(c2)
        x = conv_bn(c2, channel_size, 1, step=step + 'dense3', active=True, normalize=normalize, group=group)
    else:
        '''
        if deform==True:
            x = DeformableConv2D(channel_size,name='offset_'+step+'2')(x)
            
        '''    
        x = conv_bn(inputs,channel_size,kernel_size,step = step+'1', active=True, normalize = normalize,group=group)
        '''if step=='04':
            x = conv_bn(x, channel_size, kernel_size, step=step + '2', active=True, normalize=None, group=group)
        else:'''
        x = conv_bn(x,channel_size,kernel_size,step = step+'2', active=True, normalize = normalize,group=group)
        #x = conv_bn(x,channel_size,kernel_size,step = step+'3', active=True, normalize = normalize,group=group)
    if scse == True:
        x = scSE_block(x, step)
    '''
        elif msm==True:
        if step in ['10','20','30','40']:
            x = aspp(x,channel_size,step)
    '''

    return x

#pre-activation 00:3,01:4,02:4,03:4,04:3
def res_block(inputs, channel_size, kernel_size, step=None, normalize='gn', down_sample=False, group=32):
    if down_sample:
        r1 = conv_bn(inputs, channel_size, 1, step=step + 'res', active=False, normalize=None)
    else:
        r1 = inputs
    if normalize == 'gn':
        x1 = GroupNormalization(name='gn_' + step + '0', group=group)(inputs)
    else:
        x1 = BatchNormalization(name='bn_' + step + '0')(inputs)
    x1 = Activation('relu')(x1)
    x1 = conv_bn(x1, channel_size, kernel_size, step=step + '1', active=True, normalize=normalize, group=group)
    x1 = conv_bn(x1, channel_size, kernel_size, step=step + '2', active=False, normalize=None)
    x1 = add([x1,r1],name='add_1_'+step)
    r2 = x1
    if normalize == 'gn':
        x2 = GroupNormalization(name='gn_' + step + '2', group=32)(x1)
    else:
        x2 = BatchNormalization(name='bn_' + step + '2')(x1)
    x2 = Activation('relu')(x2)
    x2 = conv_bn(x2, channel_size, kernel_size, step=step + '3', active=True, normalize=normalize, group=group)
    x2 = conv_bn(x2, channel_size, kernel_size, step=step + '4', active=False, normalize=None)
    x2 = add([x2, r2], name='add_2_' + step)
    r3 = x2
    if normalize == 'gn':
        x3 = GroupNormalization(name='gn_' + step + '4', group=group)(x2)
    else:
        x3 = BatchNormalization(name='bn_' + step + '4')(x2)
    x3 = Activation('relu')(x3)   
    x3= conv_bn(x3, channel_size, kernel_size, step=step + '5', active=True, normalize=normalize, group=group)
    x3 = conv_bn(x3, channel_size, kernel_size, step=step + '6', active=False, normalize=None)
    x3 = add([x3, r3], name='add_3_' + step)
    r4 = x3
    if normalize == 'gn':
        x4 = GroupNormalization(name='gn_' + step + '6', group=group)(x3)
    else:
        x4 = BatchNormalization(name='bn_' + step + '6')(x3)
    x4 = Activation('relu')(x4)   
    x4 = conv_bn(x4, channel_size, kernel_size, step=step + '7', active=True, normalize=normalize, group=group)
    x4 = conv_bn(x4, channel_size, kernel_size, step=step + '8', active=False, normalize=None)
    x4 = add([x4, r4], name='add_4_' + step)
    if normalize == 'gn':
        x5 = GroupNormalization(name='gn_' + step + '8', group=group)(x4)
    else:
        x5 = BatchNormalization(name='bn_' + step + '8')(x4)
    x5 = Activation('relu')(x5)
    if step in ['00','40']:
        x = x3
        x_decoder =  GroupNormalization(name='gn_' + step + '6', group=group)(x3)
        x_decoder =  Activation('relu')(x_decoder)   
    else:
        x = x4
        x_decoder = x5
    if step == '40':
        x = x_decoder
    return x, x_decoder

def dense_conv(x, channel_size, kernel_size, step=None, normalize='gn',group=32):
    #x = GroupNormalization(name='sqgn_' + step, group=group)(x)
    #x = Activation('relu')(x)
    x = conv_bn(x,channel_size,kernel_size,step = step, active=True, normalize = None,group=group)
    return x
def dense_block(x, channel_size, kernel_size, step=None, normalize='gn',group=24,r=4):
    if step != '00':
        channels = x.shape.as_list()[-1]
        #x = GroupNormalization(name='sqgn_' + step, group=group)(x)
        #x = Activation('relu')(x)
        x = conv_bn(x,channels//2,1,step = step+'_sqeeze', active=True, normalize = normalize,group=group)
    for i in range(r):
        c = dense_conv(x,channel_size,kernel_size,step = step+'dense_'+str(i),  normalize = normalize,group=group)
        x = concatenate([c, x], axis=3)
    return x
'''
def AG_block(x1, x2, f_int, step = None, normalize = 'gn'):
    x1_h, x1_w, x1_c = K.int_shape(x1)[1], K.int_shape(x1)[2], K.int_shape(x1)[3]
    
    xl = conv_bn(x1, f_int, 3, strides=2, step='AGl_' + step, active=False, normalize=normalize)
    xg = conv_bn(x2, f_int, 1, step='AGg_' + step, active=False, normalize=normalize)

    x = add([xl, xg], name='AG_add_' + step)
    if normalize == 'frn':
        x = TLU()(x)
    else:
        x = Activation('relu')(x)
    x = conv_bn(x, 1, 1, step='AG_' + step, active=False, normalize=None)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda image: tf.compat.v1.image.resize(image, (x1_h, x1_w),method = 2, align_corners=True))(x)
    x1 = Multiply()([x1, x])
    return x1
'''
def AG_block(x1, x2, f_int, step = None, normalize = 'gn'):
    x1_h, x1_w, x1_c = K.int_shape(x1)[1], K.int_shape(x1)[2], K.int_shape(x1)[3]
    xl = conv_bn(x1, x1_c, 3, strides=2, step='AGl_' + step, active=False, normalize=normalize)
    xg = conv_bn(x2, x1_c, 1, step='AGg_' + step, active=False, normalize=normalize)
    x = add([xl, xg], name='AG_add_' + step)
    x = Activation('relu')(x)
    x = conv_bn(x, 1, 1, step='AG_' + step, active=False, normalize=None)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda image: tf.compat.v1.image.resize(image, (x1_h, x1_w),method = 2, align_corners=True))(x)
    x1 = Multiply()([x1, x])
    return x1

    
def merge_block(inputs_list,channel_size,step = None,merge = False,normalize = 'bn'):
    x = concatenate(inputs_list,axis = 3,name = 'conc_'+step)
    if merge == True:
        x = conv_bn(x,channel_size,1,'sqconv_'+step,normalize = normalize)
    return x


def fpnv2(inputs, normalize='gn'):
    fpn1 = DeSubpixelConv2D(downsampling_factor=2)(inputs)  # 48

    fpn2 = DeSubpixelConv2D(downsampling_factor=2)(fpn1)  # 192
    fpn2 = conv_bn(fpn2, 64, 1, step='fpn2', active=True, normalize=normalize)
    fpn3 = DeSubpixelConv2D(downsampling_factor=2)(fpn2)  # 576
    fpn3 = conv_bn(fpn3, 128, 1, step='fpn3', active=True, normalize=normalize)
    fpn4 = DeSubpixelConv2D(downsampling_factor=2)(fpn3)  # 2304
    fpn4 = conv_bn(fpn4, 256, 1, step='fpn4', active=True, normalize=normalize)
    return fpn1, fpn2, fpn3, fpn4

def res_unet_plusplus(pretrained_weights=None,
                  input_size=(512, 512, 1),
                  classes=3,
                  activation='softmax',
                  supervision=True,
                  ex_supervision=False,
                  weight_fusion=False,
                  use_fpn=False,
                  use_ps=False,
                  normalize='bn'):
    inputs = Input(input_size)
    w, h = input_size[0], input_size[1]

    # subnet1
    if use_ps == True:
        w, h = w // 2, h // 2
        x = DeSubpixelConv2D(downsampling_factor=2)(inputs)
        conv00 = conv_bn(x,64,3,step='input_conv',active=False,normalize=None)
        block_00_backbone, block_00 = res_block(conv00, 64, 3, step='00', normalize=normalize, down_sample=False)
    else:
        # inputs = conv_bn(inputs,12,1,step = 'input',active = True,normalize = 'gn',group=12)  #调整通道数和Subpixel后一样
        w, h = w // 2, h // 2
        x = Lambda(lambda image: tf.compat.v1.image.resize(image, (w, h), method=0))(inputs)
        x = Conv2D(12, 1, name='input_conv_ex', activation=None, kernel_initializer='he_normal')(x)
        conv00 = conv_bn(x, 64, 3, step='input_conv1', active=False, normalize=None)
        block_00_backbone, block_00 = res_block(conv00, 64, 3, step='00', normalize=normalize, down_sample=False)

    pool_10 = MaxPooling2D(pool_size=(2, 2), name='maxpool_10')(block_00_backbone)
    if use_fpn == True:
        # fpn1,fpn2,fpn3,fpn4 = fpn(inputs,(w,h),channel_list=[32,64,128,256],normalize=normalize)  #use feature pr
        fpn1, fpn2, fpn3, fpn4 = fpnv2(x, normalize=normalize)
        pool_10 = merge_block([fpn1, pool_10], 64, step='fpn1', merge=True, normalize=normalize)  # fpn1
    block_10_backbone, block_10 = res_block(pool_10, 64, 3, step='10', normalize=normalize, down_sample=True)
    up_01 = up_block(block_10, 64, (w, h), step='01', normalize=normalize)
    # merge_01 = concatenate([up_01,block_00],axis = 3,name = 'conc_01')
    #block_00 = AG_block(block_00, block_10, 64, step='00', normalize=normalize)
    merge_01 = merge_block([up_01, block_00], 64, step='01', normalize=normalize)
    block_01 = standard_block(merge_01, 64, 3, step='01', normalize=normalize)

    # subnet2
    pool_20 = MaxPooling2D(pool_size=(2, 2), name='maxpool_20')(block_10_backbone)
    if use_fpn == True:
        pool_20 = merge_block([fpn2, pool_20], 64, step='fpn2', merge=True, normalize=normalize)  # fpn2
    block_20_backbone, block_20 = res_block(pool_20, 128, 3, step='20', normalize=normalize, down_sample=True)
    up_11 = up_block(block_20, 64, (w // 2, h // 2), step='11', normalize=normalize)
    # merge_11 = concatenate([up_11,block_10],axis = 3,name = 'conc_11')
    #block_10 = AG_block(block_10, block_20, 64, step='10', normalize=normalize)
    merge_11 = merge_block([up_11, block_10], 64, step='11', normalize=normalize)
    block_11 = standard_block(merge_11, 64, 3, step='11', normalize=normalize)
    up_02 = up_block(block_11, 64, (w, h), step='02', normalize=normalize)
    # merge_02 = concatenate([up_02,block_01,block_00],axis = 3,name = 'conc_02')
    #block_01 = AG_block(block_01, block_11, 64, step='01', normalize=normalize)
    merge_02 = merge_block([up_02, block_01, block_00], 64, step='02', normalize=normalize)
    block_02 = standard_block(merge_02, 64, 3, step='02', normalize=normalize)

    # subnet3
    pool_30 = MaxPooling2D(pool_size=(2, 2), name='maxpool_30')(block_20_backbone)
    if use_fpn == True:
        pool_30 = merge_block([fpn3, pool_30], 128, step='fpn3', merge=True, normalize=normalize)  # fpn3
    block_30_backbone, block_30 = res_block(pool_30, 256, 3, step='30', normalize=normalize, down_sample=True)
    # block_30 = Dropout(0.05,name='dp_1')(block_30)
    up_21 = up_block(block_30, 128, (w // 4, h // 4), step='21', normalize=normalize)
    # merge_21 = concatenate([up_21,block_20],axis = 3,name = 'conc_21')
    #block_20 = AG_block(block_20, block_30, 128, step='20', normalize=normalize)
    merge_21 = merge_block([up_21, block_20], 128, step='21', normalize=normalize)
    block_21 = standard_block(merge_21, 128, 3, step='21', normalize=normalize)
    up_12 = up_block(block_21, 64, (w // 2, h // 2), step='12', normalize=normalize)
    # merge_12 = concatenate([up_12,block_11,block_10],axis = 3,name = 'conc_12')
    #block_11 = AG_block(block_11, block_21, 64, step='11', normalize=normalize)
    merge_12 = merge_block([up_12, block_11, block_10], 64, step='12', normalize=normalize)
    block_12 = standard_block(merge_12, 64, 3, step='12', normalize=normalize)
    up_03 = up_block(block_12, 64, (w, h), step='03', normalize=normalize)
    # merge_03 = concatenate([up_03,block_02,block_01,block_00],axis = 3,name = 'conc_03')
    #block_02 = AG_block(block_02, block_12, 64, step='02', normalize=normalize)
    merge_03 = merge_block([up_03, block_02, block_01, block_00], 64, step='03', normalize=normalize)
    block_03 = standard_block(merge_03, 64, 3, step='03', normalize=normalize)

    # subnet4
    pool_40 = MaxPooling2D(pool_size=(2, 2), name='maxpool_40')(block_30_backbone)
    if use_fpn == True:
        pool_40 = merge_block([fpn4, pool_40], 256, step='fpn4', merge=True, normalize=normalize)  # fpn4
    block_40_backbone, block_40 = res_block(pool_40, 512, 3, step='40', normalize=normalize, down_sample=True)
    block_40 = Dropout(0.2, name='dp_2')(block_40)
    up_31 = up_block(block_40, 256, (w // 8, h // 8), step='31', normalize=normalize)
    # merge_31 = concatenate([up_31,block_30],axis = 3,name = 'conc_31')
    #block_30 = AG_block(block_30, block_40, 256, step='30', normalize=normalize)
    merge_31 = merge_block([up_31, block_30], 256, step='31', normalize=normalize)
    block_31 = standard_block(merge_31, 256, 3, step='31', normalize=normalize)
    up_22 = up_block(block_31, 128, (w // 4, h // 4), step='22', normalize=normalize)
    # merge_22 = concatenate([up_22,block_21,block_20],axis = 3,name = 'conc_22')
    #block_21 = AG_block(block_21, block_31, 128, step='21', normalize=normalize)
    merge_22 = merge_block([up_22, block_21, block_20], 128, step='22', normalize=normalize)
    block_22 = standard_block(merge_22, 128, 3, step='22', normalize=normalize)
    up_13 = up_block(block_22, 64, (w // 2, h // 2), step='13', normalize=normalize)
    # merge_13 = concatenate([up_13,block_12,block_11,block_10],axis = 3,name = 'conc_13')
    #block_12 = AG_block(block_12, block_22, 64, step='12', normalize=normalize)
    merge_13 = merge_block([up_13, block_12, block_11, block_10], 64, step='13', normalize=normalize)
    block_13 = standard_block(merge_13, 64, 3, step='13', normalize=normalize)
    up_04 = up_block(block_13, 64, (w, h), step='04', normalize=normalize)
    # merge_04 = concatenate([up_04,block_03,block_02,block_01,block_00],axis = 3,name = 'conc_04')
    #block_03 = AG_block(block_03, block_13, 64, step='03', normalize=normalize)
    merge_04 = merge_block([up_04, block_03, block_02, block_01, block_00], 32, step='04', normalize=normalize)
    block_04 = standard_block(merge_04, 64, 3, step='04', normalize=normalize)

    if supervision == True:
        output1 = block_01

        output2 = block_02

        output3 = block_03

        output4 = block_04

        fusion = concatenate([output1, output2, output3, output4], axis=3, name='conc_fn')
        
        output = Conv2D(256, 1, padding='same',
                            name='conv_fn2', kernel_initializer='he_normal')(fusion)
                            
        #output = BatchNormalization(momentum=0.99,name='bn_'+'final')(output)
        
        output = Activation('relu')(output)
        
        if use_ps == True:
            output = SubpixelConv2D(upsampling_factor=2)(output)

            # for 1 lesion
            # output = Conv2D(classes,(1, 1), activation = activation,name='output_conv3', kernel_initializer = 'he_normal')(output)
            # for 4 lesions
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv2',
                            kernel_initializer='he_normal')(output)
        else:
            output = Conv2D(64, 1, name='ex_conv', activation='relu', kernel_initializer='he_normal')(output)  # 调整通道数和Subpixel后一样
            #output = conv_bn(output,48,1,step='input_conv',active=True,normalize=normalize)
            output = Lambda(
                lambda image: tf.compat.v1.image.resize(image, (input_size[0], input_size[1]), method=0))(output)
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv3',
                            kernel_initializer='he_normal')(output)

    else:
        output = block_04
        if use_ps == True:
            output = SubpixelConv2D(upsampling_factor=2)(output)
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv', kernel_initializer='he_normal')(
                output)
        else:
            output = Conv2D(16, 1, name='ex_conv', activation='relu', kernel_initializer='he_normal')(output)
            output = Lambda(lambda image: tf.compat.v1.image.resize(image, (input_size[0], input_size[1]), method=0))(
                output)

            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv1',
                            kernel_initializer='he_normal')(output)

    model = Model(inputs, output)
    print(model.summary())
    return model
def fres_unet_plusplus(pretrained_weights=None,
                  input_size=(512, 512, 1),
                  classes=3,
                  activation='softmax',
                  supervision=True,
                  ex_supervision=False,
                  weight_fusion=False,
                  use_fpn=False,
                  use_ps=False,
                  normalize='bn'):
    inputs = Input(input_size)
    w, h = input_size[0], input_size[1]

    # subnet1
    if use_ps == True:
        w, h = w // 2, h // 2
        x = DeSubpixelConv2D(downsampling_factor=2)(inputs)
        conv00 = conv_bn(x,64,3,step='input_conv',active=False,normalize=None)
        block_00_backbone, block_00 = res_block(conv00, 64, 3, step='00', normalize=normalize, down_sample=False)
    else:
        # inputs = conv_bn(inputs,12,1,step = 'input',active = True,normalize = 'gn',group=12)  #调整通道数和Subpixel后一样
        w, h = w // 2, h // 2
        x = Lambda(lambda image: tf.compat.v1.image.resize(image, (w, h), method=0))(inputs)
        x = Conv2D(12, 1, name='input_conv', activation=None, kernel_initializer='he_normal')(x)
        conv00 = conv_bn(x, 64, 3, step='input_conv', active=False, normalize=None)
        block_00_backbone, block_00 = res_block(conv00, 64, 3, step='00', normalize=normalize, down_sample=False)

    pool_10 = MaxPooling2D(pool_size=(2, 2), name='maxpool_10')(block_00_backbone)
    if use_fpn == True:
        # fpn1,fpn2,fpn3,fpn4 = fpn(inputs,(w,h),channel_list=[32,64,128,256],normalize=normalize)  #use feature pr
        fpn1, fpn2, fpn3, fpn4 = fpnv2(x, normalize=normalize)
        pool_10 = merge_block([fpn1, pool_10], 64, step='fpn1', merge=True, normalize=normalize)  # fpn1
    block_10_backbone, block_10 = res_block(pool_10, 64, 3, step='10', normalize=normalize, down_sample=True)
    up_01 = up_block(block_10, 64, (w, h), step='01', normalize=normalize)
    # merge_01 = concatenate([up_01,block_00],axis = 3,name = 'conc_01')
    #block_00 = AG_block(block_00, block_10, 64, step='00', normalize=normalize)
    merge_01 = merge_block([up_01, block_00], 64, step='01', normalize=normalize)
    block_01 = standard_block(merge_01, 64, 3, step='01', res=True, normalize=normalize)

    # subnet2
    pool_20 = MaxPooling2D(pool_size=(2, 2), name='maxpool_20')(block_10_backbone)
    if use_fpn == True:
        pool_20 = merge_block([fpn2, pool_20], 64, step='fpn2', merge=True, normalize=normalize)  # fpn2
    block_20_backbone, block_20 = res_block(pool_20, 128, 3, step='20', normalize=normalize, down_sample=True)
    up_11 = up_block(block_20, 64, (w // 2, h // 2), step='11', normalize=normalize)
    # merge_11 = concatenate([up_11,block_10],axis = 3,name = 'conc_11')
    #block_10 = AG_block(block_10, block_20, 64, step='10', normalize=normalize)
    merge_11 = merge_block([up_11, block_10], 64, step='11', normalize=normalize)
    block_11 = standard_block(merge_11, 64, 3, step='11', res=True, normalize=normalize)
    up_02 = up_block(block_11, 64, (w, h), step='02', normalize=normalize)
    # merge_02 = concatenate([up_02,block_01,block_00],axis = 3,name = 'conc_02')
    #block_01 = AG_block(block_01, block_11, 64, step='01', normalize=normalize)
    merge_02 = merge_block([up_02, block_01, block_00], 64, step='02', normalize=normalize)
    block_02 = standard_block(merge_02, 64, 3, step='02', res=True, normalize=normalize)

    # subnet3
    pool_30 = MaxPooling2D(pool_size=(2, 2), name='maxpool_30')(block_20_backbone)
    if use_fpn == True:
        pool_30 = merge_block([fpn3, pool_30], 128, step='fpn3', merge=True, normalize=normalize)  # fpn3
    block_30_backbone, block_30 = res_block(pool_30, 256, 3, step='30', normalize=normalize, down_sample=True)
    # block_30 = Dropout(0.05,name='dp_1')(block_30)
    up_21 = up_block(block_30, 128, (w // 4, h // 4), step='21', normalize=normalize)
    # merge_21 = concatenate([up_21,block_20],axis = 3,name = 'conc_21')
    #block_20 = AG_block(block_20, block_30, 128, step='20', normalize=normalize)
    merge_21 = merge_block([up_21, block_20], 128, step='21', normalize=normalize)
    block_21 = standard_block(merge_21, 128, 3, step='21', res=True, normalize=normalize)
    up_12 = up_block(block_21, 64, (w // 2, h // 2), step='12', normalize=normalize)
    # merge_12 = concatenate([up_12,block_11,block_10],axis = 3,name = 'conc_12')
    #block_11 = AG_block(block_11, block_21, 64, step='11', normalize=normalize)
    merge_12 = merge_block([up_12, block_11, block_10], 64, step='12', normalize=normalize)
    block_12 = standard_block(merge_12, 64, 3, step='12', res=True, normalize=normalize)
    up_03 = up_block(block_12, 64, (w, h), step='03', normalize=normalize)
    # merge_03 = concatenate([up_03,block_02,block_01,block_00],axis = 3,name = 'conc_03')
    #block_02 = AG_block(block_02, block_12, 64, step='02', normalize=normalize)
    merge_03 = merge_block([up_03, block_02, block_01, block_00], 64, step='03', normalize=normalize)
    block_03 = standard_block(merge_03, 64, 3, step='03', res=True, normalize=normalize)

    # subnet4
    pool_40 = MaxPooling2D(pool_size=(2, 2), name='maxpool_40')(block_30_backbone)
    if use_fpn == True:
        pool_40 = merge_block([fpn4, pool_40], 256, step='fpn4', merge=True, normalize=normalize)  # fpn4
    block_40_backbone, block_40 = res_block(pool_40, 512, 3, step='40', normalize=normalize, down_sample=True)
    block_40 = Dropout(0.2, name='dp_2')(block_40)
    up_31 = up_block(block_40, 256, (w // 8, h // 8), step='31', normalize=normalize)
    # merge_31 = concatenate([up_31,block_30],axis = 3,name = 'conc_31')
    #block_30 = AG_block(block_30, block_40, 256, step='30', normalize=normalize)
    merge_31 = merge_block([up_31, block_30], 256, step='31', normalize=normalize)
    block_31 = standard_block(merge_31, 256, 3, step='31', res=True, normalize=normalize)
    up_22 = up_block(block_31, 128, (w // 4, h // 4), step='22', normalize=normalize)
    # merge_22 = concatenate([up_22,block_21,block_20],axis = 3,name = 'conc_22')
    #block_21 = AG_block(block_21, block_31, 128, step='21', normalize=normalize)
    merge_22 = merge_block([up_22, block_21, block_20], 128, step='22', normalize=normalize)
    block_22 = standard_block(merge_22, 128, 3, step='22', res=True, normalize=normalize)
    up_13 = up_block(block_22, 64, (w // 2, h // 2), step='13', normalize=normalize)
    # merge_13 = concatenate([up_13,block_12,block_11,block_10],axis = 3,name = 'conc_13')
    #block_12 = AG_block(block_12, block_22, 64, step='12', normalize=normalize)
    merge_13 = merge_block([up_13, block_12, block_11, block_10], 64, step='13', normalize=normalize)
    block_13 = standard_block(merge_13, 64, 3, step='13', res=True, normalize=normalize)
    up_04 = up_block(block_13, 64, (w, h), step='04', normalize=normalize)
    # merge_04 = concatenate([up_04,block_03,block_02,block_01,block_00],axis = 3,name = 'conc_04')
    #block_03 = AG_block(block_03, block_13, 64, step='03', normalize=normalize)
    merge_04 = merge_block([up_04, block_03, block_02, block_01, block_00], 32, step='04', normalize=normalize)
    block_04 = standard_block(merge_04, 64, 3, step='04', res=True, normalize=normalize)

    if supervision == True:
        output1 = block_01

        output2 = block_02

        output3 = block_03

        output4 = block_04

        fusion = concatenate([output1, output2, output3, output4], axis=3, name='conc_fn')
        #fusion = GroupNormalization(name='conv_fn1_gn_', group=32)(fusion)
        #fusion = Activation('relu')(fusion)
        output = Conv2D(256, 1, padding='same',
                            name='conv_fn2', kernel_initializer='he_normal')(fusion)
        output = Activation('relu')(output)
        if use_ps == True:
            output = SubpixelConv2D(upsampling_factor=2)(output)

            # for 1 lesion
            # output = Conv2D(classes,(1, 1), activation = activation,name='output_conv3', kernel_initializer = 'he_normal')(output)
            # for 4 lesions
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv2',
                            kernel_initializer='he_normal')(output)
        else:
            output = Conv2D(48, 1, name='ex_conv', activation=None, kernel_initializer='he_normal')(
                output)  # 调整通道数和Subpixel后一样
            output = Lambda(
                lambda image: tf.compat.v1.image.resize(image, (input_size[0], input_size[1]), method=0))(output)
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv3',
                            kernel_initializer='he_normal')(output)

    else:
        output = block_04
        if use_ps == True:
            output = SubpixelConv2D(upsampling_factor=2)(output)
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv', kernel_initializer='he_normal')(
                output)
        else:
            output = Conv2D(16, 1, name='ex_conv', activation=None, kernel_initializer='he_normal')(output)
            output = Lambda(lambda image: tf.compat.v1.image.resize(image, (input_size[0], input_size[1]), method=0))(
                output)

            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv1',
                            kernel_initializer='he_normal')(output)

    model = Model(inputs, output)
    print(model.summary())
    return model
def resat_unet_plusplus(pretrained_weights=None,
                  input_size=(512, 512, 1),
                  classes=3,
                  activation='softmax',
                  supervision=True,
                  ex_supervision=False,
                  weight_fusion=False,
                  use_fpn=False,
                  use_ps=False,
                  normalize='bn'):
    inputs = Input(input_size)
    w, h = input_size[0], input_size[1]

    # subnet1
    if use_ps == True:
        w, h = w // 2, h // 2
        x = DeSubpixelConv2D(downsampling_factor=2)(inputs)
        conv00 = conv_bn(x,64,3,step='input_conv',active=False,normalize=None)
        block_00_backbone, block_00 = res_block(conv00, 64, 3, step='00', normalize=normalize, down_sample=False)
    else:
        # inputs = conv_bn(inputs,12,1,step = 'input',active = True,normalize = 'gn',group=12)  #调整通道数和Subpixel后一样
        w, h = w // 2, h // 2
        x = Lambda(lambda image: tf.compat.v1.image.resize(image, (w, h), method=0))(inputs)
        x = Conv2D(12, 1, name='input_conv', activation=None, kernel_initializer='he_normal')(x)
        conv00 = conv_bn(x, 64, 3, step='input_conv', active=False, normalize=None)
        block_00_backbone, block_00 = res_block(conv00, 64, 3, step='00', normalize=normalize, down_sample=False)

    pool_10 = MaxPooling2D(pool_size=(2, 2), name='maxpool_10')(block_00_backbone)
    if use_fpn == True:
        # fpn1,fpn2,fpn3,fpn4 = fpn(inputs,(w,h),channel_list=[32,64,128,256],normalize=normalize)  #use feature pr
        fpn1, fpn2, fpn3, fpn4 = fpnv2(x, normalize=normalize)
        pool_10 = merge_block([fpn1, pool_10], 64, step='fpn1', merge=True, normalize=normalize)  # fpn1
    block_10_backbone, block_10 = res_block(pool_10, 64, 3, step='10', normalize=normalize, down_sample=True)
    up_01 = up_block(block_10, 64, (w, h), step='01', normalize=normalize)
    # merge_01 = concatenate([up_01,block_00],axis = 3,name = 'conc_01')
    #block_00 = AG_block(block_00, up_01, 64, step='00', normalize=normalize)
    merge_01 = merge_block([up_01, block_00], 64, step='01', normalize=normalize)
    block_01 = standard_block(merge_01, 64, 3, step='01', normalize=normalize)

    # subnet2
    pool_20 = MaxPooling2D(pool_size=(2, 2), name='maxpool_20')(block_10_backbone)
    if use_fpn == True:
        pool_20 = merge_block([fpn2, pool_20], 64, step='fpn2', merge=True, normalize=normalize)  # fpn2
    block_20_backbone, block_20 = res_block(pool_20, 128, 3, step='20', normalize=normalize, down_sample=True)
    up_11 = up_block(block_20, 64, (w // 2, h // 2), step='11', normalize=normalize)
    # merge_11 = concatenate([up_11,block_10],axis = 3,name = 'conc_11')
    block_10 = AG_block(block_10, block_20, 64, step='10', normalize=normalize)
    merge_11 = merge_block([up_11, block_10], 64, step='11', normalize=normalize)
    block_11 = standard_block(merge_11, 64, 3, step='11', normalize=normalize)
    up_02 = up_block(block_11, 64, (w, h), step='02', normalize=normalize)
    # merge_02 = concatenate([up_02,block_01,block_00],axis = 3,name = 'conc_02')
    #block_01 = AG_block(block_01, up_02, 64, step='01', normalize=normalize)
    merge_02 = merge_block([up_02, block_01, block_00], 64, step='02', normalize=normalize)
    block_02 = standard_block(merge_02, 64, 3, step='02', normalize=normalize)

    # subnet3
    pool_30 = MaxPooling2D(pool_size=(2, 2), name='maxpool_30')(block_20_backbone)
    if use_fpn == True:
        pool_30 = merge_block([fpn3, pool_30], 128, step='fpn3', merge=True, normalize=normalize)  # fpn3
    block_30_backbone, block_30 = res_block(pool_30, 256, 3, step='30', normalize=normalize, down_sample=True)
    # block_30 = Dropout(0.05,name='dp_1')(block_30)
    up_21 = up_block(block_30, 128, (w // 4, h // 4), step='21', normalize=normalize)
    # merge_21 = concatenate([up_21,block_20],axis = 3,name = 'conc_21')
    block_20 = AG_block(block_20, block_30, 128, step='20', normalize=normalize)
    merge_21 = merge_block([up_21, block_20], 128, step='21', normalize=normalize)
    block_21 = standard_block(merge_21, 128, 3, step='21', normalize=normalize)
    up_12 = up_block(block_21, 64, (w // 2, h // 2), step='12', normalize=normalize)
    # merge_12 = concatenate([up_12,block_11,block_10],axis = 3,name = 'conc_12')
    block_11 = AG_block(block_11, block_21, 64, step='11', normalize=normalize)
    merge_12 = merge_block([up_12, block_11, block_10], 64, step='12', normalize=normalize)
    block_12 = standard_block(merge_12, 64, 3, step='12', normalize=normalize)
    up_03 = up_block(block_12, 64, (w, h), step='03', normalize=normalize)
    # merge_03 = concatenate([up_03,block_02,block_01,block_00],axis = 3,name = 'conc_03')
    #block_02 = AG_block(block_02, up_03, 64, step='02', normalize=normalize)
    merge_03 = merge_block([up_03, block_02, block_01, block_00], 64, step='03', normalize=normalize)
    block_03 = standard_block(merge_03, 64, 3, step='03', normalize=normalize)

    # subnet4
    pool_40 = MaxPooling2D(pool_size=(2, 2), name='maxpool_40')(block_30_backbone)
    if use_fpn == True:
        pool_40 = merge_block([fpn4, pool_40], 256, step='fpn4', merge=True, normalize=normalize)  # fpn4
    block_40_backbone, block_40 = res_block(pool_40, 512, 3, step='40', normalize=normalize, down_sample=True)
    #block_40 = Dropout(0.2, name='dp_2')(block_40)
    up_31 = up_block(block_40, 256, (w // 8, h // 8), step='31', normalize=normalize)
    # merge_31 = concatenate([up_31,block_30],axis = 3,name = 'conc_31')
    block_30 = AG_block(block_30, block_40, 256, step='30', normalize=normalize)
    merge_31 = merge_block([up_31, block_30], 256, step='31', normalize=normalize)
    block_31 = standard_block(merge_31, 256, 3, step='31', normalize=normalize)
    up_22 = up_block(block_31, 128, (w // 4, h // 4), step='22', normalize=normalize)
    # merge_22 = concatenate([up_22,block_21,block_20],axis = 3,name = 'conc_22')
    block_21 = AG_block(block_21, block_31, 128, step='21', normalize=normalize)
    merge_22 = merge_block([up_22, block_21, block_20], 128, step='22', normalize=normalize)
    block_22 = standard_block(merge_22, 128, 3, step='22', normalize=normalize)
    up_13 = up_block(block_22, 64, (w // 2, h // 2), step='13', normalize=normalize)
    # merge_13 = concatenate([up_13,block_12,block_11,block_10],axis = 3,name = 'conc_13')
    block_12 = AG_block(block_12, block_22, 64, step='12', normalize=normalize)
    merge_13 = merge_block([up_13, block_12, block_11, block_10], 64, step='13', normalize=normalize)
    block_13 = standard_block(merge_13, 64, 3, step='13', normalize=normalize)
    up_04 = up_block(block_13, 64, (w, h), step='04', normalize=normalize)
    # merge_04 = concatenate([up_04,block_03,block_02,block_01,block_00],axis = 3,name = 'conc_04')
    #block_03 = AG_block(block_03, up_04, 64, step='03', normalize=normalize)
    merge_04 = merge_block([up_04, block_03, block_02, block_01, block_00], 32, step='04', normalize=normalize)
    block_04 = standard_block(merge_04, 64, 3, step='04', normalize=normalize)

    if supervision == True:
        output1 = block_01

        output2 = block_02

        output3 = block_03

        output4 = block_04

        fusion = concatenate([output1, output2, output3, output4], axis=3, name='conc_fn')
        
        output = Conv2D(256, 1, padding='same',
                            name='conv_fn2', kernel_initializer='he_normal')(fusion)
        output = Activation('relu')(output)
        if use_ps == True:
            output = SubpixelConv2D(upsampling_factor=2)(output)

            # for 1 lesion
            # output = Conv2D(classes,(1, 1), activation = activation,name='output_conv3', kernel_initializer = 'he_normal')(output)
            # for 4 lesions
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv2',
                            kernel_initializer='he_normal')(output)
        else:
            output = Conv2D(48, 1, name='ex_conv', activation=None, kernel_initializer='he_normal')(
                output)  # 调整通道数和Subpixel后一样
            output = Lambda(
                lambda image: tf.compat.v1.image.resize(image, (input_size[0], input_size[1]), method=0))(output)
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv3',
                            kernel_initializer='he_normal')(output)

    else:
        output = block_04
        if use_ps == True:
            output = SubpixelConv2D(upsampling_factor=2)(output)
            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv', kernel_initializer='he_normal')(
                output)
        else:
            output = Conv2D(16, 1, name='ex_conv', activation=None, kernel_initializer='he_normal')(output)
            output = Lambda(lambda image: tf.compat.v1.image.resize(image, (input_size[0], input_size[1]), method=0))(
                output)

            output = Conv2D(classes, (1, 1), activation=activation, name='output_conv1',
                            kernel_initializer='he_normal')(output)

    model = Model(inputs, output)
    print(model.summary())
    return model

def original_unet_plusplus(pretrained_weights = None,
                  input_size = (512,512,1),
                  classes = 3,
                  activation ='softmax',
                  supervision = False,
                  ex_supervision = False,
                  weight_fusion = False,
                  use_fpn = False,
                  use_ps = False,
                  normalize = 'gn'):
    
    inputs = Input(input_size)
    w, h = input_size[0]//2, input_size[1]//2
    x = Lambda(lambda image: tf.compat.v1.image.resize(image, (w, h), method=0, align_corners=True))(inputs)
    
    #subnet1    
    block_00 = standard_block(x,32,3,step = '00',normalize = normalize)    
    pool_10 = MaxPooling2D(pool_size = (2,2), name = 'maxpool_10')(block_00)    
    block_10 = standard_block(pool_10,64,3,step = '10',normalize = normalize)
    up_01 = mini_up_block(block_10,32,(w,h),step = '01',normalize = normalize,linear=False)
    #merge_01 = concatenate([up_01,block_00],axis = 3,name = 'conc_01')
    merge_01 = merge_block([up_01,block_00],64,step = '01',normalize = normalize)
    block_01 = standard_block(merge_01,32,3,step = '01',normalize = normalize)
    
    #subnet2 
    pool_20 = MaxPooling2D(pool_size = (2,2), name = 'maxpool_20')(block_10)
    block_20 = standard_block(pool_20,128,3,step = '20',normalize = normalize)
    up_11 = mini_up_block(block_20,64,(w//2,h//2),step = '11',normalize = normalize,linear=False)
    #merge_11 = concatenate([up_11,block_10],axis = 3,name = 'conc_11')
    merge_11 = merge_block([up_11,block_10],64,step = '11',normalize = normalize)
    block_11 = standard_block(merge_11,64,3,step = '11',normalize = normalize)
    up_02 = mini_up_block(block_11,32,(w,h),step = '02',normalize = normalize,linear=False)
    #merge_02 = concatenate([up_02,block_01,block_00],axis = 3,name = 'conc_02')
    merge_02 = merge_block([up_02,block_01,block_00],32,step = '02',normalize = normalize)
    block_02 = standard_block(merge_02,32,3,step = '02',normalize = normalize)
    
    #subnet3 
    pool_30 = MaxPooling2D(pool_size = (2,2), name = 'maxpool_30')(block_20)
    block_30 = standard_block(pool_30,256,3,step = '30',normalize = normalize)
    #block_30 = Dropout(0.05,name='dp_1')(block_30)
    up_21 = mini_up_block(block_30,128,(w//4,h//4),step = '21',normalize = normalize,linear=False)
    #merge_21 = concatenate([up_21,block_20],axis = 3,name = 'conc_21')
    merge_21 = merge_block([up_21,block_20],128,step = '21',normalize = normalize)
    block_21 = standard_block(merge_21,128,3,step = '21',normalize = normalize)
    up_12 = mini_up_block(block_21,64,(w//2,h//2),step = '12',normalize = normalize,linear=False)
    #merge_12 = concatenate([up_12,block_11,block_10],axis = 3,name = 'conc_12')
    merge_12 = merge_block([up_12,block_11,block_10],64,step = '12',normalize = normalize)
    block_12 = standard_block(merge_12,64,3,step = '12',normalize = normalize)
    up_03 = mini_up_block(block_12,32,(w,h),step = '03',normalize = normalize,linear=False)
    #merge_03 = concatenate([up_03,block_02,block_01,block_00],axis = 3,name = 'conc_03')
    merge_03 = merge_block([up_03,block_02,block_01,block_00],32,step = '03',normalize = normalize)
    block_03 = standard_block(merge_03,32,3,step = '03',normalize = normalize)
    
    #subnet4 
    pool_40 = MaxPooling2D(pool_size = (2,2), name = 'maxpool_40')(block_30)
    block_40 = standard_block(pool_40,512,3,step = '40',normalize = normalize)    
    #block_40 = Dropout(0.2,name='dp_2')(block_40)
    up_31 = mini_up_block(block_40,256,(w//8,h//8),step = '31',normalize = normalize,linear=False)
    #merge_31 = concatenate([up_31,block_30],axis = 3,name = 'conc_31')
    merge_31 = merge_block([up_31,block_30],256,step = '31',normalize = normalize)
    block_31 = standard_block(merge_31,256,3,step = '31',normalize = normalize)
    up_22 = mini_up_block(block_31,128,(w//4,h//4),step = '22',normalize = normalize,linear=False)
    #merge_22 = concatenate([up_22,block_21,block_20],axis = 3,name = 'conc_22')
    merge_22 = merge_block([up_22,block_21,block_20],128,step = '22',normalize = normalize)
    block_22 = standard_block(merge_22,128,3,step = '22',normalize = normalize)
    up_13 = mini_up_block(block_22,64,(w//2,h//2),step = '13',normalize = normalize,linear=False)
    #merge_13 = concatenate([up_13,block_12,block_11,block_10],axis = 3,name = 'conc_13')
    merge_13 = merge_block([up_13,block_12,block_11,block_10],64,step = '13',normalize = normalize)
    block_13 = standard_block(merge_13,64,3,step = '13',normalize = normalize)
    up_04 = mini_up_block(block_13,32,(w,h),step = '04',normalize = normalize,linear=False)
    #merge_04 = concatenate([up_04,block_03,block_02,block_01,block_00],axis = 3,name = 'conc_04')
    merge_04 = merge_block([up_04,block_03,block_02,block_01,block_00],32,step = '04',normalize = normalize)
    block_04 = standard_block(merge_04,32,3,step = '04',normalize = normalize)
    
    if supervision == True:
        output1 = block_01
        
        output2 = block_02

        output3 = block_03
       
        output4 = block_04
        
        fusion = concatenate([output1,output2,output3,output4],axis = 3,name = 'conc_fn')
        #output = conv_bn(fusion,32,1,step = 'fn',normalize = normalize)
        output = Conv2D(32,(1, 1), activation = 'relu',name='fn_conv', kernel_initializer = 'he_normal')(fusion)
        output = Lambda(lambda image: tf.compat.v1.image.resize(image, (w*2, h*2), method=0, align_corners=True))(output)
        output = Conv2D(classes,(1, 1), activation = activation,name='output_conv', kernel_initializer = 'he_normal')(output)
        

    else:
        output = block_04
        #output = Conv2D(32,(1, 1), activation = 'relu',name='fn_conv', kernel_initializer = 'he_normal')(output)        
        output = Lambda(lambda image: tf.compat.v1.image.resize(image, (w*2, h*2), method=0, align_corners=True))(output)  
        output = Conv2D(classes,(1, 1), activation = activation,name='output_conv', kernel_initializer = 'he_normal')(output)
    model = Model(inputs, output)   
    print(model.summary())
    return model

def unet(pretrained_weights=None,
                  input_size=(512, 512, 1),
                  classes=3,
                  activation='softmax',
                  use_fpn=False,
                  use_ps=False,
                  normalize='bn'):
    inputs = Input(input_size)
    w, h = input_size[0], input_size[1]

    # subnet1
    if use_ps == True:
        w, h = w // 2, h // 2
        x = DeSubpixelConv2D(downsampling_factor=2)(inputs)
        block_00 = standard_block(x, 32, 3, step='00x', normalize=normalize)
    else:
        w, h = w // 2, h // 2
        x = Lambda(lambda image: tf.compat.v1.image.resize(image, (w, h), method=0, align_corners=True))(inputs)
        block_00 = standard_block(x, 32, 3, step='00x', normalize=normalize)

    pool_10 = MaxPooling2D(pool_size=(2, 2), name='maxpool_10')(block_00)
    block_10 = standard_block(pool_10, 64, 3, step='10', normalize=normalize)


    # subnet2
    pool_20 = MaxPooling2D(pool_size=(2, 2), name='maxpool_20')(block_10)
    block_20 = standard_block(pool_20, 128, 3, step='20', normalize=normalize)


    # subnet3
    pool_30 = MaxPooling2D(pool_size=(2, 2), name='maxpool_30')(block_20)
    block_30 = standard_block(pool_30, 256, 3, step='30', normalize=normalize)
    # block_30 = Dropout(0.05,name='dp_1')(block_30)


    # subnet4
    pool_40 = MaxPooling2D(pool_size=(2, 2), name='maxpool_40')(block_30)
    block_40 = standard_block(pool_40, 512, 3, step='40', normalize=normalize)
    #block_40 = Dropout(0.2, name='dp_2')(block_40)
    up_31 = mini_up_block(block_40, 256, (w // 8, h // 8), step='31', normalize=normalize)
    # merge_31 = concatenate([up_31,block_30],axis = 3,name = 'conc_31')
    merge_31 = merge_block([up_31, block_30], 256, step='31', normalize=normalize)
    block_31 = standard_block(merge_31, 256, 3, step='31', normalize=normalize)
    up_22 = mini_up_block(block_31, 128, (w // 4, h // 4), step='22', normalize=normalize)
    # merge_22 = concatenate([up_22,block_21,block_20],axis = 3,name = 'conc_22')
    merge_22 = merge_block([up_22, block_20], 128, step='22', normalize=normalize)
    block_22 = standard_block(merge_22, 128, 3, step='22', normalize=normalize)
    up_13 = mini_up_block(block_22, 64, (w // 2, h // 2), step='13', normalize=normalize)
    # merge_13 = concatenate([up_13,block_12,block_11,block_10],axis = 3,name = 'conc_13')
    merge_13 = merge_block([up_13, block_10], 64, step='13', normalize=normalize)
    block_13 = standard_block(merge_13, 64, 3, step='13', normalize=normalize)
    up_04 = mini_up_block(block_13, 32, (w, h), step='04', normalize=normalize)
    # merge_04 = concatenate([up_04,block_03,block_02,block_01,block_00],axis = 3,name = 'conc_04')
    merge_04 = merge_block([up_04, block_00], 32, step='04', normalize=normalize)
    block_04 = standard_block(merge_04,32, 3, step='04', normalize=normalize)





    if use_ps == True:
        block_04 = conv_bn(block_04,128,1,step='f1',active=True,normalize=None)
        output = SubpixelConv2D(upsampling_factor=2)(block_04)
        output = Conv2D(classes, (1, 1), activation=activation, name='output_conv2',
                        kernel_initializer='he_normal')(output)
    else:
        block_04 = Lambda(lambda image: tf.compat.v1.image.resize(image, (w*2, h*2), method=0, align_corners=True))(block_04) 
        output = Conv2D(classes, (1, 1), activation=activation, name='output_conv',
                        kernel_initializer='he_normal')(block_04)



    model = Model(inputs, output)
    print(model.summary())
    return model