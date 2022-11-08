#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python.keras import backend as K

def generalized_dice_coeff():
    def generalized_dice(y_true, y_pred):
        Ncl = y_pred.shape[-1]

        w = K.sum(y_true, axis=(0,1,2))
        w = 1/(w**2+0.000001)
        # Compute gen dice coef:
        numerator = y_true*y_pred
        numerator = w*K.sum(numerator,axis=(0,1,2))
        numerator = K.sum(numerator)
        denominator = y_true+y_pred
        denominator = w*K.sum(denominator,axis=(0,1,2))
        denominator = K.sum(denominator)
        gen_dice_coef = 2*numerator/denominator
        return gen_dice_coef
    return generalized_dice

def generalized_dice_loss_fun():
    def generalized_dice_loss(y_true, y_pred):
        return 1 - generalized_dice_coeff()(y_true=y_true, y_pred=y_pred)
    return generalized_dice_loss

'''
def log_tversky_coef_fun(alpha,beta,gamma=1):
    def log_tversky_coef(y_true, y_pred):
        epsilon = 1
        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true
        # 求得每个sample的每个类的dice
        fn=K.sum(p1 * g0, axis=( 1, 2))
        fp=K.sum(p0 * g1, axis=( 1, 2))
        # 求得每个sample的每个类的dice
        num = K.sum(p0 * g0, axis=( 1, 2))+epsilon
        den = num + alpha * fn + beta * fp+epsilon
        T =  K.pow( -K.log(num / den),gamma)  #[class_num]
        T=K.mean(T,axis=0) #[class_num]
        # 求得每个类的dice
        #dices=K.mean(T,axis=0) #[class_num]
        return K.mean(T)
    return log_tversky_coef
'''
def log_tversky_coef_fun(alpha,beta,gamma=1):
    def log_tversky_coef(y_true, y_pred):
        epsilon=1e-6
        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true
        # 求得每个sample的每个类的dice
        fn=K.sum(p1 * g0, axis=( 1, 2))
        fp=K.sum(p0 * g1, axis=( 1, 2))
        tp=K.sum(p0 * g0, axis=( 1, 2))
        # 求得每个sample的每个类的dice
        num = tp+epsilon
        den = tp + alpha * fn + beta * fp+epsilon
        
        T =  K.pow( -K.log(num / den),gamma)  #[class_num]

        # 求得每个类的dice
        T=K.mean(T,axis=-1) #[class_num]
        return K.mean(T)
    return log_tversky_coef
def adapt_tversky_coef_fun():
    def tversky_coef(y_true, y_pred):
        
        epsilon =1e-6
        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true
        # 求得每个sample的每个类的dice
        fn=K.sum(p1 * g0, axis=( 1, 2))
        fp=K.sum(p0 * g1, axis=( 1, 2))
        tp=K.sum(p0 * g0, axis=( 1, 2))
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=( 1, 2))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=( 1, 2))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=( 1, 2))
        R = true_positives / (possible_positives + epsilon)
        P = true_positives / (predicted_positives + epsilon)
        '''
        R = (tp) / (tp + fn + epsilon)
        #R = K.mean(R, axis=0)
        P = (tp) / (tp + fp + epsilon)
        #P = K.mean(P, axis=0)
        alpha = R / (R + P + epsilon)
        alpha = 0.5 + 0.5*tf.math.sin(x = 3.14*(alpha - 0.5)) 
        tf.clip_by_value(alpha, 0.1, 0.9)  #限定alpha在[0.1,0.9]
        beta = 1-alpha
        # 求得每个sample的每个类的dice
        #num = K.sum(p0 * g0, axis=( 1, 2))
        den = tp + alpha * fp + beta * fn + epsilon   #fn、fp可能写反了
        T = tp / den  #[batchsize,class_num]
        T = K.mean(T, axis=0)
        # 求得每个类的dice
        
        return K.mean(T,axis=-1) 
    return tversky_coef

    
def tversky_coef_fun(alpha,beta):
    def tversky_coef(y_true, y_pred):
        
        epsilon =1e-6
        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true
        # 求得每个sample的每个类的dice
        fn=K.sum(p1 * g0, axis=( 1, 2))
        fp=K.sum(p0 * g1, axis=( 1, 2))
        tp=K.sum(p0 * g0, axis=( 1, 2))
        '''
        fn=K.mean(fn, axis=0)
        fp=K.mean(fp, axis=0)
        tp=K.mean(tp, axis=0) #[class_num]
        '''
        # 求得每个sample的每个类的dice
        #num = K.sum(p0 * g0, axis=( 1, 2))
        den = tp + alpha * fp + beta * fn+epsilon   #fn、fp可能写反了
        T = tp / den  #[batchsize,class_num]
        T = K.mean(T, axis=0)
        # 求得每个类的dice
        
        return K.mean(T,axis=-1) 
    return tversky_coef
def categorical_crossentropy_v2(w=[0.1,0.9,0.9,0.9,0.9]):
 
    #weights = K.variable(w1,w2)
    #weights = (K.sum(y_true,axis=(0,1,2))+epsilon)
    epsilon = K.epsilon()
    
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)   #[bs,x,y,classes]
        p_loss = -K.sum(y_true * K.log(y_pred)*w , axis=(-1))
        #n_loss = -K.sum((1-y_true) * K.log(1-y_pred) , axis=(-1))
        #loss = K.mean((w1*p_loss+w2*n_loss),axis=0)
        loss = K.mean(p_loss)
        return loss
    return loss
def focal_loss(gamma=2.0, a=1):  
    epsilon =1e-6
    alpha = tf.constant(a, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        ce_loss = y_true * K.log(y_pred)
        focal_loss = K.pow(1 - y_pred, gamma) * ce_loss * alpha
        focal_loss = -K.sum(focal_loss, axis=-1)
        return K.mean(focal_loss,axis=0)
    return loss

def tversky_coef_loss_fun(alpha,beta):
    def tversky_coef_loss(y_true,y_pred):
        return 1-tversky_coef_fun(alpha=alpha,beta=beta)(y_true=y_true,y_pred=y_pred)
    return tversky_coef_loss
def log_tversky_coef_loss_fun(alpha,beta,gamma=1):
    def log_tversky_coef_loss(y_true,y_pred):
        return log_tversky_coef_fun(alpha=alpha,beta=beta,gamma=gamma)(y_true=y_true,y_pred=y_pred)
    return log_tversky_coef_loss
def log_tversky_CE_loss_fun(alpha,beta,gamma=1,factor=0.3):
    def log_tversky_coef_loss(y_true,y_pred):
        return log_tversky_coef_fun(alpha=alpha,beta=beta,gamma=gamma)(y_true=y_true,y_pred=y_pred)*factor+categorical_crossentropy_v2(w=w)(y_true=y_true,y_pred=y_pred)
    return log_tversky_coef_loss
def tversky_CE_loss_fun(alpha,beta,factor):
    def tversky_coef_loss(y_true,y_pred):
        return 1-tversky_coef_fun(alpha=alpha,beta=beta)(y_true=y_true,y_pred=y_pred)+factor*K.categorical_crossentropy(y_true,y_pred)
    return tversky_coef_loss
def tversky_focal_loss_fun(alpha,beta,factor,gamma, a):
    def tversky_coef_loss(y_true,y_pred):
        return 1-tversky_coef_fun(alpha=alpha,beta=beta)(y_true=y_true,y_pred=y_pred)*factor+focal_loss(gamma=gamma, a=a)(y_true=y_true,y_pred=y_pred)
    return tversky_coef_loss
def tversky_CEv2_loss_fun(alpha,beta,w,factor):
    def tversky_coef_loss(y_true,y_pred):
        return (1-tversky_coef_fun(alpha=alpha,beta=beta)(y_true=y_true,y_pred=y_pred))*factor+categorical_crossentropy_v2(w=w)(y_true=y_true,y_pred=y_pred)
    return tversky_coef_loss
def adapt_tversky_CEv2_loss_fun(w,factor):
    def adapt_tversky_coef_loss(y_true,y_pred):
        return (1-adapt_tversky_coef_fun()(y_true=y_true,y_pred=y_pred))*factor+categorical_crossentropy_v2(w=w)(y_true=y_true,y_pred=y_pred)
    return adapt_tversky_coef_loss

def tversky_BCE_loss_fun(alpha,beta):
    def tversky_coef_loss(y_true,y_pred):
        return (1-tversky_coef_fun(alpha=alpha,beta=beta)(y_true=y_true,y_pred=y_pred))*0.4+K.binary_crossentropy(y_true,y_pred)
    return tversky_coef_loss
def GD_CEv2_loss_fun(w,factor):
    def GD_coef_loss(y_true,y_pred):
        return (1-generalized_dice_coeff()(y_true=y_true,y_pred=y_pred))*factor+categorical_crossentropy_v2(w=w)(y_true=y_true,y_pred=y_pred)
    return GD_coef_loss




'''

def weighted_categorical_crossentropy(weights):
 
    weights = K.variable(weights)
    #weights = (K.sum(y_true,axis=(0,1,2))+epsilon)
    epsilon = K.epsilon()
    
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)   #[bs,x,y,classes]
    
        loss = -K.sum(y_true * K.log(y_pred) , axis=(1,2))
        loss = K.mean(K.pow(loss, gamma),axis=0)*weights
        return K.mean(loss,axis=-1)
    return loss
'''




def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss2_fixed


def mixloss(gamma=2.0, alpha=1.0,a=0.5,b=0.5,r=0.5):
    epsilon =1e-6
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        ce_loss = y_true * K.log(y_pred)
        focal_loss = K.pow(1 - y_pred, gamma) * ce_loss * alpha
        focal_loss = -K.sum(focal_loss, axis=-1)
        
        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true
        # 求得每个sample的每个类的dice
        num = K.sum(p0 * g0, axis=( 1, 2))
        den = num + a * K.sum(p0 * g1,axis= ( 1, 2)) + b * K.sum(p1 * g0, axis=( 1, 2))
        T = num / den  #[batch_size,class_num]

        # 求得每个类的dice
        dices=K.mean(T,axis=0) #[class_num]
        dice_loss= 1-K.mean(dices)
        return focal_loss*r+dice_loss
    return loss
    
    
def focal_tversky_fun(gamma,alpha,beta):    
    epsilon =1e-6
    def focal_tversky_coef(y_true, y_pred):
        
        
        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true
        fn=K.sum(p1 * g0, axis=( 1, 2))
        fp=K.sum(p0 * g1, axis=( 1, 2))
        # 求得每个sample的每个类的dice
        num = K.sum(p0 * g0, axis=( 1, 2))
        den = num + alpha * fn + beta * fp
        T = num / den  #[batch_size,class_num]
        # 求得每个类的dice
        dices=K.mean(T,axis=0) #[class_num] 
        dices=K.mean(dices)
        fc_tversky_loss=K.pow((1 - dices), gamma)
        
        return fc_tversky_loss
    return focal_tversky_coef

def focal_tversky_CE_loss_fun(gamma,alpha,beta,factor):
    def focal_tversky_loss(y_true,y_pred):
        return focal_tversky_fun(gamma=gamma,alpha=alpha,beta=beta)(y_true=y_true,y_pred=y_pred)+factor*K.categorical_crossentropy(y_true,y_pred)
    return focal_tversky_loss


def dice_coef_fun(smooth=1):
    def dice_coef(y_true, y_pred):
        # 求得每个sample的每个类的dice
        intersection = K.sum(y_true * y_pred, axis=(1, 2))
        union = K.sum(y_true, axis=(1, 2)) + K.sum(y_pred, axis=(1, 2))
        sample_dices = (2. * intersection + smooth) / (union + smooth)  # 一维数组 为各个类别的dice
        # 求得每个类的dice
        dices = K.mean(sample_dices, axis=0)
        return K.mean(dices)  # 所有类别dice求平均的dice

    return dice_coef


def dice_coef_loss_fun(smooth=1):
    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef_fun(smooth=smooth)(y_true=y_true, y_pred=y_pred)

    return dice_coef_loss