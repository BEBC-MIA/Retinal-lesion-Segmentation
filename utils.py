#!/usr/bin/env python
# coding: utf-8

# In[ ]:
'''
from matplotlib import pyplot as plt
import math
from keras.callbacks import *
from keras import backend as K
from tensorflow.python.keras.optimizers import adam
'''
from matplotlib import pyplot as plt
import math
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras import backend as K
import tensorflow as tf
#from tensorflow.python.keras.optimizers import adam


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if math.isnan(loss) or loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find_v1(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):
        num_batches = epochs * x_train.shape[0] / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        # Save weights into a file
        self.model.save_weights('tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[callback])

        # Restore the weights to the state before model fitting
        self.model.load_weights('tmp.h5')

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)
        
    def find(self, aug_gen, start_lr, end_lr, batch_size=600, epochs=1, num_train = 10000):
        num_batches = int(epochs * num_train / batch_size)
        steps_per_epoch = int(num_train / batch_size) 
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        # Save weights into a file
        self.model.save_weights('H:/python project/DR_seg_data/tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit_generator(aug_gen,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[callback])

        # Restore the weights to the state before model fitting
        self.model.load_weights('H:/python project/DR_seg_data/tmp.h5')

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=20, n_skip_end=5):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        plt.ylim(y_lim)

        
def scheduler(epoch,epochlist=[8,16,40,56,72,76],lrlist=[0.5,0.25,0.125,1/64,1/256,1/512]):
    # 每隔10个epoch，学习率减小为原来的1/2
    for i,j in enumerate(epochlist):
        if j == epoch:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr*lrlist[i])
            print("lr changed to {}".format(lr*lrlist[i]))
    return K.get_value(model.optimizer.lr)
        
class poly_decay(Callback): 
    def __init__(self,
                 max_epochs=80,
                 step_each_epoch=608,
                 power = 0.9, 
                 ):
        self.max_epochs = max_epochs
        self.step_each_epoch = step_each_epoch
        self.power = power
        self.batch = 0
    def on_train_begin(self, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)
    def on_epoch_begin(self, epoch, logs={}):
        print(K.get_value(self.model.optimizer.lr))   
    def on_batch_end(self, batch, logs={}):
        self.batch += 1
        #ite = K.get_value(model.optimizer.iterations)
        lr = self.lr*((1 - (self.batch / float(self.max_epochs*self.step_each_epoch)))**self.power) 
        if lr<=0:
            lr=0
        K.set_value(self.model.optimizer.lr, lr)
class get_lr(Callback): 
    def __init__(self,
                 epoch_list=[10,30,50,65,75],
                 
                 ):
        self.epoch_list = epoch_list        
        self.ep = 0
    def on_train_begin(self, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)
        print('base_lr=',self.lr)
    def on_epoch_begin(self, epoch, logs={}):
        lr = K.get_value(self.model.optimizer.lr)
        print(self.ep,'：',lr)
        self.ep += 1           
class step_decay(Callback): 
    def __init__(self,
                 epoch_list=[10,30,50,65,75],
                 decay_factor=0.1,
                 ):
        self.epoch_list = epoch_list
        self.decay_factor = decay_factor
        self.ep = 0
    def on_train_begin(self, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)
    def on_epoch_begin(self, epoch, logs={}):
        lr = K.get_value(self.model.optimizer.lr)
        print(self.ep,'：',lr)
        for i ,element in enumerate(self.epoch_list):
            if self.ep == element:
                K.set_value(self.model.optimizer.lr, lr*self.decay_factor)               
        self.ep += 1   

'''   
def poly_decay(epoch):    
    # initialize the maximum number of epochs, base learning rate,    # and power of the polynomial    
    maxEpochs = 80   
    step_each_epoch=608#根据自己的情况设置    
    baseLR = 0.0001    
    power = 0.9    
    ite = K.get_value(model.optimizer.iterations)       # compute the new learning rate based on polynomial decay    
    alpha = baseLR*((1 - (ite / float(maxEpochs*step_each_epoch)))**power)        # return the new learning rate    
    return alpha
'''
class SGDRScheduler(Callback):
    '''Schedule learning rates with restarts
     A simple restart technique for stochastic gradient descent.
    The learning rate decays after each batch and peridically resets to its
    initial value. Optionally, the learning rate is additionally reduced by a
    fixed factor at a predifined set of epochs.
     # Arguments
        epochsize: Number of samples per epoch during training.
        batchsize: Number of samples per batch during training.
        start_epoch: First epoch where decay is applied.
        epochs_to_restart: Initial number of epochs before restarts.
        mult_factor: Increase of epochs_to_restart after each restart.
        lr_fac: Decrease of learning rate at epochs given in
                lr_reduction_epochs.
        lr_reduction_epochs: Fixed list of epochs at which to reduce
                             learning rate.
     # References
        - [SGDR: Stochastic Gradient Descent with Restarts](http://arxiv.org/abs/1608.03983)
    '''
    def __init__(self,
                 epochsize,
                 batchsize,
                 epochs_to_restart=2,
                 mult_factor=2,
                 lr_fac=0.1,
                 Ir_min=5e-7,
                 lr_reduction_epochs=(60, 120, 160)):
        super(SGDRScheduler, self).__init__()
        self.epoch = -1
        self.batch_since_restart = 0
        self.next_restart = epochs_to_restart
        self.epochsize = epochsize
        self.batchsize = batchsize
        self.epochs_to_restart = epochs_to_restart
        self.mult_factor = mult_factor
        self.batches_per_epoch = self.epochsize / self.batchsize
        self.lr_fac = lr_fac
        self.lr_reduction_epochs = lr_reduction_epochs
        self.lr_log = []
        self.Ir_min = Ir_min

    def on_train_begin(self, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1
        #print(self.lr)

    def on_batch_end(self, batch, logs={}):
        
        fraction_to_restart = self.batch_since_restart / \
            (self.batches_per_epoch * self.epochs_to_restart)
        lr = self.Ir_min + 0.5 * (self.lr - self.Ir_min) * (1 + np.cos(fraction_to_restart * np.pi))
        K.set_value(self.model.optimizer.lr, lr)

        self.batch_since_restart += 1
        self.lr_log.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.epochs_to_restart *= self.mult_factor
            self.next_restart += self.epochs_to_restart

        if (self.epoch + 1) in self.lr_reduction_epochs:
            self.lr *= self.lr_fac
        print( K.get_value(self.model.optimizer.lr))
            
'''
def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    #print(aa = tf.cast(y_true *y_pred, tf.bool))

    true_positives = K.sum(K.clip(y_true *y_pred, 0, 1), axis=( 1, 2))
    possible_positives = K.sum(y_true, axis=( 1, 2))
    recall = true_positives / (possible_positives + K.epsilon())
    return K.mean(recall,axis=0) 
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.clip(y_true * K.round(y_pred), 0, 1), axis=( 1, 2))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=( 1, 2))
    precision = true_positives / (predicted_positives + K.epsilon())
    return K.mean(precision,axis=0)      
'''           
def precision(y_true, y_pred):
    eps=1e-6
    p0 = y_pred  # proba that voxels are class i
    p1 = 1 - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = 1 - y_true
    # 求得每个sample的每个类的dice
    tp=K.sum(p0 * g0, axis=( 1, 2))
    fp=K.sum(p0 * g1, axis=( 1, 2))
    P=tp/(tp+fp+eps)
    P=K.mean(P,axis=0)
    return P

     
def recall(y_true, y_pred):
    eps=1e-6
    p0 = y_pred  # proba that voxels are class i
    p1 = 1 - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = 1 - y_true
    # 求得每个sample的每个类的dice
    tp=K.sum(p0 * g0, axis=( 1, 2))
    fn=K.sum(p1 * g0, axis=( 1, 2))   
    R=tp/(tp+fn+eps)
    R=K.mean(R,axis=0)
    return R
    
def P_HE(y_true, y_pred):
    phe=precision(y_true, y_pred)
    return phe[2]
def R_HE(y_true, y_pred):
    rhe=recall(y_true, y_pred)
    return rhe[2]
def P_EX(y_true, y_pred):
    phe=precision(y_true, y_pred)
    return phe[1]
def R_EX(y_true, y_pred):
    rhe=recall(y_true, y_pred)
    return rhe[1]  
def P_MA(y_true, y_pred):
    phe=precision(y_true, y_pred)
    return phe[3]
def R_MA(y_true, y_pred):
    rhe=recall(y_true, y_pred)
    return rhe[3]
def P_SE(y_true, y_pred):
    phe=precision(y_true, y_pred)
    return phe[4]
def R_SE(y_true, y_pred):
    rhe=recall(y_true, y_pred)
    return rhe[4]

def f1_EX(y_true, y_pred):
    eps=1e-6
    f1 = 2*recall(y_true, y_pred)*precision(y_true, y_pred)/(recall(y_true, y_pred)+precision(y_true, y_pred)+eps)
    return f1[1]
def f1_HE(y_true, y_pred):
    eps=1e-6
    f1 = 2*recall(y_true, y_pred)*precision(y_true, y_pred)/(recall(y_true, y_pred)+precision(y_true, y_pred)+eps)
    return f1[2]
def f1_MA(y_true, y_pred):
    eps=1e-6
    f1 = 2*recall(y_true, y_pred)*precision(y_true, y_pred)/(recall(y_true, y_pred)+precision(y_true, y_pred)+eps)
    return f1[3]
    #return f1[2]
def f1_SE(y_true, y_pred):
    eps=1e-6
    f1 = 2*recall(y_true, y_pred)*precision(y_true, y_pred)/(recall(y_true, y_pred)+precision(y_true, y_pred)+eps)
    return f1[4]

#for multi-classes
def mf1(y_true, y_pred):
    eps=1e-6
    f1 = 2*recall(y_true, y_pred)*precision(y_true, y_pred)/(recall(y_true, y_pred)+precision(y_true, y_pred)+eps)
    return (K.sum(f1)-f1[0])/(4)
    #return (K.sum(f1)-f1[0])/(2)
'''
def mf1(y_true, y_pred):
    eps=1e-6
    f1 = 2*recall(y_true, y_pred)*precision(y_true, y_pred)/(recall(y_true, y_pred)+precision(y_true, y_pred)+eps)
    return f1
'''    
def mIoU(y_true, y_pred):
    eps=1e-6
    # if np.max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2]) - intersection
    ious=K.mean((intersection + eps) / (union + eps),axis=0)
    return ious
def IoU(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return K.mean(ious)
def IoUBK(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return ious[0]
'''
def IoUEX(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return ious[0]
'''
#for multi-classes
def IoUEX(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return ious[1]
    
def IoUHE(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return ious[2]
def IoUMA(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return ious[3]
    #return ious[2]
def IoUSE(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return ious[4]
def realIoU(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return (K.sum(ious)- ious[0])/4
def IoUW(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return ious[1]
def IoUR(y_true, y_pred):
    ious=mIoU(y_true, y_pred)
    return ious[2]


