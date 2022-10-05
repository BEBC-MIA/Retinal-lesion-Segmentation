#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tensorflow.keras import backend as K
from scipy.integrate import simps, trapz

def get_y_true(indir,classes,target_size=(420,420),task=None):     #get y_true from groundthruth   
    class1_indir=os.path.join(indir,'EX')
    class2_indir=os.path.join(indir,'HE')
    class3_indir=os.path.join(indir,'MA')
    class4_indir=os.path.join(indir,'SE')
    filelist=os.listdir(class1_indir)    
    classes_np_list=[]
    h,w=target_size
    if classes==1:
        classes_np=np.empty((classes,)+(len(filelist),)+target_size,np.float32)
    else: 
        classes_np=np.empty((classes-1,)+(len(filelist),)+target_size,np.float32)
    for j,filename in enumerate(filelist):
        if task=='ex_ma':
            label1=cv2.imread(os.path.join(class1_indir,filename),0)
            label1=cv2.resize(label1,target_size,cv2.INTER_AREA)
            ret,label1=cv2.threshold(label1,127,255,cv2.THRESH_BINARY)
            label2=cv2.imread(os.path.join(class3_indir,filename),0)
            label2=cv2.resize(label2,target_size,cv2.INTER_AREA)
            ret,label2=cv2.threshold(label2,127,255,cv2.THRESH_BINARY)
        else:   
            label1=cv2.imread(os.path.join(class1_indir,filename),0)
            label1=cv2.resize(label1,target_size,cv2.INTER_AREA)
            ret,label1=cv2.threshold(label1,127,255,cv2.THRESH_BINARY)
            label2=cv2.imread(os.path.join(class2_indir,filename),0)
            label2=cv2.resize(label2,target_size,cv2.INTER_AREA)
            ret,label2=cv2.threshold(label2,127,255,cv2.THRESH_BINARY)
            label3=cv2.imread(os.path.join(class3_indir,filename),0)
            label3=cv2.resize(label3,target_size,cv2.INTER_AREA)
            ret,label3=cv2.threshold(label3,127,255,cv2.THRESH_BINARY)
            label4=cv2.imread(os.path.join(class4_indir,filename),0)
            label4=cv2.resize(label4,target_size,cv2.INTER_AREA)
            ret,label4=cv2.threshold(label4,127,255,cv2.THRESH_BINARY)
        #cv2.imshow('l',label1)
        #cv2.waitKey(0)
        label_1=np.zeros(target_size)
        label_2=np.zeros(target_size)
        label_3=np.zeros(target_size)
        label_4=np.zeros(target_size)
        
        if classes==3 and task==None:
            label_1[np.where(label1==255)]=1
            label_1[np.where(label4==255)]=1
            label_2[np.where(label2==255)]=1
            label_2[np.where(label3==255)]=1
    
            classes_np[(0,j)]=label_1
            classes_np[(1,j)]=label_2
            
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
            classes_np2=classes_np[1].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np2)

        elif classes==5 and task==None:
            label_1[np.where(label1==255)]=1
            label_2[np.where(label2==255)]=1
            label_3[np.where(label3==255)]=1
            label_4[np.where(label4==255)]=1
               
            classes_np[(0,j)]=label_1
            classes_np[(1,j)]=label_2
            classes_np[(2,j)]=label_3
            classes_np[(3,j)]=label_4
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
            classes_np2=classes_np[1].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np2)
            classes_np3=classes_np[2].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np3)
            classes_np4=classes_np[3].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np4)           
        elif classes==3 and task=='ex_se':
            label_1[np.where(label1==255)]=1           
            label_2[np.where(label4==255)]=1  
            classes_np[(0,j)]=label_1
            classes_np[(1,j)]=label_2
            
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
            classes_np2=classes_np[1].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np2)
        elif classes==3 and task=='ex_ma':
            label_1[np.where(label1==255)]=1           
            label_2[np.where(label2==255)]=1  
            classes_np[(0,j)]=label_1
            classes_np[(1,j)]=label_2
            
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
            classes_np2=classes_np[1].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np2)
        elif classes==3 and task=='ma_he':
            label_1[np.where(label2==255)]=1           
            label_2[np.where(label3==255)]=1  
            classes_np[(0,j)]=label_1
            classes_np[(1,j)]=label_2
            
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
            classes_np2=classes_np[1].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np2)
        elif classes==1 and task=='ma': 
            label_1[np.where(label3==255)]=1
            classes_np[(0,j)]=label_1
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
        elif classes==1 and task=='ex': 
            label_1[np.where(label1==255)]=1
            classes_np[(0,j)]=label_1
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
        elif classes==1 and task=='he': 
            label_1[np.where(label2==255)]=1
            classes_np[(0,j)]=label_1
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
        elif classes==1 and task=='se': 
            label_1[np.where(label4==255)]=1
            classes_np[(0,j)]=label_1
            classes_np1=classes_np[0].reshape(len(filelist)*h*w)
            classes_np_list.append(classes_np1)
    print('classes_np',classes_np.shape)
    print('classes_np1',classes_np1.shape)
    return classes_np_list

def get_y_pred(results,classes,target_size=(420,420)):    #get y_pred from predict-results      
    y_pred = []
    h,w = target_size 
    results_flatten = results.reshape((len(results)*w*h,classes))
    if classes == 1:                     
        y_pred.append(results_flatten[:,0])        
    else:       
        for i in range(classes):                
            y_pred.append(results_flatten[:,i])         #y_pred[0] is background class
            
    return y_pred     
    
def accuracy(y_true, y_pred):
    #if not K.is_tensor(y_pred):
        #y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.cast(K.equal(y_true, y_pred), K.floatx())

def binary_accuracy(y_true, y_pred, threshold=0.5):
    if threshold != 0.5:
        threshold = K.cast(threshold, y_pred.dtype)
        y_pred = K.cast(y_pred > threshold, y_pred.dtype)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    
def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    #print(aa = tf.cast(y_true *y_pred, tf.bool))
    #y_true.numpy()
    y_pred=y_pred.numpy()
    shape = y_pred.shape
    index = np.argmax(y_pred, axis=3)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                y_pred[i,j,k,index[i,j,k]] = 1
    true_positives = np.sum(y_true * y_pred, axis=(1, 2))
    possible_positives = np.sum(y_true, axis=(1, 2))
    recall = true_positives / (possible_positives + 1e-6)
    return np.mean(recall,axis=0) 
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    #y_true.numpy()
    y_pred=y_pred.numpy()
    shape = y_pred.shape
    index = np.argmax(y_pred, axis=3)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                y_pred[i,j,k,index[i,j,k]] = 1
    true_positives = np.sum(y_true * y_pred, axis=(1, 2))
    predicted_positives = np.sum(y_pred, axis=(1, 2))
    precision = true_positives / (predicted_positives + 1e-6)
    return np.mean(precision,axis=0)      
'''
def AUPR(pos_prob,y_true,classname,threshold_num = 10):   
    pos_prob = np.array(pos_prob)
    y_true = np.array(y_true)
    pos = y_true[y_true==1]
    #threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    pos_prob_sort=np.sort(pos_prob)[::-1]
    threshold=np.linspace(1,0,threshold_num)
    threshold_list=threshold.tolist()
    #y = y_true[pos_prob.argsort()[::-1]]
    recall = [] ; precision = []
    tp = 0 ; fp = 0
    auc = 0  
    print(threshold_list)
    for j,thr in enumerate(threshold_list):
        #print(pos_prob[np.where(pos_prob>=thr)])
        tp = 0 ; fp = 0
        #print(j,thr )
        if thr>=pos_prob_sort[0]:
            recall.append(0)
            precision.append(1)
        else:    
            for i in range(len(pos_prob_sort[np.where(pos_prob_sort>thr)])):
                if y[i]==1:
                    tp +=1
                else:
                    fp +=1
            recall.append(tp/(len(pos)))
            precision.append(tp/(tp+fp))
            auc += (recall[j]-recall[j-1])*(precision[j]+precision[j-1])*0.5
    plt.figure(figsize=(10,10))
    plt.plot(recall,precision,label=classname+"(AUPR: {:.4f})".format(auc),linewidth=2)
    plt.plot([0,1],[1,0], 'r--')
    plt.xlabel("Recall",fontsize=16)
    plt.ylabel("Precision",fontsize=16)
    plt.title("Precision Recall Curve",fontsize=17)
    plt.legend(fontsize=16)      
    return precision,recall,auc
'''
def AUPR(pos_prob,y_true,classname,threshold_num = 33):   
    pos_prob = np.array(pos_prob)
    y_true = np.array(y_true)
    pos = y_true[y_true==1]
    #threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    pos_prob_sort=np.sort(pos_prob)[::-1]
    threshold=np.linspace(1,0,threshold_num)
    threshold_list=threshold.tolist()
    #y = y_true[pos_prob.argsort()[::-1]]
    recall = [] ; precision = []
    tp = 0 ; fp = 0
    auc = 0  
    print(threshold_list)
    for j,thr in enumerate(threshold_list):
        #print(pos_prob[np.where(pos_prob>=thr)])
        tp = 0 ; fp = 0
        #print(j,thr )
        '''
        当设定预测值≥阈值时（此时由于存在预测值=1的情况，所以tp！=0，即recall！=0；而fp有可能也！=0，则precision＜1），
        曲线不一定过（0，1）点；当设定预测值＞阈值时（此时由于不可能存在概率＞1的点，因此tp=fp=0，则recall=0，precision=1）
        '''
        for i in range(len(pos_prob_sort[np.where(pos_prob_sort>thr)])):             
            if y[i]==1:
                tp +=1
            else:
                fp +=1
        if (tp+fp)==0:
            recall.append(0)
            precision.append(1)
        else:    
            recall.append(tp/(len(pos)))
            precision.append(tp/(tp+fp))
        if j == 0: 
            auc += 0
        else:   
            auc += (recall[j]-recall[j-1])*(precision[j]+precision[j-1])*0.5    #求离散点连成的折线与坐标轴围成的面积
            #auc += (recall[j]-recall[j-1])*precision[j]            #求离散点坐标与轴围成的面积和
    
    plt.figure(figsize=(10,10))
    plt.plot(recall,precision,label=classname+"(AUPR: {:.4f})".format(auc),linewidth=2)
    plt.plot([0,1],[1,0], 'r--')
    plt.xlabel("Recall",fontsize=16)
    plt.ylabel("Precision",fontsize=16)
    plt.title("Precision Recall Curve",fontsize=17)
    plt.legend(fontsize=16) 
        
    #auc1 = trapz(precision, recall)   
    #auc2 = simps(precision, recall)   
    #print(auc1, auc2)         
    return precision,recall,auc


def AUPR_auto(pos_prob,y_true,classname):   
    precision, recall, thresholds = precision_recall_curve(y_true, pos_prob)
    print(thresholds)
    area = auc(recall, precision)
    
    plt.figure(figsize=(10,6))
    plt.plot(recall,precision,label=classname+"(AUPR: {:.3f})".format(area),linewidth=2)

    plt.xlabel("Recall",fontsize=16)
    plt.ylabel("Precision",fontsize=16)
    plt.title("Precision Recall Curve",fontsize=17)
    plt.legend(fontsize=16)
    
    # Compute Precision-Recall and plot curve   
    return precision,recall,area

def ROC(pos_prob, y_true, classname, threshold_num=100):
    pos_prob = np.array(pos_prob)
    y_true = np.array(y_true)
    pos = y_true[y_true==1]
    neg = y_true[y_true==0]
    print(pos.shape,neg.shape)
    y = y_true[pos_prob.argsort()[::-1]]
    pos_prob_sort=np.sort(pos_prob)[::-1]
    threshold=np.linspace(1,0,threshold_num)
    threshold_list=threshold.tolist()
    tpr_all = [] ; fpr_all = []
    auc=0
    x_step = 1/float(len(neg))
    y_step = 1/float(len(pos))
    print(x_step,len(neg),y_step,len(pos))
    y_sum = 0                                  # 用于计算AUC
    for j,thr in enumerate(threshold_list):
        tpr = 0 ; fpr = 0
        #print(j,thr )
        if thr>pos_prob_sort[0]:
            tpr_all.append(0)
            fpr_all.append(0)
        else:
            for i in range(len(pos_prob_sort[np.where(pos_prob_sort>thr)])):
                if y[i] == 1:
                    tpr += y_step
                else:
                    fpr += x_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
            auc += (fpr_all[j]-fpr_all[j-1])*(tpr_all[j]+tpr_all[j-1])*0.5

    '''
    plt.figure(figsize=(10,10))
    plt.plot(fpr_all,tpr_all,label=classname+"(AUC: {:.3f})".format(auc),linewidth=2)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel("False Positive Rate",fontsize=16)
    plt.ylabel("True Positive Rate",fontsize=16)
    plt.title("ROC Curve",fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    '''
    return tpr_all, fpr_all, auc

def ROC_auto(pos_prob,y_true,classname):
    fp, tp, thresholds = roc_curve(y_true, pos_prob)
    '''
    fp, tp = [0], [0]
    thr = np.linspace(1,0,33)
  
    for i ,element in enumerate(thresholds):   #根据阈值重采样
        if element <= thr:
            fp.append(fpr[i])
            tp.append(tpr[i])
    fp.append(1)
    tp.append(1)   
    '''     
    area = auc(fp, tp)
    
    plt.figure(figsize=(10,10))
    plt.plot(fp,tp,label=classname+"(AUC: {:.4f})".format(area),linewidth=2)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel("False Positive Rate",fontsize=16)
    plt.ylabel("True Positive Rate",fontsize=16)
    plt.title("ROC Curve",fontsize=16)
    plt.legend(fontsize=16) 
        
    return tp,fp,area        
# 获得总体TPR，FPR和相应的AUCtpr_lr,fpr_lr,auc_lr = get_roc(pos_prob_lr,y_test)  

def compute_pr_f1(y_pred,y_true,class_num=3,class_list=['EX&SE','HE&MA']):    
    f1_dict = {}
    if class_num == 3:
        class_list = ['EX','MA']
        print(np.array(y_pred).shape)
        #result_dict = {class_list[0]:,class_list[1]:}
        mf1=0
        for j,class_name in enumerate(class_list):
            if len(y_pred[j+1]) != len(y_true[j]):
                raise ValueError("Shape mismatch: y_pred and y_true must have the same shape.")
            pos_prob = np.zeros(len(y_pred[0]))
            for i in range(len(y_pred[0])):
                pred_1 = y_pred[0][i]
                pred_2 = y_pred[1][i]
                pred_3 = y_pred[2][i]           
                class_arr = np.array([pred_1,pred_2,pred_3])               
                item = np.argmax(class_arr)
                if item == j + 1:    #pred_1 is background
                    pos_prob[i] = 1
            pos_prob = pos_prob.astype(np.bool)  
            pos_true = np.array(y_true[j]).astype(np.bool)                 
            tp = np.logical_and(pos_prob, pos_true).sum()
            true_pos = pos_true.sum()
            prob_pos = pos_prob.sum()
            fp = prob_pos - tp
            tn = len(y_pred[0]) - true_pos - prob_pos + tp
            #print(tp,true_pos,prob_pos)
            recall = tp / true_pos            
            precision = tp / prob_pos
            sp = tn / (tn + fp)
            f1 = 2*recall*precision/(recall+precision)
            mf1 = mf1 + f1
            f1_dict.setdefault('class_name', []).append(class_name)
            f1_dict.setdefault('recall', []).append(recall)
            f1_dict.setdefault('precision', []).append(precision)
            f1_dict.setdefault('F1-score', []).append(f1)
            f1_dict.setdefault('specificity ', []).append(sp)
            print(class_name+' : ','recall = ',recall,' precision = ',precision,' F1-score= ',f1)
        f1_dict['mean_F1'] = mf1/len(class_list)
        print('mean_F1:%f'%(mf1/len(class_list)))
        '''
        class_list = ['EX&SE','HE&MA']
        print(np.array(y_pred).shape)
        #result_dict = {class_list[0]:,class_list[1]:}
        for j,class_name in enumerate(class_list):
            if len(y_pred[j+1]) != len(y_true[j]):
                raise ValueError("Shape mismatch: y_pred and y_true must have the same shape.")
            pos_prob = np.zeros(len(y_pred[0]))
            for i in range(len(y_pred[0])):
                pred_1 = y_pred[0][i]
                pred_2 = y_pred[1][i]
                pred_3 = y_pred[2][i]           
                class_arr = np.array([pred_1,pred_2,pred_3])               
                item = np.argmax(class_arr)
                if item == j + 1:    #pred_1 is background
                    pos_prob[i] = 1
            pos_prob = pos_prob.astype(np.bool)  
            pos_true = np.array(y_true[j]).astype(np.bool)                 
            tp = np.logical_and(pos_prob, pos_true).sum()
            true_pos = pos_true.sum()
            prob_pos = pos_prob.sum()
            #print(tp,true_pos,prob_pos)
            recall = tp / true_pos            
            precision = tp / prob_pos
            f1 = 2*recall*precision/(recall+precision)
            print(class_name+' : ','recall = ',recall,' precision = ',precision,' F1-score= ',f1)
        '''
            
    if class_num == 5:        
        class_list = ['EX','HE','MA','SE']
        print(np.array(y_pred).shape)
        #result_dict = {class_list[0]:,class_list[1]:}
        mf1=0
        for j,class_name in enumerate(class_list):
            if len(y_pred[j+1]) != len(y_true[j]):
                raise ValueError("Shape mismatch: y_pred and y_true must have the same shape.")
            pos_prob = np.zeros(len(y_pred[0]))
            for i in range(len(y_pred[0])):
                pred_1 = y_pred[0][i]
                pred_2 = y_pred[1][i]
                pred_3 = y_pred[2][i]
                pred_4 = y_pred[3][i]
                pred_5 = y_pred[4][i] 
                class_arr = np.array([pred_1,pred_2,pred_3,pred_4,pred_5])               
                item = np.argmax(class_arr)
                if item == j + 1:    #pred_1 is background
                    pos_prob[i] = 1
            pos_prob = pos_prob.astype(np.bool)  
            pos_true = np.array(y_true[j]).astype(np.bool)                 
            tp = np.logical_and(pos_prob, pos_true).sum()
            true_pos = pos_true.sum()
            prob_pos = pos_prob.sum()
            fp = prob_pos - tp
            tn = len(y_pred[0]) - true_pos - prob_pos + tp
            #print(tp,true_pos,prob_pos)
            recall = tp / true_pos            
            precision = tp / prob_pos
            sp = tn / (tn + fp)
            f1 = 2*recall*precision/(recall+precision)
            mf1 = mf1 + f1
            f1_dict.setdefault('class_name', []).append(class_name)
            f1_dict.setdefault('recall', []).append(recall)
            f1_dict.setdefault('precision', []).append(precision)
            f1_dict.setdefault('F1-score', []).append(f1)
            f1_dict.setdefault('specificity ', []).append(sp)
            print(class_name+' : ','recall = ',recall,' precision = ',precision,' F1-score= ',f1)
        f1_dict['mean_F1'] = mf1/len(class_list)
        print('mean_F1:%f'%(mf1/len(class_list)))
    if class_num == 1:
        class_list = class_list
        print(np.array(y_pred).shape)
        #result_dict = {class_list[0]:,class_list[1]:}
        for j,class_name in enumerate(class_list):
            if len(y_pred[j]) != len(y_true[j]):
                raise ValueError("Shape mismatch: y_pred and y_true must have the same shape.")
            pos_prob = np.zeros(len(y_pred[0]))
            for i in range(len(y_pred[0])):
                pred_1 = y_pred[0][i]
                
                class_arr = np.array([pred_1]) 
                #print(class_arr)
                #class_arr[np.where(class_arr)>0.5]=1
                if class_arr[0] >0.5:    #pred_1 is background
                    pos_prob[i] = 1
            pos_prob = pos_prob.astype(np.bool)  
            pos_true = np.array(y_true[j]).astype(np.bool)                 
            tp = np.logical_and(pos_prob, pos_true).sum()
            true_pos = pos_true.sum()
            prob_pos = pos_prob.sum()
            #print(tp,true_pos,prob_pos)
            recall = tp / true_pos            
            precision = tp / prob_pos
            f1 = 2*recall*precision/(recall+precision)
            print(class_name+' : ','recall = ',recall,' precision = ',precision,' F1-score= ',f1)
    return f1_dict
def compute_dice(val_path,truth_path,classes=3,task=None):
   
    def compute(img1,img2,empty_score=1.0):
        """
        Computes the Dice coefficient, a measure of set similarity.
        Parameters
        ----------
        im1 : array-like, bool
            Any array of arbitrary size. If not boolean, will be converted.
        im2 : array-like, bool
            Any other array of identical size. If not boolean, will be converted.
        Returns
        -------
        dice : float
            Dice coefficient as a float on range [0,1].
            Maximum similarity = 1
            No similarity = 0
            Both are empty (sum eq to zero) = empty_score

        Notes
        -----
        The order of inputs for `dice` is irrelevant. The result will be
        identical if `im1` and `im2` are switched.
        """
        if img1.shape != img2.shape:
            #print(img1.shape,'!=',img2.shape)
            img2 = cv2.resize(img2, img1.shape,interpolation = cv2.INTER_NEAREST)
            #raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
        im1 = np.squeeze(img1)
        im2 = np.squeeze(img2)
        im1 = np.asarray(im1).astype(np.bool)
        im2 = np.asarray(im2).astype(np.bool)

        

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return empty_score

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        return 2. * intersection.sum() / im_sum
    
    truth_path1 = os.path.join(truth_path,'EX')
    truth_path2 = os.path.join(truth_path,'HE')
    truth_path3 = os.path.join(truth_path,'MA')
    truth_path4 = os.path.join(truth_path,'SE')
    class_list1 = ['EX&SE','HE&MA']
    class_list2 = ['EX','HE','MA','SE']
    dice_dict = {}
    if classes == 3:
        dice1,dice2 = 0,0
        val_path1 = os.path.join(val_path,'EX&SE_predict')
        val_path2 = os.path.join(val_path,'HE&MA_predict')
        val_list1 = os.listdir(val_path1)
        val_list2 = os.listdir(val_path2)       
        for val_name in val_list1:
            #realname = val_name.split('_',2)[1]#DDR
            realname = val_name.split('_',3)[1]+'_'+val_name.split('_',3)[2]#IDRiD
            truth1 = cv2.imread(os.path.join(truth_path1,realname+'.png'),0)
            truth2 = cv2.imread(os.path.join(truth_path2,realname+'.png'),0)
            truth3 = cv2.imread(os.path.join(truth_path3,realname+'.png'),0)
            truth4 = cv2.imread(os.path.join(truth_path4,realname+'.png'),0)
            predict1 = cv2.imread(os.path.join(val_path1,val_name),0)
            predict2 = cv2.imread(os.path.join(val_path2,'HE&MA_'+realname+'_predict.png'),0)
            truth1 = cv2.bitwise_or(truth1,truth4)
            truth2 = cv2.bitwise_or(truth2,truth3)
            for class_name in class_list1:
                if class_name == 'EX&SE':
                    Dice = compute(predict1,truth1)
                    dice1 = dice1 + Dice
                elif class_name == 'HE&MA':
                    Dice = compute(predict2,truth2)
                    dice2 = dice2 + Dice
        dice1 = dice1/len(val_list1)
        dice2 = dice2/len(val_list1)
        print('EX&SE_Dice = ',dice1,'\n','HE&MA_Dice = ',dice2)
    elif classes == 5:
        dice1,dice2,dice3,dice4 = 0,0,0,0               
        val_path1 = os.path.join(val_path,'EX_predict')
        val_path2 = os.path.join(val_path,'HE_predict')
        val_path3 = os.path.join(val_path,'MA_predict')
        val_path4 = os.path.join(val_path,'SE_predict')
        val_list1 = os.listdir(val_path1)
        val_list2 = os.listdir(val_path2)       
        for val_name in val_list1:
            #realname = val_name.split('_',2)[1]#DDR
            realname = val_name.split('_',3)[1]+'_'+val_name.split('_',3)[2]#IDRiD
            truth1 = cv2.imread(os.path.join(truth_path1,realname+'.png'),0)
            truth2 = cv2.imread(os.path.join(truth_path2,realname+'.png'),0)
            truth3 = cv2.imread(os.path.join(truth_path3,realname+'.png'),0)
            truth4 = cv2.imread(os.path.join(truth_path4,realname+'.png'),0)
            #print(os.path.join(truth_path1,realname+'.png'))
            predict1 = cv2.imread(os.path.join(val_path1,'EX_'+realname+'_predict.png'),0)
            predict2 = cv2.imread(os.path.join(val_path2,'HE_'+realname+'_predict.png'),0)
            predict3 = cv2.imread(os.path.join(val_path3,'MA_'+realname+'_predict.png'),0)
            predict4 = cv2.imread(os.path.join(val_path4,'SE_'+realname+'_predict.png'),0)          
            for class_name in class_list2:
                if class_name == 'EX':
                    Dice = compute(predict1,truth1)
                    dice1 = dice1 + Dice
                elif class_name == 'HE':
                    Dice = compute(predict2,truth2)
                    dice2 = dice2 + Dice
                elif class_name == 'MA':
                    Dice = compute(predict3,truth3)
                    dice3 = dice3 + Dice
                elif class_name == 'SE':
                    Dice = compute(predict4,truth4)
                    dice4 = dice4 + Dice

        dice1 = dice1/len(val_list1)
        dice2 = dice2/len(val_list1)
        dice3 = dice3/len(val_list1)
        dice4 = dice4/len(val_list1)
        dice_dict['EX_Dice'], dice_dict['HE_Dice'], dice_dict['MA_Dice'], dice_dict['SE_Dice'] = dice1, dice2, dice3, dice4
        print('EX_Dice = ',dice1,'\n','HE_Dice = ',dice2,'\n','MA_Dice = ',dice3,'\n','SE_Dice = ',dice4)
    elif classes == 1:
        dice1 = 0 
        if task=='ex':
            val_path1 = os.path.join(val_path,'EX_predict')
            truth_path1 = os.path.join(truth_path,'EX')
        elif task=='he':
            val_path1 = os.path.join(val_path,'HE_predict')
            truth_path1 = os.path.join(truth_path,'HE')
        elif task=='ma':
            val_path1 = os.path.join(val_path,'MA_predict')  
            truth_path1 = os.path.join(truth_path,'MA')
        elif task=='se':
            val_path1 = os.path.join(val_path,'SE_predict')
            truth_path1 = os.path.join(truth_path,'SE')

        val_list1 = os.listdir(val_path1)
      
        for val_name in val_list1:
            #realname = val_name.split('_',2)[1]#DDR
            realname = val_name.split('_',3)[1]+'_'+val_name.split('_',3)[2]#IDRiD
            truth1 = cv2.imread(os.path.join(truth_path1,realname+'.png'),0)

            #print(os.path.join(truth_path1,realname+'.png'))
            predict1 = cv2.imread(os.path.join(val_path1,val_name),0)
            Dice = compute(predict1,truth1)
            dice1 = dice1 + Dice
        dice1 = dice1/len(val_list1)
        
        print(task+'_Dice = ',dice1)
    return dice_dict
        



