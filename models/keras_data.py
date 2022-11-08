from __future__ import print_function #from __future__ imports must occur at the beginning of the file
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from keras_preprocessing import image
import numpy as np 
import os
import glob
import skimage.transform as trans
from PIL import Image 
import os.path 
import math
import cv2 
import random
from skimage import data,io
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


'''
for tf1.13.1
'''
'''
from __future__ import print_function #from __future__ imports must occur at the beginning of the file
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from keras_preprocessing import image
import numpy as np 
import os
import glob
import skimage.transform as trans
from PIL import Image 
import os.path 
import math
import cv2 
import random
from skimage import data,io
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
'''
bg = [0,0,0]
EX = [255,255,255]
HE = [255,0,0]
MA = [0,0,255]
SE = [255,255,0]
clo=np.asarray([
      [165, 42, 42],
      [0, 192, 0],
      [196, 196, 196],
      [190, 153, 153],
      [180, 165, 180],
      [102, 102, 156],
      [128, 64, 255],
      [140, 140, 200],
      [170, 170, 170],
      [250, 170, 160],
      [96, 96, 96],
      [230, 150, 140],
      [128, 64, 128]])




clo=np.array([bg, EX, HE, MA,SE])
#clo=np.array([bg,HE, MA, EX,SE])
COLOR_DICT = clo
def run_aug(img,label,zoom_range=None,rotation_range=None, elastic_trans=None):
    #print(zoom_range,elastic_trans)
    def zoom(img,label,zoom_range):        
        img = image.random_zoom(img,zoom_range,row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant')        
        label= image.random_zoom(label,zoom_range,row_axis=0,col_axis=1,channel_axis=2,fill_mode='constant',interpolation_order=0)
        return img,label
    def rotation(img, label, deg):
        #print('deg:', deg)
        (h, w) = img.shape[0:2]
        (cX, cY) = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
        image = cv2.warpAffine(img, M, (w, h))
        label = cv2.warpAffine(label, M, (w, h))
        return image,label
    def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
        #print('elastic')
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.                
         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]   #(512,512)表示图像的尺寸                
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
        # 其中center_square是图像的中心，square_size=512//3=170
        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
        M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
        #默认使用 双线性插值，
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)

        # # random_state.rand(*shape) 会产生一个和 shape 一样打的服从[0,1]均匀分布的矩阵
        # * 2 - 1 是为了将分布平移到 [-1, 1] 的区间
        # 对random_state.rand(*shape)做高斯卷积，没有对图像做高斯卷积，为什么？因为论文上这样操作的
        # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
        # 实际上 dx 和 dy 就是在计算论文中弹性变换的那三步：产生一个随机的位移，将卷积核作用在上面，用 alpha 决定尺度的大小
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵                
        # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))                
        return map_coordinates(image, indices, order=0, mode='constant').reshape(shape)                

    img_shape=img.shape
    label_shape=label.shape
    n = random.randint(1,1)
    if n == 1:       
        for i in range(img.shape[0]):
            im=img[i]
            la=label[i]  
            if zoom_range!=None:
                r = random.uniform(zoom_range[0],zoom_range[1])      
                im,la = zoom(im,la,(r,r))
                img[i]=im
                label[i]=la
            if rotation_range != None:
                deg = random.randrange(0, 360, rotation_range)
                im, la = rotation(im, la, deg)
                img[i] = im
                label[i] = np.expand_dims(la,axis=-1)
            if elastic_trans!=None:
                #im = cv2.imread(os.path.join(img_path, img_list[i]))
                #print(im.shape,la.shape)
                if im.shape[-1]>1:
                    im1=im[:,:,0]
                    im2=im[:,:,1]
                    im3=im[:,:,2]
                    la1 = la[:,:,0]
                    #print(im1.shape,im2.shape,im3.shape,la1.shape)
                    im_merge = np.concatenate((im1[..., None],im2[..., None],im3[..., None], la1[..., None]), axis=2)   
                    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                           im_merge.shape[1] * 0.08)
                    # Split image and mask
                    im_t1 = im_merge_t[..., 0]
                    im_t2 = im_merge_t[..., 1]
                    im_t3 = im_merge_t[..., 2]
                    la_t1 = im_merge_t[..., 3] 
                    im[:,:,0]= im_t1
                    im[:,:,1]= im_t2
                    im[:,:,2]= im_t3
                    la[:,:,0]=la_t1
                    img[i] = im
                    label[i] = la
                else:
                    im1=im[:,:,0]
                    la1 = la[:,:,0]
                    im_merge = np.concatenate((im[..., None],la[..., None]), axis=2) 
                    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 1, im_merge.shape[1] * 0.025,
                                           im_merge.shape[1] * 0.00)
                    # Split image and mask
                    im[:,:,0]= im_merge_t[..., 0]
                    la[:,:,0]=im_merge_t[..., 1] 
                    img[i] = im
                    label[i] = la
                #print(im.shape)   
    return (img,label)   

def adjustData(img,label,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        label = label[:,:,:,0] if(len(label.shape) == 4) else label[:,:,0]
        #print('-----1-----', img.shape, label.shape)
        new_label = np.zeros(label.shape + (num_class,))
        #print('new_label.shape',new_label.shape)
        for i in range(num_class):
            #for one pixel in the image, find the class in label and convert it into one-hot vector
            #index = np.where(label == i)
            #index_label = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(label.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_label[index_label] = 1
            new_label[label == i,i] = 1
        #new_label = np.reshape(new_label,(new_label.shape[0],new_label.shape[1]*new_label.shape[2],new_label.shape[3])) if flag_multi_class else np.reshape(new_label,(new_label.shape[0]*new_label.shape[1],new_label.shape[2]))
        label = new_label
        
    elif(np.max(img) > 1):
        img = img / 255
        label = label /255
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    #print(label)
    return (img,label)



def trainGenerator(batch_size,
            aug_dict,
            label_args_dict,
            train_path,
            image_folder,
            label_folder,
            vs_label_folder='vessel_mask_zoom',
            od_label_folder='od_mask_zoom',
            image_color_mode = "rgb",
            label_color_mode = "grayscale",
            image_save_prefix  = "image",label_save_prefix = "label",
            flag_multi_class = True,
            shuffle=True,    
            num_class=5,
            save_to_dir = None,
            extra_aug=False,
            extra_aug_dict=None,
            target_size = (420,420),seed = 1):
    '''
    can generate image and label at the same time
    use the same seed for image_datagen and label_datagen to ensure the transformation for image and label is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    #print(extra_aug_dict,extra_aug)
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**label_args_dict)
    vs_label_datagen = ImageDataGenerator(**label_args_dict)
    od_label_datagen = ImageDataGenerator(**label_args_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        shuffle=shuffle,
        color_mode = image_color_mode,
        target_size = target_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        batch_size = batch_size,
        seed = seed)
    label_generator = label_datagen.flow_from_directory(
        train_path,       
        classes = [label_folder],
        class_mode = None,
        shuffle=shuffle,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = label_save_prefix,
        seed = seed)
    vs_label_generator = vs_label_datagen.flow_from_directory(
        train_path,
        classes=[vs_label_folder],
        class_mode=None,
        shuffle=shuffle,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed)
    od_label_generator = od_label_datagen.flow_from_directory(
        train_path,
        classes=[od_label_folder],
        class_mode=None,
        shuffle=shuffle,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, label_generator, vs_label_generator, od_label_generator)
    for (img, label, vs_label, od_label) in train_generator:
        #print('------2-------',img.shape,label.shape)
        if extra_aug==True:
            r = random.randint(0,1)
            if r <= 1:
                img,label = run_aug(img,label,**extra_aug_dict)
        img,label = adjustData(img,label,flag_multi_class,num_class)
        #yield (img,{'output1': label, 'output2': label, 'output3': label, 'output4': label, 'output5': label})
        yield (img, label)



def testGenerator(test_path,target_size = (420,420),flag_multi_class = True,as_gray = False):
    filelist=os.listdir(test_path)
    for filename in filelist:
        (realname, extension) = os.path.splitext(filename)
        img = io.imread(os.path.join(test_path,filename),as_gray = as_gray)
        img = img / 255
  
        img = trans.resize(img,target_size,mode = 'constant')
        #print(img.shape)
    
        if as_gray:
            img = np.reshape(img,img.shape+(1,))
        else: 
            img=img
        img = np.reshape(img,(1,)+img.shape)
        #print(img.shape)
        yield img


def geneTrainNpy(image_path,label_path,flag_multi_class = True,num_class = 5,image_prefix = "image",label_prefix = "label",image_as_gray = False, label_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    label_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        label = io.imread(item.replace(image_path,label_path).replace(image_prefix,label_prefix),as_gray = label_as_gray)
        label = np.reshape(label,label.shape + (1,)) if label_as_gray else label
        img,label = adjustData(img,label,flag_multi_class,num_class)
        image_arr.append(img)
        label_arr.append(label)
    image_arr = np.array(image_arr)
    label_arr = np.array(label_arr)
    return image_arr,label_arr



def saveResult(save_path,test_path,target_size,npyfile,flag_multi_class=True,classes=5,task=None):
    filelist_test=os.listdir(test_path)
    name=[]   
    for filename in filelist_test:
        (realname, extension) = os.path.splitext(filename)
        name.append(realname)
    for i,item in enumerate(npyfile):
        cd=[]
        if flag_multi_class:
            if classes==5:
                img = item
                img_out = np.zeros(img[:, :, 0].shape + (3,))
                img_out_EX = img_out.copy()
                img_out_HE = img_out.copy()
                img_out_MA = img_out.copy()
                img_out_SE = img_out.copy()
                #print(img)
                for row in range(img.shape[0]):
                    for col in range(img.shape[1]):                    
                        index_of_class = np.argmax(img[row, col])                   
                        img_out[row, col] = COLOR_DICT[index_of_class]
                        if index_of_class==1:
                            img_out_EX[row, col]=[255,255,255] 
                            img_out_EX = img_out_EX.astype(np.uint8)
                        elif index_of_class==2:
                            img_out_HE[row, col]=[255,255,255]
                            img_out_HE = img_out_HE.astype(np.uint8)
                        elif index_of_class==3:
                            img_out_MA[row, col]=[255,255,255]
                            img_out_MA = img_out_MA.astype(np.uint8)
                        elif index_of_class==4:
                            img_out_SE[row, col]=[255,255,255]
                            img_out_SE = img_out_SE.astype(np.uint8)

                img_out_EX = trans.resize(img_out_EX,target_size)
                img_out_HE = trans.resize(img_out_HE,target_size)
                img_out_MA = trans.resize(img_out_MA,target_size)
                img_out_SE = trans.resize(img_out_SE,target_size)

                cd.append(img_out_EX)
                cd.append(img_out_HE)
                cd.append(img_out_MA)
                cd.append(img_out_SE)
                img = img_out.astype(np.uint8)
                img_test=io.imread(os.path.join(test_path, '%s.png' % name[i]))
                img_test = trans.resize(img_test,target_size)
                img = trans.resize(img,target_size)
                io.imsave(os.path.join(save_path, '%s.png' % name[i]), img_test)
                io.imsave(os.path.join(save_path, '%s_predict.png' % name[i]), img)
                for j,s in enumerate(['EX','HE','MA','SE']):
                    path=os.path.join(save_path,s+'_predict')
                    if not os.path.exists(path):
                        os.makedirs(path)
                    io.imsave(os.path.join(path,'%s_predict.png' % (s+'_'+name[i])), cd[j])
            elif classes==3:
                img = item
                img_out = np.zeros(img[:, :, 0].shape + (3,))
                img_out_W = img_out.copy()
                img_out_R = img_out.copy()              
                #print(img)
                for row in range(img.shape[0]):
                    for col in range(img.shape[1]):                    
                        index_of_class = np.argmax(img[row, col])                   
                        img_out[row, col] = COLOR_DICT[index_of_class]
                        if index_of_class==1:
                            img_out_W[row, col]=[255,255,255] 
                            img_out_W = img_out_W.astype(np.uint8)
                        elif index_of_class==2:
                            img_out_R[row, col]=[255,255,255]
                            img_out_R = img_out_R.astype(np.uint8)                       
                img_out_W = trans.resize(img_out_W,target_size)
                img_out_R = trans.resize(img_out_R,target_size)            
                cd.append(img_out_W)
                cd.append(img_out_R)                
                img = img_out.astype(np.uint8)
                img_test = io.imread(os.path.join(test_path, '%s.png' % name[i]))
                img_test = trans.resize(img_test,target_size)
                img = trans.resize(img,target_size)
                io.imsave(os.path.join(save_path, '%s.png' % name[i]), img_test)
                io.imsave(os.path.join(save_path, '%s_predict.png' % name[i]), img)
                if task==None:
                    for j,s in enumerate(['EX&SE','HE&MA']):
                        path=os.path.join(save_path,s+'_predict')
                        if not os.path.exists(path):
                            os.makedirs(path)
                        io.imsave(os.path.join(path,'%s_predict.png' % (s+'_'+name[i])), cd[j])
                elif task=='ma_he':
                    for j,s in enumerate(['HE','MA']):
                        path=os.path.join(save_path,s+'_predict')
                        if not os.path.exists(path):
                            os.makedirs(path)
                        io.imsave(os.path.join(path,'%s_predict.png' % (s+'_'+name[i])), cd[j])
                elif task=='ex_ma':
                    for j,s in enumerate(['EX','MA']):
                        path=os.path.join(save_path,s+'_predict')
                        if not os.path.exists(path):
                            os.makedirs(path)
                        io.imsave(os.path.join(path,'%s_predict.png' % (s+'_'+name[i])), cd[j])
        else:
            if task=='ex':
                foldname='EX'
            elif task=='he':
                foldname='HE'
            elif task=='ma':
                foldname='MA'
            elif task=='se':
                foldname='SE'    
            img = item[:, :, 0]
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            img = img * 255.
            img_test=io.imread(os.path.join(test_path, '%s.png' % name[i]))
            img_test = trans.resize(img_test,target_size)
            img = trans.resize(img,target_size)
            io.imsave(os.path.join(save_path, '%s.png' % name[i]), img_test)
            path=os.path.join(save_path,foldname+'_predict')              
            if not os.path.exists(path):
                os.makedirs(path)
            io.imsave(os.path.join(path,'%s_predict.png' % (foldname+'_'+name[i])), img)
            
def drawline(origin_path,mask_path,classes=5):
    
    def draw(img_origin,img_mask,R,G,B):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        h,w=img_origin.shape[:2]
        #cv2.imshow("f",img_origin)
        #cv2.imshow("f1",img_mask)
        #cv2.waitKey(0)
        img_mask = cv2.dilate(img_mask,kernel,iterations=0)
        img_mask,contours1,hierarchy1= cv2.findContours(img_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
       # print(contours1)
        for i in range(0,len(contours1),1):
            cv2.drawContours(img_origin,contours1,i,(B,G,R),1) 
        return img_origin

    if classes==5:
        mask_path1=os.path.join(mask_path,'EX_predict')
        mask_path2=os.path.join(mask_path,'HE_predict')
        mask_path3=os.path.join(mask_path,'MA_predict')
        mask_path4=os.path.join(mask_path,'SE_predict')
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀

            img_origin=cv2.imread(os.path.join(origin_path,fileName))

            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('EX_'+realname)),0)
            img_mask2=cv2.imread(os.path.join(mask_path2,'%s_predict.png'%('HE_'+realname)),0)
            img_mask3=cv2.imread(os.path.join(mask_path3,'%s_predict.png'%('MA_'+realname)),0)
            img_mask4=cv2.imread(os.path.join(mask_path4,'%s_predict.png'%('SE_'+realname)),0)
            
            img_origin=draw(img_origin,img_mask1,255,255,255)
            img_origin=draw(img_origin,img_mask2,255,0,0)
            img_origin=draw(img_origin,img_mask3,0,0,255)
            img_origin=draw(img_origin,img_mask4,255,255,0)
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin)
    elif classes==3:
        mask_path1=os.path.join(mask_path,'EX&SE_predict')
        mask_path2=os.path.join(mask_path,'HE&MA_predict')
        
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('EX&SE_'+realname)),0)
            img_mask2=cv2.imread(os.path.join(mask_path2,'%s_predict.png'%('HE&MA_'+realname)),0)
            
            

            img_origin=draw(img_origin,img_mask1,255,255,255)
            img_origin=draw(img_origin,img_mask2,255,0,0)
            
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin)    
        
def drawline_truth(origin_path,mask_path,save_path,classes=5,target_size=(1024,1024)):
    
    mask_path1=os.path.join(mask_path,'EX')
    mask_path2=os.path.join(mask_path,'HE')
    mask_path3=os.path.join(mask_path,'MA')
    mask_path4=os.path.join(mask_path,'SE')
    fileList1 = os.listdir(origin_path)
    print(fileList1)
    for fileName in fileList1:
        (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀

        img_origin = cv2.imread(os.path.join(origin_path, fileName))
        img_origin = cv2.resize(img_origin, target_size)
        img_mask1 = cv2.imread(os.path.join(mask_path1, fileName), 0)
        img_mask1 = cv2.resize(img_mask1, target_size, interpolation=cv2.INTER_NEAREST)
        img_mask2 = cv2.imread(os.path.join(mask_path2, fileName), 0)
        img_mask2 = cv2.resize(img_mask2, target_size, interpolation=cv2.INTER_NEAREST)
        img_mask3 = cv2.imread(os.path.join(mask_path3, fileName), 0)
        img_mask3 = cv2.resize(img_mask3, target_size, interpolation=cv2.INTER_NEAREST)
        img_mask4 = cv2.imread(os.path.join(mask_path4, fileName), 0)
        img_mask4 = cv2.resize(img_mask4, target_size, interpolation=cv2.INTER_NEAREST)
        def draw(img_origin,img_mask,R,G,B):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
            h,w=img_origin.shape[:2]
            #cv2.imshow("f",img_origin)
            #cv2.imshow("f1",img_mask)
            #cv2.waitKey(0)
            img_mask = cv2.dilate(img_mask,kernel,iterations=0)

            img_mask,contours1,hierarchy1= cv2.findContours(img_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
           # print(contours1)
            for i in range(0,len(contours1),1):
                cv2.drawContours(img_origin,contours1,i,(B,G,R),1) 
            return img_origin
        if classes==5:
            img_origin=draw(img_origin,img_mask1,255,0,0)
            img_origin=draw(img_origin,img_mask2,0,255,0)
            img_origin=draw(img_origin,img_mask3,0,0,255)
            img_origin=draw(img_origin,img_mask4,255,255,0)
        elif classes==3:
            img_origin=draw(img_origin,img_mask1,255,255,255)
            img_origin=draw(img_origin,img_mask2,255,0,0)
            img_origin=draw(img_origin,img_mask3,250,0,0)
            img_origin=draw(img_origin,img_mask4,255,255,255)
                
        cv2.imwrite(os.path.join(save_path, '%s_groundtruth.png'% realname), img_origin) 
def drawmask(origin_path,mask_path,classes=5,task=None,target_size=(1024,1024)):
    
    def draw(img_origin,img_mask,R,G,B):
        img_origin[np.where(img_mask>127)]=[B,G,R]
        return img_origin

    if classes==5:
        mask_path1=os.path.join(mask_path,'EX_predict')
        mask_path2=os.path.join(mask_path,'HE_predict')
        mask_path3=os.path.join(mask_path,'MA_predict')
        mask_path4=os.path.join(mask_path,'SE_predict')
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀

            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_origin = cv2.resize(img_origin,target_size)
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('EX_'+realname)),0)
            img_mask2=cv2.imread(os.path.join(mask_path2,'%s_predict.png'%('HE_'+realname)),0)
            img_mask3=cv2.imread(os.path.join(mask_path3,'%s_predict.png'%('MA_'+realname)),0)
            img_mask4=cv2.imread(os.path.join(mask_path4,'%s_predict.png'%('SE_'+realname)),0)
            
            img_origin=draw(img_origin,img_mask1,255,0,0)
            img_origin=draw(img_origin,img_mask2,0,255,0)
            img_origin=draw(img_origin,img_mask3,0,0,255)
            img_origin=draw(img_origin,img_mask4,255,255,0)
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin)
    elif classes==3 and task=='ex_ma':
        mask_path1=os.path.join(mask_path,'EX_predict')
        mask_path2=os.path.join(mask_path,'MA_predict')
        
        fileList1 = os.listdir(origin_path)    
        print(fileList1)
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('EX_'+realname)),0)
            img_mask2=cv2.imread(os.path.join(mask_path2,'%s_predict.png'%('MA_'+realname)),0)
            
            

            img_origin=draw(img_origin,img_mask1,255,255,255)
            img_origin=draw(img_origin,img_mask2,0,0,255)
            
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin) 
    elif classes==1:
        if task=='ex':
            mask_path1=os.path.join(mask_path,'EX_predict')
            fileList1 = os.listdir(origin_path)    
            for fileName in fileList1:
                (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
                img_origin=cv2.imread(os.path.join(origin_path,fileName))
                img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('EX_'+realname)),0)
                img_origin=draw(img_origin,img_mask1,255,0,0)
                cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin) 
        elif task=='he':
            mask_path1=os.path.join(mask_path,'HE_predict')
            fileList1 = os.listdir(origin_path)    
            for fileName in fileList1:
                (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
                img_origin=cv2.imread(os.path.join(origin_path,fileName))
                img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('HE_'+realname)),0)
                img_origin=draw(img_origin,img_mask1,0,255,0) 
                cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin) 
        elif task=='ma':
            mask_path1=os.path.join(mask_path,'MA_predict')
            fileList1 = os.listdir(origin_path)    
            for fileName in fileList1:
                (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
                img_origin=cv2.imread(os.path.join(origin_path,fileName))
                img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('MA_'+realname)),0)
                img_origin=draw(img_origin,img_mask1,0,0,255)
                cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin) 
        elif task=='se':
            mask_path1=os.path.join(mask_path,'SE_predict')
            fileList1 = os.listdir(origin_path)    
            for fileName in fileList1:
                (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
                img_origin=cv2.imread(os.path.join(origin_path,fileName))
                img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('SE_'+realname)),0)
                img_origin=draw(img_origin,img_mask1,255,255,0)
                cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin) 
def drawmask_truth(origin_path,mask_path,save_path,classes=5,task=None,target_size=(1024,1024)):
    if task=='ex_ma' and classes==3:
        mask_path1=os.path.join(mask_path,'EX')
        mask_path2=os.path.join(mask_path,'MA')
    else:    
        mask_path1=os.path.join(mask_path,'EX')
        mask_path2=os.path.join(mask_path,'HE')
        mask_path3=os.path.join(mask_path,'MA')
        mask_path4=os.path.join(mask_path,'SE')
    fileList1 = os.listdir(origin_path)
    def draw(img_origin,img_mask,R,G,B):
        img_origin[np.where(img_mask>127)]=[B,G,R]
        return img_origin
    print(fileList1)
    for fileName in fileList1:
        (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀

        img_origin=cv2.imread(os.path.join(origin_path,fileName))
        img_origin = cv2.resize(img_origin,target_size)
        
        
       
        if classes==5:           
            img_mask1=cv2.imread(os.path.join(mask_path1,fileName),0)
            img_mask1 = cv2.resize(img_mask1, target_size, interpolation=cv2.INTER_NEAREST)
            img_mask2=cv2.imread(os.path.join(mask_path2,fileName),0)
            img_mask2 = cv2.resize(img_mask2, target_size, interpolation=cv2.INTER_NEAREST)
            img_mask3=cv2.imread(os.path.join(mask_path3,fileName),0)
            img_mask3 = cv2.resize(img_mask3, target_size, interpolation=cv2.INTER_NEAREST)
            img_mask4=cv2.imread(os.path.join(mask_path4,fileName),0)
            img_mask4 = cv2.resize(img_mask4, target_size, interpolation=cv2.INTER_NEAREST)
            img_origin=draw(img_origin,img_mask1,255,0,0)
            img_origin=draw(img_origin,img_mask2,0,255,0)
            img_origin=draw(img_origin,img_mask3,0,0,255)
            img_origin=draw(img_origin,img_mask4,255,255,0)
        elif classes==3 and task=='ex_ma':
            img_mask1=cv2.imread(os.path.join(mask_path1,fileName),0)
            img_mask2=cv2.imread(os.path.join(mask_path2,fileName),0)
            img_origin=draw(img_origin,img_mask1,255,255,255)
            #img_origin=draw(img_origin,img_mask2,255,0,0)
            img_origin=draw(img_origin,img_mask2,0,0,255)
            #img_origin=draw(img_origin,img_mask4,255,255,255)
           
        elif classes==1:
            if task=='ex':
                img_mask1=cv2.imread(os.path.join(mask_path1,fileName),0)
                img_origin=draw(img_origin,img_mask1,255,0,0)
            elif task=='he':
                img_mask2=cv2.imread(os.path.join(mask_path2,fileName),0)
                img_origin=draw(img_origin,img_mask2,0,255,0)
            elif task=='ma':
                img_mask3=cv2.imread(os.path.join(mask_path3,fileName),0)
                img_origin=draw(img_origin,img_mask3,0,0,255)
            elif task=='se':
                img_mask4=cv2.imread(os.path.join(mask_path4,fileName),0)
                img_origin=draw(img_origin,img_mask4,255,255,0)
        cv2.imwrite(os.path.join(save_path, '%s_groundtruth.png'% realname), img_origin)        
def drawline_d2(origin_path,mask_path,classes=3,task=None):
    
    def draw(img_origin,img_mask,R,G,B):
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
                h,w=img_origin.shape[:2]
                #cv2.imshow("f",img_origin)
                #cv2.imshow("f1",img_mask)
                #cv2.waitKey(0)
                img_mask = cv2.dilate(img_mask,kernel,iterations=0)
                img_mask,contours1,hierarchy1= cv2.findContours(img_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
               # print(contours1)
                for i in range(0,len(contours1),1):
                    cv2.drawContours(img_origin,contours1,i,(B,G,R),1) 
                return img_origin

    if task=='ma_he':
        mask_path1=os.path.join(mask_path,'HE_predict')
        mask_path2=os.path.join(mask_path,'MA_predict')
        
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('HE_'+realname)),0)
            img_mask2=cv2.imread(os.path.join(mask_path2,'%s_predict.png'%('MA_'+realname)),0)
            
            
            img_origin=draw(img_origin,img_mask1,255,0,0)
            img_origin=draw(img_origin,img_mask2,250,128,128)
            
            
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin)    
    elif task=='ex_se':
        mask_path1=os.path.join(mask_path,'EX_predict')
        mask_path2=os.path.join(mask_path,'SE_predict')
        
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('EX_'+realname)),0)
            img_mask2=cv2.imread(os.path.join(mask_path2,'%s_predict.png'%('SE_'+realname)),0)
            
            

            img_origin=draw(img_origin,img_mask1,255,255,255)
            img_origin=draw(img_origin,img_mask2,255,255,0)
            
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin) 
    elif task=='ex':
        mask_path1=os.path.join(mask_path,'EX_predict')    
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('EX_'+realname)),0)
            img_origin=draw(img_origin,img_mask1,255,255,255)           
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin)
    elif task=='he':
        mask_path1=os.path.join(mask_path,'HE_predict')    
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('HE_'+realname)),0)
            img_origin=draw(img_origin,img_mask1,255,0,0)           
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin)          
    elif task=='ma':
        mask_path1=os.path.join(mask_path,'MA_predict')    
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('MA_'+realname)),0)
            img_origin=draw(img_origin,img_mask1,250,128,128)           
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin) 
    elif task=='SE':
        mask_path1=os.path.join(mask_path,'SE_predict')    
        fileList1 = os.listdir(origin_path)    
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀
            img_origin=cv2.imread(os.path.join(origin_path,fileName))
            img_mask1=cv2.imread(os.path.join(mask_path1,'%s_predict.png'%('SE_'+realname)),0)
            img_origin=draw(img_origin,img_mask1,255,255,0)           
            cv2.imwrite(os.path.join(mask_path, '%s_predict.png'% realname), img_origin)          
            
def drawline_truth_d2(origin_path,mask_path,save_path,classes=3,task=None):
    
    mask_path1=os.path.join(mask_path,'EX')
    mask_path2=os.path.join(mask_path,'HE')
    mask_path3=os.path.join(mask_path,'MA')
    mask_path4=os.path.join(mask_path,'SE')
    fileList1 = os.listdir(origin_path)
    print(fileList1)
    for fileName in fileList1:
        (realname, extension) = os.path.splitext(fileName) #分离文件名和后缀

        img_origin=cv2.imread(os.path.join(origin_path,fileName))
        img_mask1=cv2.imread(os.path.join(mask_path1,fileName),0)
        img_mask2=cv2.imread(os.path.join(mask_path2,fileName),0)
        img_mask3=cv2.imread(os.path.join(mask_path3,fileName),0)
        img_mask4=cv2.imread(os.path.join(mask_path4,fileName),0)
        def draw(img_origin,img_mask,R,G,B):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
            h,w=img_origin.shape[:2]
            #cv2.imshow("f",img_origin)
            #cv2.imshow("f1",img_mask)
            #cv2.waitKey(0)
            img_mask = cv2.dilate(img_mask,kernel,iterations=0)

            img_mask,contours1,hierarchy1= cv2.findContours(img_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
           # print(contours1)
            for i in range(0,len(contours1),1):
                cv2.drawContours(img_origin,contours1,i,(B,G,R),1) 
            return img_origin
        if task=='ma_he':
            
            img_origin=draw(img_origin,img_mask2,255,0,0)
            img_origin=draw(img_origin,img_mask3,250,128,128)
            
        elif task=='ex_se':
            img_origin=draw(img_origin,img_mask1,255,255,255)          
            img_origin=draw(img_origin,img_mask4,255,255,0)
        elif task=='ex':
            img_origin=draw(img_origin,img_mask1,255,255,255)          
        elif task=='he':
            img_origin=draw(img_origin,img_mask1,255,0,0)  
        elif task=='ma':
            img_origin=draw(img_origin,img_mask1,250,128,128)  
        elif task=='se':
            img_origin=draw(img_origin,img_mask1,255,255,0)  
        cv2.imwrite(os.path.join(save_path, '%s_groundtruth.png'% realname), img_origin) 
        
                