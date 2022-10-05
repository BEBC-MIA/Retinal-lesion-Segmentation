import cv2
import numpy as np
import os
import random
import math
from math import fabs, sin, cos, radians
from scipy.stats import mode
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import os.path
import time


def blend(lesion, background, mask, center, label, lesion_class=None, type=cv2.NORMAL_CLONE):
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_d = cv2.dilate(mask, kernel, iterations=5)
    # dilate lesion masks to make sure that possion blending can not influence the interior of lesions excessively
    mixed_img = cv2.seamlessClone(lesion, background, mask_d, center, type)
    mixed_label = label.copy()
    if lesion_class == 'EX':
        mask[np.where(mask == 255)] = 1
    elif lesion_class == 'HE':
        mask[np.where(mask == 255)] = 2
    elif lesion_class == 'MA':
        mask[np.where(mask == 255)] = 3
    elif lesion_class == 'SE':
        mask[np.where(mask == 255)] = 4
    else:
        print('please input lesion-class')
    for i in range(h):
        for j in range(w):
            if mask[i, j] != 0:
                mixed_label[center[1] - h // 2 + i, center[0] - w // 2 + j] = mask[
                    i, j]  # add new lesion labels in original label-image
    return mixed_img, mixed_label


def aug_data(img, label,
             rescale=False,
             rot=False,
             fl=False,
             elastic_trans=False,
             rescale_rate=1,
             degree=0,
             flipCode=0,
             filled_color=-1):
    def zoom(img, rate):
        h, w = img.shape[1], img.shape[0]
        new_h = round(h * rate)
        new_w = round(w * rate)
        if len(img.shape) == 3:
            inter = cv2.INTER_LINEAR
        else:
            inter = cv2.INTER_NEAREST
        new_img = cv2.resize(img, (new_h, new_w), interpolation=inter)
        #print(new_img.shape,'n',img.shape)
        return new_img

    def rotation(img, degree, filled_color=filled_color):  # rotation
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        if filled_color == -1:
            filled_color = mode([img[0, 0], img[0, -1],
                                 img[-1, 0], img[-1, -1]]).mode[0]
        if np.array(filled_color).shape[0] == 2:
            if isinstance(filled_color, int):
                filled_color = (filled_color, filled_color, filled_color)
        else:
            filled_color = tuple([int(i) for i in filled_color])
        height, width = img.shape[:2]
        # 旋转后的尺寸
        height_new = int(width * fabs(sin(radians(degree))) +
                         height * fabs(cos(radians(degree))))
        width_new = int(height * fabs(sin(radians(degree))) +
                        width * fabs(cos(radians(degree))))
        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        mat_rotation[0, 2] += (width_new - width) / 2
        mat_rotation[1, 2] += (height_new - height) / 2
        # Pay attention to the type of elements of filler_color, which should be
        # the int in pure python, instead of those in numpy.
        img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                     borderValue=filled_color)
        # 填充四个角
        mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
        mask[:] = 0
        seed_points = [(0, 0), (0, height_new - 1), (width_new - 1, 0),
                       (width_new - 1, height_new - 1)]
        for i in seed_points:
            cv2.floodFill(img_rotated, mask, i, filled_color)
        if len(img_rotated.shape) == 2:
            _, img_rotated = cv2.threshold(img_rotated, 127, 255, cv2.THRESH_BINARY)
        return img_rotated

    def flip(img, flipCode=flipCode):  # flip
        flip_img = cv2.flip(img, flipCode)
        return flip_img

    def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
        # 其中center_square是图像的中心，square_size=512//3=170
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
        M = cv2.getAffineTransform(pts1, pts2)  # 获取变换矩阵
        # 默认使用 双线性插值，
        image[:, :, :3] = cv2.warpAffine(image[:, :, :3], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT)
        image[:, :, 3] = cv2.warpAffine(image[:, :, 3], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT,
                                        flags=0)

        # # random_state.rand(*shape) 会产生一个和 shape 一样打的服从[0,1]均匀分布的矩阵
        # * 2 - 1 是为了将分布平移到 [-1, 1] 的区间
        # 对random_state.rand(*shape)做高斯卷积，没有对图像做高斯卷积，为什么？因为论文上这样操作的
        # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
        # 实际上 dx 和 dy 就是在计算论文中弹性变换的那三步：产生一个随机的位移，将卷积核作用在上面，用 alpha 决定尺度的大小
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)  # 构造一个尺寸与dx相同的O矩阵
        # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  # 网格采样点函数
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
        return map_coordinates(image, indices, order=0, mode='nearest').reshape(shape)

    if rescale:
        img = zoom(img, rate=rescale_rate)
        label = zoom(label, rate=rescale_rate)
    if fl:
        img = flip(img, flipCode=flipCode)
        label = flip(label, flipCode=flipCode)
    if rot:
        img = rotation(img, degree, filled_color=filled_color)
        label = rotation(label, degree, filled_color=filled_color)
    if elastic_trans:
        if img.shape[-1] > 1:
            im1 = img[:, :, 0]
            im2 = img[:, :, 1]
            im3 = img[:, :, 2]
            # la1 = la[:,:,0]
            la1 = label
            im_merge = np.concatenate((im1[..., None],
                                       im2[..., None],
                                       im3[..., None],
                                       la1[..., None]), axis=2)

            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                           im_merge.shape[1] * 0.08)
            # Split image and mask
            im_t1 = im_merge_t[..., 0]
            im_t2 = im_merge_t[..., 1]
            im_t3 = im_merge_t[..., 2]
            label = im_merge_t[..., 3]
            img[:, :, 0] = im_t1
            img[:, :, 1] = im_t2
            img[:, :, 2] = im_t3
        else:
            im1 = img
            la1 = label
            im_merge = np.concatenate((im1[..., None],
                                       la1[..., None]), axis=2)
            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                           im_merge.shape[1] * 0.08)
            # Split image and mask
            img = im_merge_t[..., 0]
            label = im_merge_t[..., 1]
    return img, label


# 计算class_weight
def class_count(path, class_num):
    fl = os.listdir(path)
    p = np.zeros(class_num)
    total = np.zeros(class_num)
    n = 0
    for fn in fl:
        img = cv2.imread(os.path.join(path, fn), 0)
        img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4), interpolation=0) #减小计算量
        h, w = img.shape
        img = img.reshape(h * w)
        for i in range(class_num):
            p[i] = sum(img == i)
        for m in range(len(p)):
            total[m] = total[m] + p[m]
        print(fn)
    most = max(total)
    percentage = total / h / w / len(fl)
    for m in range(len(total)):
        total[m] = most / total[m]
    print(class_num)
    print('class-weight:', total, '\nclass-percentage:', percentage)


def main(args,
         aug=True):
    type = cv2.NORMAL_CLONE
    dataset = args.dataset
    material_dir = os.path.join(args.material_dir, dataset)
    random_dens = {}
    dens_dict = {}
    if dataset == 'IDRiD':
        original_data_dir = os.path.join(args.original_data_dir, dataset, 'train/4 classes')
        lesion_class = ['EX', 'HE', 'MA', 'SE']
        random_dens['EX'] = tuple(map(int, args.dens_EX.split('_')))
        random_dens['HE'] = tuple(map(int, args.dens_HE.split('_')))
        random_dens['MA'] = tuple(map(int, args.dens_MA.split('_')))
        random_dens['SE'] = tuple(map(int, args.dens_SE.split('_')))
    if dataset == 'e_ophtha':
        original_data_dir = os.path.join(args.original_data_dir, dataset, 'train/2 classes')
        lesion_class = ['EX', 'MA']
        random_dens['EX'] = tuple(map(int, args.dens_EX.split('_')))
        random_dens['MA'] = tuple(map(int, args.dens_MA.split('_')))
    img_dir = os.path.join(original_data_dir, 'image_zoom_hd')  # whole RGB images       IDRiD=image_zoom_hd
    label_dir = os.path.join(original_data_dir, 'label_zoom_hd')  # label for whole images      IDRiD=label_zoom_hd
    vessel_mask_dir = os.path.join(original_data_dir, 'vessel_mask_zoom_hd')  # label for vessel   IDRiD=vessel_mask_zoom_hd
    od_mask_dir = os.path.join(original_data_dir, 'od_mask_zoom_hd')  # label for vessel      IDRiD=od_mask_zoom_hd
    blended_img_dir = os.path.join(original_data_dir, 'image_zoom_blend_hd_aug'+str(args.aug_rate+1)+'_EX'+str(random_dens.get('EX', (0,0))[1])+'_HE'+str(random_dens.get('HE', (0,0))[1])+'_MA'+str(random_dens.get('MA', (0,0))[1])+'_SE'+str(random_dens.get('SE', (0,0))[1]))       #IDRiD=image_zoom_blend_hd_aug20_EX60_HE15_MA100_SE0
    blended_label_dir = os.path.join(original_data_dir, 'label_zoom_blend_hd_aug'+str(args.aug_rate+1)+'_EX'+str(random_dens.get('EX', (0,0))[1])+'_HE'+str(random_dens.get('HE', (0,0))[1])+'_MA'+str(random_dens.get('MA', (0,0))[1])+'_SE'+str(random_dens.get('SE', (0,0))[1]))
    if not os.path.exists(blended_img_dir):
        os.makedirs(blended_img_dir)
    if not os.path.exists(blended_label_dir):
        os.makedirs(blended_label_dir)

    EX_cr_img_dir = os.path.join(material_dir, 'image/EX')  # cropped RGB images
    HE_cr_img_dir = os.path.join(material_dir, 'image/HE')
    MA_cr_img_dir = os.path.join(material_dir, 'image/MA')
    SE_cr_img_dir = os.path.join(material_dir, 'image/SE')
    EX_cr_label_dir = os.path.join(material_dir, 'label/EX')  # cropped label for 4 kinds of lesions
    HE_cr_label_dir = os.path.join(material_dir, 'label/HE')
    MA_cr_label_dir = os.path.join(material_dir, 'label/MA')
    SE_cr_label_dir = os.path.join(material_dir, 'label/SE')

    or_img_list = os.listdir(img_dir)
    print(or_img_list)
    '''
    for or_img_fullname in or_img_list:
        or_img_name, extension = os.path.splitext(or_img_fullname)
        background = cv2.imread(os.path.join(img_dir, or_img_fullname))
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        _, background_binary = cv2.threshold(background_gray, 30, 255,
                                             cv2.THRESH_BINARY)  # getting mask for rgb background
        label = cv2.imread(os.path.join(label_dir, or_img_fullname), 0)

        cv2.imshow('background', background)
        cv2.imshow('label', label * 50)


        H, W = background.shape[:2]
    '''
    start = time.time()
    for r in range(0, args.aug_rate+0, 1):
        for or_img_fullname in or_img_list:
            print(or_img_fullname)
            or_img_name, extension = os.path.splitext(or_img_fullname)
            background = cv2.imread(os.path.join(img_dir, or_img_fullname))
            background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            _, background_binary = cv2.threshold(background_gray, 30, 255,
                                                 cv2.THRESH_BINARY)  # getting mask for rgb background
            label = cv2.imread(os.path.join(label_dir, or_img_fullname), 0)
            vessel_mask = cv2.imread(os.path.join(vessel_mask_dir, or_img_fullname), 0)
            od_mask = cv2.imread(os.path.join(od_mask_dir, or_img_fullname), 0)
            tmp_label1 = cv2.bitwise_or(label.copy(), od_mask)  # od masks only
            tmp_label2 = cv2.bitwise_or(tmp_label1, vessel_mask)  # vessel and od masks

            # cv2.imshow('background', background)
            # cv2.imshow('label', tmp_label*50)
            # cv2.waitKey(200)

            H, W = background.shape[:2]
            blended_img_name = or_img_name + '_' + str(r) + '_' + extension
            if random_dens != None:
                for classes in lesion_class:
                    lesion_dens = random.randint(random_dens[classes][0], random_dens[classes][1])
                    dens_dict[classes] = lesion_dens
                '''
                if dens_dict['MA']==0: #MA为最早期出现的病变，SE为最晚出现的病变
                    dens_dict['EX']=0
                    dens_dict['HE']=0
                    dens_dict['SE']=0
                if dens_dict['EX']==0 or dens_dict['HE']==0:
                    dens_dict['SE']=0
                '''
                print('dens_dict: ', dens_dict)
                

            for classes in lesion_class:
                #print(classes)
                if classes == 'EX':
                    cr_img_list = os.listdir(EX_cr_img_dir)
                    cr_img_dir = EX_cr_img_dir
                    cr_label_dir = EX_cr_label_dir
                    dens = dens_dict['EX']
                elif classes == 'HE':
                    cr_img_list = os.listdir(HE_cr_img_dir)
                    cr_img_dir = HE_cr_img_dir
                    cr_label_dir = HE_cr_label_dir
                    dens = dens_dict['HE']
                elif classes == 'MA':
                    cr_img_list = os.listdir(MA_cr_img_dir)
                    cr_img_dir = MA_cr_img_dir
                    cr_label_dir = MA_cr_label_dir
                    dens = dens_dict['MA']
                elif classes == 'SE':
                    cr_img_list = os.listdir(SE_cr_img_dir)
                    cr_img_dir = SE_cr_img_dir
                    cr_label_dir = SE_cr_label_dir
                    dens = dens_dict['SE']
                else:
                    print('please input lesion-class with a list')
                    dens = 0
                    cr_img_list = None
                    cr_img_dir = None
                    cr_label_dir = None

                # print(cr_img_list)
                n = 0
                while n < dens:
                    n = n + 1
                    r_name = random.randint(0, len(cr_img_list) - 1)
                    cr_img = cv2.imread(os.path.join(cr_img_dir, cr_img_list[r_name]))
                    cr_label = cv2.imread(os.path.join(cr_label_dir, cr_img_list[r_name]), 0)
                    if aug:
                        trigger = random.randint(1, 1)
                        if trigger == 1:
                            rescale_rate = random.uniform(0.8, 1.2)
                            degree = random.randrange(0, 360, 90)
                            flipCode = random.randint(-1, 1)
                            cr_img, cr_label = aug_data(cr_img,
                                                        cr_label,
                                                        rescale=True,
                                                        rot=True,
                                                        fl=True,
                                                        elastic_trans=False,
                                                        rescale_rate=rescale_rate,
                                                        degree=degree,
                                                        flipCode=flipCode,
                                                        filled_color=-1)
                            '''
                            cv2.imshow('1', cr_img)
                            cv2.imshow('2', cr_label)
                            cv2.waitKey(1)
                            '''
                    h, w = cr_label.shape
                    r_h = random.randint(0 + math.ceil(h / 2), H - math.ceil(h / 2))
                    r_w = random.randint(0 + math.ceil(w / 2), W - math.ceil(w / 2))
                    # make sure that the lesion patches will not be added out of whole image.
                    center = (r_w, r_h)
                    # print(h, w)
                    if classes == 'MA':  # MA的融合位置应包含血管和视盘区域的约束条件(tmp_label2=label or vessel_mask or od_mask)
                        # tmp_label = tmp_label2                          #添加的病变间存在重叠
                        tmp_label = cv2.bitwise_or(label, tmp_label2)  # 添加的病变间不存在重叠
                    else:  # 其它病变的融合位置仅对视盘区域进行约束(tmp_label1=label or od_mask)
                        # tmp_label = tmp_label1
                        tmp_label = cv2.bitwise_or(label, tmp_label1)

                    '''
                    print(classes)
                    cv2.imshow('tmp_label', tmp_label * 50)
                    cv2.waitKey(200)
                    '''
                    if  classes == 'HE':  # sift out tiny HE

                        if np.min(background_binary[r_h - math.ceil(h / 2):r_h + math.ceil(h / 2),
                                  r_w - math.ceil(w / 2):r_w + math.ceil(w / 2)]) == 0 or \
                                np.max(cv2.bitwise_and(
                                    tmp_label[r_h - math.ceil(h / 2):r_h + int(h / 2),
                                    r_w - math.ceil(w / 2):r_w + int(w / 2)],
                                    cr_label, mask=cr_label) != 0) or cr_label.size < 2500:  #idrid 2500, DDR 600
                            n = n - 1
                        else:
                            blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                               lesion_class=classes, type=type)
                            '''
                                if classes == 'MA' or classes == 'HE':
                                    blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                                   lesion_class=classes,type = cv2.NORMAL_CLONE)
                                else:
                                    blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                                       lesion_class=classes, type=type)
                            '''

                            background, label = blended_img, blended_label

                    elif  classes == 'EX':  # sift out tiny EX

                        if np.min(background_binary[r_h - math.ceil(h / 2):r_h + math.ceil(h / 2),
                                  r_w - math.ceil(w / 2):r_w + math.ceil(w / 2)]) == 0 or \
                                np.max(cv2.bitwise_and(
                                    tmp_label[r_h - math.ceil(h / 2):r_h + int(h / 2),
                                    r_w - math.ceil(w / 2):r_w + int(w / 2)],
                                    cr_label, mask=cr_label) != 0) or cr_label.size < 1600:  #idrid 1600, DDR 400
                            n = n - 1
                        else:
                            blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                               lesion_class=classes, type=type)
                            '''
                                if classes == 'MA' or classes == 'HE':
                                    blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                                   lesion_class=classes,type = cv2.NORMAL_CLONE)
                                else:
                                    blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                                       lesion_class=classes, type=type)
                            '''

                            background, label = blended_img, blended_label

                    else:

                        if np.min(background_binary[r_h - math.ceil(h / 2):r_h + math.ceil(h / 2),
                                  r_w - math.ceil(w / 2):r_w + math.ceil(w / 2)]) == 0 or \
                                np.max(cv2.bitwise_and(
                                    tmp_label[r_h - math.ceil(h / 2):r_h + int(h / 2),
                                    r_w - math.ceil(w / 2):r_w + int(w / 2)],
                                    cr_label, mask=cr_label) != 0) or cr_label.size < 16:

                            # make sure that the lesion patches will not be added out of retina and overlapped on other lesions.
                            # print('pass')
                            n = n - 1


                        else:
                            blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                               lesion_class=classes, type=type)
                            '''
                                if classes == 'MA' or classes == 'HE':
                                    blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                                   lesion_class=classes,type = cv2.NORMAL_CLONE)
                                else:
                                    blended_img, blended_label = blend(cr_img, background, cr_label, center, label,
                                                                       lesion_class=classes, type=type)
                            '''

                            background, label = blended_img, blended_label

                        # (background[r_h, r_w])
                        # print(cr_img_list[r_name])
                        # print(center)

            cv2.imwrite(os.path.join(blended_img_dir, blended_img_name), background)
            cv2.imwrite(os.path.join(blended_label_dir, blended_img_name), label)
    elapsed = (time.time() - start)
    print("Time used:", elapsed)
    #class_count(blended_label_dir, len(lesion_class) + 1)


import argparse
if __name__ == '__main__':
    paraser = argparse.ArgumentParser()
    paraser.add_argument('--dataset', type=str, default='e_ophtha',
                         help='The name of target dataset')
    paraser.add_argument('--original_data_dir', type=str, default='..',
                         help='root directory')
    paraser.add_argument('--material_dir', type=str, default='./lesion_library_no_edge_hd',
                         help='The path of cropped lesion patches')
    paraser.add_argument('--aug_rate', type=int, default=19,
                         help='The augmentation rate of training set')
    paraser.add_argument('--dens_EX', type=str, default='0_60',
                         help='The density range of pasted HE lesion, min_max')
    paraser.add_argument('--dens_HE', type=str, default='0_0',
                         help='The density range of pasted MA lesion, min_max')
    paraser.add_argument('--dens_MA', type=str, default='0_100',
                         help='The density range of pasted EX lesion, min_max')
    paraser.add_argument('--dens_SE', type=str, default='0_0',
                         help='The density range of pasted SE lesion, min_max')

    args = paraser.parse_args()
    main(args)

    '''
    input: list of lesion-classes, list of density, using data-augmentation?, 
    path of lesion material library(Sub-folder: /image/(EX,HE,MA,SE), /label/(EX,HE,MA,SE)),
    path of original_data(Sub-folder: /image_zoom, /label_zoom)
    '''
