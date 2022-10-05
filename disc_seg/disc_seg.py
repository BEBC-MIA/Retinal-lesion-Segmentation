from skimage import exposure
import cv2
import numpy as np
import os
import argparse
'''
img: 原始眼底图像
vessel_mask: 对应血管掩膜(0/255)
size: 检测窗大小（略大于视盘直径），需根据具体数据集的的视盘尺寸设置
step: 检测窗滑动步长
'''
def find_roi(img,
             vessel_mask,
             EX_mask = None,
             SE_mask = None,
             size = 160,
             step = 10,
             img_size=(1024,1024)):
    # Use a breakpoint in the code line below to debug your script.
    if img.shape[-1] == 3:
        h, w = img.shape[:2]
    else:
        h, w = img.shape
    gray_list = []
    center_list = []
    tmp_vessel_mask = vessel_mask
    tmp_vessel_mask[np.where(vessel_mask == 255)] = 127  # d127=b0111111
    #cv2.imshow('vessel_mask', tmp_vessel_mask)
    img_tmp = img
    if EX_mask.size != 0:
        img_tmp = cv2.bitwise_and(img_tmp, EX_mask)
    if SE_mask.size != 0:
        img_tmp = cv2.bitwise_and(img_tmp, SE_mask)
    img_tmp = cv2.medianBlur(img_tmp, 13)
    img_tmp = cv2.bitwise_or(img_tmp, tmp_vessel_mask) #if: x>127, x bit_or 127 = 255 else: x bit_or 127 = 127
    img_tmp = cv2.resize(img_tmp, (w//2, h//2), interpolation=0)
    img_tmp = cv2.resize(img_tmp, (w, h), interpolation=0)


    #cv2.imshow('img_tmp', img_tmp)
    for i in range(size//2, h, step):
        for j in range(size//2, w, step):
            crop = img_tmp[i-size//2:i+size//2,j-size//2:j+size//2]
            new_crop = np.zeros_like(crop)
            circle_tmp = cv2.circle(new_crop, (size//2,size//2), size//2, 255, -1)
            #_, circle_tmp = cv2.threshold(circle_tmp, 0, 255, cv2.THRESH_BINARY)
            crop = cv2.bitwise_and(crop, circle_tmp)

            #cv2.imshow('crop', crop)
            #cv2.waitKey(1)
            m_gray = np.mean(crop)
            gray_list.append(m_gray)
            center_list.append((i, j))

    center = center_list[gray_list.index(max(gray_list))]  #OD中心点坐标
    crop = img[center[0] - size//2:center[0] + size//2, center[1] - size//2:center[1] + size//2]
    vessel_crop_mask = vessel_mask[center[0] - size//2:center[0] + size//2, center[1] - size//2:center[1] + size//2]
    print(center)
    '''
    crop = cv2.blur(crop,(40,40))
    _, crop = cv2.threshold(crop, 140, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    crop = cv2.dilate(crop,kernel,iterations=4)
    '''
    od_mask = np.zeros((h,w), dtype="uint8")
    od_mask = cv2.circle(od_mask, (center[1],center[0]), size//2, 255, -1)


    #crop = cv2.bitwise_or(crop, vessel_crop_mask)
    #cv2.imshow('crop', crop)
    #cv2.imshow('od_mask', od_mask)
    #cv2.waitKey(600)
    return od_mask

def main(args):
    dataset = args.dataset
    if not args.train_path:
        if dataset == 'IDRiD':
            train_path = dataset + '/train/4 classes'
        if dataset == 'e_ophtha':
            train_path = dataset + '/train/2 classes'
    else:
        train_path = args.train_path
    img_path = train_path + '/image_zoom_hd'  # 原始图像路径
    vessel_mask_path = train_path +'/vessel_mask_zoom_hd'  # 对应血管掩膜路径
    lesion_mask_path = train_path +'/label_zoom_hd'  # 病变label路径（如果有的话）
    od_mask_path = train_path + '/od_mask_zoom_hd'  # od mask输出路径
    if dataset == 'IDRiD':
        od_size = 180  # e_ophtha为160
    else:
        od_size = 160

    step = 10  # 移动步长
    process_size = (1024, 1024)

    if not os.path.exists(od_mask_path):
        os.makedirs(od_mask_path)
    img_list = os.listdir(img_path)
    print(img_list)
    for name in img_list:
        img = cv2.imread(os.path.join(img_path, name), 0)
        h, w = img.shape[:2]
        img = cv2.resize(img, process_size)
        # img_g = cv2.GaussianBlur(img, (process_size[0]+1,process_size[1]+1), 0)
        # img -= img_g.astype(np.uint8)
        vessel_mask = cv2.imread(os.path.join(vessel_mask_path, name), 0)
        target_size = img.shape[:2]
        vessel_mask = cv2.resize(vessel_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        lesion_mask = cv2.resize(cv2.imread(os.path.join(lesion_mask_path, name), 0), (target_size[1], target_size[0]),
                                 interpolation=cv2.INTER_NEAREST)
        if lesion_mask.size != 0:
            EX_mask = np.zeros_like(lesion_mask)
            EX_mask[np.where(lesion_mask) == 1] = 255
            _, EX_mask = cv2.threshold(EX_mask, 0, 255, cv2.THRESH_BINARY_INV)
            SE_mask = np.zeros_like(lesion_mask)
            SE_mask[np.where(lesion_mask) == 4] = 255
            _, SE_mask = cv2.threshold(SE_mask, 0, 255, cv2.THRESH_BINARY_INV)
        od_mask = cv2.resize(find_roi(img, vessel_mask,
                                      EX_mask=EX_mask, SE_mask=SE_mask,
                                      size=od_size, step=step), (w, h), interpolation=cv2.INTER_NEAREST)

        # vs = cv2.subtract(img, img_g)
        # print(np.min(vs))
        img = cv2.resize(img, (w, h))
        # vs = cv2.bitwise_or(img, od_mask)
        cv2.imwrite(os.path.join(od_mask_path, name), od_mask)
        # cv2.imwrite(os.path.join(od_mask_path, 'vs_'+name), vs)

if __name__ == '__main__':
    paraser = argparse.ArgumentParser()
    paraser.add_argument('--dataset', type=str, default='IDRiD')
    paraser.add_argument('--train_path', type=str, default='../IDRiD/train/4 classes', help='path of training set')
    args = paraser.parse_args()
    main(args)

