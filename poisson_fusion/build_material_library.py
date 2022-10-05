import cv2
import numpy as np
import os
import argparse
import sys
def crop(img, label, fullname, img_save_dir, label_save_dir, dataset='IDRiD'):
    def smooth_boundary(cr_img):
        #img_list = os.listdir(cr_img_dir)
        bk = cr_img
        bk_gray = cv2.cvtColor(cr_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(bk_gray, 20, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.erode(thresh, kernel, iterations=2)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        mean = np.mean(bk[np.where(np.sum(bk, axis=2) > 30)], axis=0).astype(np.int)
        # calculate mean rgb level of every cropped lesion images
        thresh_bgr[np.where(thresh == 0)] = mean
        sum_thresh_bgr = np.sum(thresh_bgr, axis=2)
        bk[np.where(sum_thresh_bgr == np.sum(mean))] = mean
        return bk

    h, w = label.shape
    _, thresh = cv2.threshold(label, 0, 255, cv2.THRESH_BINARY)
    if dataset == 'IDRiD':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))       #IDRiD=9
    if dataset == 'e_ophtha':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    #remove holes within lesion, and make sure cropped images have unbroken features
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    new_label = np.zeros((h, w, 1), np.uint8)
    new_label.fill(255)
    j = 0
    for i in range(len(contours)):
        print(hierarchy[0,i,3])
        area = cv2.contourArea(contours[i])
        if area > 1 and hierarchy[0,i,3] == -1:   #是否为lesion外轮廓（一级轮廓）
            rect = cv2.minAreaRect(contours[i])
            cv2.drawContours(new_label, contours[i], -1, (0, 0, 0), 1)
            box = np.int0(cv2.boxPoints(rect))
            # draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 2)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            hight = y2 - y1
            width = x2 - x1
            print(y1, y2, x1, x2)
            crop_label = label[y1:y1 + hight, x1:x1 + width]
            crop_img = img[y1:y1 + hight, x1:x1 + width]  # crop lesion region
            #crop_img = cv2.bitwise_and(crop_img, crop_img, mask=crop_label)   # delete background of lesion region
            name, extension = os.path.splitext(fullname)
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            if not os.path.exists(label_save_dir):
                os.makedirs(label_save_dir)
            if np.sum(crop_label) != 0 :
                bk_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                _, img_thresh = cv2.threshold(bk_gray, 20, 255, cv2.THRESH_BINARY)  # remove bloom on the edge
                if np.where(img_thresh == 0)[0].size == 0 and np.where(img_thresh == 0)[1].size == 0:  # make sure that the cropped patches belong to retina area.
                    #crop_img = smooth_boundary(crop_img)
                    cv2.imwrite(os.path.join(img_save_dir, name+'_'+str(j)+extension), crop_img)
                    cv2.imwrite(os.path.join(label_save_dir, name+'_'+str(j)+extension), crop_label)
                    j = j+1



def main(args):
    dataset = args.dataset
    material_dir = os.path.join(args.material_dir, dataset)  # 裁剪病变保存文件夹
    original_data_dir = os.path.join(args.original_data_dir, dataset+'/train')  # 原始数据，包括原始rgb图像和label
    if dataset == 'IDRiD':
        lesion_class = ['EX', 'HE', 'MA', 'SE']
    if dataset == 'e_ophtha':
        lesion_class = ['EX', 'MA']
    if len(lesion_class)==4:
        img_dir = os.path.join(original_data_dir, '4 classes/image_zoom_hd')  # whole RGB images
        label_dir = os.path.join(original_data_dir, 'label_zoom_hd')  # label for whole images
        EX_or_label_dir = os.path.join(label_dir, 'EX')  # label for 4 kinds of lesions
        HE_or_label_dir = os.path.join(label_dir, 'HE')
        MA_or_label_dir = os.path.join(label_dir, 'MA')
        SE_or_label_dir = os.path.join(label_dir, 'SE')
        or_label_dir = (EX_or_label_dir, HE_or_label_dir, MA_or_label_dir, SE_or_label_dir)

        EX_cr_img_dir = os.path.join(material_dir, 'image/EX')  # cropped RGB images
        HE_cr_img_dir = os.path.join(material_dir, 'image/HE')
        MA_cr_img_dir = os.path.join(material_dir, 'image/MA')
        SE_cr_img_dir = os.path.join(material_dir, 'image/SE')
        EX_cr_label_dir = os.path.join(material_dir, 'label/EX')  # cropped label for 4 kinds of lesions
        HE_cr_label_dir = os.path.join(material_dir, 'label/HE')
        MA_cr_label_dir = os.path.join(material_dir, 'label/MA')
        SE_cr_label_dir = os.path.join(material_dir, 'label/SE')
        crop_img_dir = (EX_cr_img_dir, HE_cr_img_dir, MA_cr_img_dir, SE_cr_img_dir)
        crop_label_dir = (EX_cr_label_dir, HE_cr_label_dir, MA_cr_label_dir, SE_cr_label_dir)

    elif len(lesion_class)==2:
        img_dir = os.path.join(original_data_dir, '2 classes/image_zoom_hd')  # whole RGB images
        label_dir = os.path.join(original_data_dir, 'label_zoom_hd')  # label for whole images
        EX_or_label_dir = os.path.join(label_dir, 'EX')  # label for 2 kinds of lesions
        MA_or_label_dir = os.path.join(label_dir, 'MA')
        or_label_dir = (EX_or_label_dir, MA_or_label_dir)
        EX_cr_img_dir = os.path.join(material_dir, 'image/EX')  # cropped RGB images
        MA_cr_img_dir = os.path.join(material_dir, 'image/MA')
        EX_cr_label_dir = os.path.join(material_dir, 'label/EX')  # cropped label for 2 kinds of lesions
        MA_cr_label_dir = os.path.join(material_dir, 'label/MA')
        crop_img_dir = (EX_cr_img_dir, MA_cr_img_dir)
        crop_label_dir = (EX_cr_label_dir, MA_cr_label_dir)

    imglist = os.listdir(img_dir)
    for img_name in imglist:
        '''
        EX_or_label = cv2.imread(os.path.join(or_label_dir[0], img_name))
        HE_or_label = cv2.imread(os.path.join(or_label_dir[1], img_name))
        MA_or_label = cv2.imread(os.path.join(or_label_dir[2], img_name))
        SE_or_label = cv2.imread(os.path.join(or_label_dir[3], img_name))
        '''

        img = cv2.imread(os.path.join(img_dir, img_name))
        for i in range(len(lesion_class)):
            print(i)
            label = cv2.imread(os.path.join(or_label_dir[i], img_name), 0)
            crop(img, label, img_name, crop_img_dir[i], crop_label_dir[i], dataset=dataset)



            #smooth_boundary(crop_img_dir[i])
        # cv2.imshow('original image', EX_or_label)
        # cv2.waitKey(1)
        #crop_img(img_dir, label_dir, or_label_dir, crop_img_dir, crop_label_dir)



if __name__ == '__main__':
    paraser = argparse.ArgumentParser()
    paraser.add_argument('--dataset', type=str, default='IDRiD')
    paraser.add_argument('--original_data_dir', type=str, default='..')
    paraser.add_argument('--material_dir', type=str, default='./lesion_library_no_edge_hd')
    args = paraser.parse_args()
    main(args)

