# FOV with threshold-masks
from PIL import Image
import os.path
import glob
import math
import numpy as np
import cv2
from skimage import data, io


def zoom(img_path, label_path, img_savedir, label_savedir, mixlabel_savedir, size=(2752,2752), ds='IDRiD'):
    def convert(img, width=size[1], height=size[0], mode='L'):
        # new_pic=Image.open(jpgfile)
        img = Image.fromarray(img)
        h, w = img.size
        shift = int(abs(h - w) * 0.5)
        if mode == 'RGB':
            ch = (0, 0, 0)
            resize_mode = Image.ANTIALIAS
        else:
            ch = 0
            resize_mode = Image.NEAREST
        if h > w:  # 添加边框修补成正方形
            length = h
            new_pic = Image.new(mode, (length, length), color=ch)
            new_pic.paste(img, (0, shift))
        else:
            length = w
            new_pic = Image.new(mode, (length, length), color=ch)
            new_pic.paste(img, (shift, 0))

        # new_img=new_pic.resize((width,height),Image.NEAREST)
        new_img = new_pic.resize((width, height), resize_mode)
        return np.asarray(new_img)

    if ds == 'IDRiD':
        max_width = []
        max_hight = []
        if not os.path.exists(img_savedir):
            os.makedirs(img_savedir)
        if not os.path.exists(label_savedir):
            # os.makedirs(label_savedir)
            os.makedirs(label_savedir + '/EX')
            os.makedirs(label_savedir + '/HE')
            os.makedirs(label_savedir + '/MA')
            os.makedirs(label_savedir + '/SE')
        if not os.path.exists(mixlabel_savedir):
            os.makedirs(mixlabel_savedir)
        # outdir="H:/python project/lesion_dp3_keras/val/image/"
        fileList = os.listdir(img_path)
        for jpgfile in fileList:
            (realname, extension) = os.path.splitext(jpgfile)
            img = cv2.imread(os.path.join(img_path, jpgfile))
            print(realname)
            dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            label1 = cv2.imread(os.path.join(label_path, 'EX', realname + '.tif'), 0)
            label2 = cv2.imread(os.path.join(label_path, 'HE', realname + '.tif'), 0)
            label3 = cv2.imread(os.path.join(label_path, 'MA', realname + '.tif'), 0)
            label4 = cv2.imread(os.path.join(label_path, 'SE', realname + '.tif'), 0)
            label_all = np.zeros_like(label1)  # 用1-4标记四种病变
            label_all[np.where(label1 > 0)] = 1
            label_all[np.where(label2 > 0)] = 2
            label_all[np.where(label3 > 0)] = 3
            label_all[np.where(label4 > 0)] = 4

            # print(label1)

            dst = cv2.GaussianBlur(dst, (1, 1), 0)
            dst = cv2.blur(dst, (5, 5))
            ret, thresh = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)
            kernel = np.ones((9, 9), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=3)
            # cv2.imshow("d",dst)
            # cv2.waitKey(0)
            _, contours, hierarchy1 = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            print(len(contours), jpgfile)
            print(hierarchy1)
            a = []
            for i in range(len(contours)):
                a.append(len(contours[i]))
            m = max(a)

            n = a.index(m)
            rect = cv2.minAreaRect(contours[n])

            # cv2.imwrite("path1"+str(n)+".png",img)
            box = np.int0(cv2.boxPoints(rect))
            # draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 2)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            hight = y2 - y1
            width = x2 - x1

            max_hight.append(hight)
            max_width.append(width)
            print(max(max_width), max(max_hight))
            new_img = np.zeros((hight, width, 3), np.uint8)
            new_label = np.zeros((hight, width), np.uint8)
            crop_img = convert(img[y1:y1 + hight, x1:x1 + width], mode='RGB')
            # cv2.imshow('1',crop_img)
            # cv2.waitKey(0)
            crop_label1 = convert(label1[y1:y1 + hight, x1:x1 + width])
            crop_label2 = convert(label2[y1:y1 + hight, x1:x1 + width])
            crop_label3 = convert(label3[y1:y1 + hight, x1:x1 + width])
            crop_label4 = convert(label4[y1:y1 + hight, x1:x1 + width])
            crop_label_all = convert(label_all[y1:y1 + hight, x1:x1 + width])
            cv2.imwrite(os.path.join(img_savedir, realname + '.png'), crop_img)

            cv2.imwrite(os.path.join(label_savedir, 'EX', realname + '.png'), crop_label1)
            cv2.imwrite(os.path.join(label_savedir, 'HE', realname + '.png'), crop_label2)
            cv2.imwrite(os.path.join(label_savedir, 'MA', realname + '.png'), crop_label3)
            cv2.imwrite(os.path.join(label_savedir, 'SE', realname + '.png'), crop_label4)
            cv2.imwrite(os.path.join(mixlabel_savedir, realname + '.png'), crop_label_all)
    elif ds == 'e_ophtha':
        max_width = []
        max_hight = []
        if not os.path.exists(img_savedir):
            os.makedirs(img_savedir)
        if not os.path.exists(label_savedir):
            # os.makedirs(label_savedir)
            os.makedirs(label_savedir + '/EX')
            os.makedirs(label_savedir + '/MA')
        if not os.path.exists(mixlabel_savedir):
            os.makedirs(mixlabel_savedir)
        # outdir="H:/python project/lesion_dp3_keras/val/image/"
        fileList = os.listdir(img_path)
        for jpgfile in fileList:
            (realname, extension) = os.path.splitext(jpgfile)
            img = cv2.imread(os.path.join(img_path, jpgfile))
            print(realname)
            dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            label1 = cv2.imread(os.path.join(label_path, 'EX', realname + '.png'), 0)
            label2 = cv2.imread(os.path.join(label_path, 'MA', realname + '.png'), 0)
            label_all = np.zeros_like(label1)  # 用1-4标记四种病变
            label_all[np.where(label1 > 0)] = 1
            label_all[np.where(label2 > 0)] = 2
            # print(label1)

            dst = cv2.GaussianBlur(dst, (3, 3), 0)
            dst = cv2.blur(dst, (1, 1))
            ret, thresh = cv2.threshold(dst, 3, 255, cv2.THRESH_BINARY)
            # cv2.imshow("d",dst)
            # cv2.waitKey(0)
            _, contours, hierarchy1 = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            print(len(contours), jpgfile)

            a = []
            for i in range(len(contours)):
                a.append(len(contours[i]))
            m = max(a)

            n = a.index(m)
            rect = cv2.minAreaRect(contours[n])

            # cv2.imwrite("path1"+str(n)+".png",img)
            box = np.int0(cv2.boxPoints(rect))
            # draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 2)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            hight = y2 - y1
            width = x2 - x1

            max_hight.append(hight)
            max_width.append(width)
            print(max(max_width), max(max_hight))
            crop_img = convert(img[y1:y1 + hight, x1:x1 + width], mode='RGB')
            crop_label1 = convert(label1[y1:y1 + hight, x1:x1 + width])
            crop_label2 = convert(label2[y1:y1 + hight, x1:x1 + width])
            crop_label_all = convert(label_all[y1:y1 + hight, x1:x1 + width])
            cv2.imwrite(os.path.join(img_savedir, realname + '.png'), crop_img)
            cv2.imwrite(os.path.join(label_savedir, 'EX', realname + '.png'), crop_label1)
            cv2.imwrite(os.path.join(label_savedir, 'MA', realname + '.png'), crop_label2)
            cv2.imwrite(os.path.join(mixlabel_savedir, realname + '.png'), crop_label_all)
def main(args):
    dataset = args.dataset
    for f in ['train', 'test']:
        if dataset == 'IDRiD':
            img_path = os.path.join('original_data/IDRiD/image', f)
            label_path = os.path.join('original_data/IDRiD/label', f)
            if f == 'test':
                f = 'val'
            img_savedir = os.path.join('IDRiD', f, '4 classes/image_zoom_hd')
            label_savedir = os.path.join('IDRiD', f, 'label_zoom_hd')
            mixlabel_savedir = os.path.join('IDRiD', f, '4 classes/label_zoom_hd')
            zoom(img_path, label_path, img_savedir, label_savedir, mixlabel_savedir, size=(2752,2752), ds='IDRiD')
        elif dataset == 'e_ophtha':
            img_path = os.path.join('e_ophtha/image', f)
            label_path = os.path.join('e_ophtha/label', f)
            if f == 'test':
                f = 'val'
            img_savedir = os.path.join('e_ophtha', f, '2 classes/image_zoom_hd')
            label_savedir = os.path.join('e_ophtha', f, 'label_zoom_hd')
            mixlabel_savedir = os.path.join('e_ophtha', f, '2 classes/label_zoom_hd')
            zoom(img_path, label_path, img_savedir, label_savedir, mixlabel_savedir, size=(1024,1024), ds='e_ophtha')
import argparse
if __name__ == '__main__':
    paraser = argparse.ArgumentParser()
    paraser.add_argument('--dataset', type=str, default='IDRiD')
    args = paraser.parse_args()
    main(args)
