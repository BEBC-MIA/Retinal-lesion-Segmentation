import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchsummary import summary
import sys
sys.path.append('vessel_seg')
from unet import Unet
from lossfunction import *
from loss import DiceLoss
from dataset import transform_data
import cv2
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

# 是否使用cuda

"""
# 把多个步骤整合到一起, channel=（channel-mean）/std, 因为是分别对三个通道处理
x_transforms = transforms.Compose([       #transforms.Compose()串联多个transform操作
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()
# 参数解析器,用来解析从终端读取的命令
"""
parse = argparse.ArgumentParser()
device = torch.device("cuda")


def train_model(model, criterion, optimizer, dataload, writer, classes=1, epoch=0):
    dt_size = len(dataload.dataset)
    epoch_loss = 0
    step = 0
    for x, y in tqdm(dataload, ncols=60):
        # for x, y in dataload:
        step += 1
        inputs = x.to(device)
        labels = y.to(device)
        # print(labels.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        # print(outputs)
        if classes == 1:
            outputs = nn.Sigmoid()(outputs)
            loss = criterion(outputs, labels)
        else:
            # labels = labels.squeeze(1)  #使用内置的交叉熵函数时需要压缩维度，且不需要softmax
            outputs = nn.Softmax(dim=1)(outputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
    mean_epoch_loss = epoch_loss / ((dt_size - 1) // dataload.batch_size + 1)
    print("epoch %d mean_train_loss:%0.3f" % (epoch, mean_epoch_loss))
    writer.add_scalar('train_loss', mean_epoch_loss, global_step=epoch)
    return model


def val_model(model, criterion, dataload, writer, classes=1, epoch=0):
    model.eval()
    with torch.no_grad():  # 不进行梯度计算和反向传播
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in tqdm(dataload, ncols=60):
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # forward
            outputs = model(inputs)
            # print(outputs)
            if classes == 1:
                outputs = nn.Sigmoid()(outputs)
                loss = criterion(outputs, labels)
            else:
                # labels = labels.squeeze(1)  #使用内置的交叉熵函数时需要压缩维度，且不需要softmax
                outputs = nn.Softmax(dim=1)(outputs)
                loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            # print("%d/%d,val_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        mean_epoch_loss = epoch_loss / ((dt_size - 1) // dataload.batch_size + 1)
        print("epoch %d mean_val_loss:%0.3f" % (epoch, mean_epoch_loss))
    writer.add_scalar('val_loss', mean_epoch_loss, global_step=epoch)
    return mean_epoch_loss


# 训练模型
def train(batch_size=1,
          classes=2,
          one_hot='n',
          max_epoch=60,
          target_size=(512, 512),
          train_path=None,
          val_path=None,
          ckp=None,
          log_path=None):
    writer = SummaryWriter(log_path)
    model = Unet(3, classes).to(device)
    summary(model, (3,) + target_size)

    if classes == 1:
        # criterion = nn.BCELoss()
        criterion = DiceLoss()
    else:
        # criterion = nn.CrossEntropyLoss()  #pytorch CEloss 内置softmax函数，因此network不需要添加softmax！
        criterion = MulticlassDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    dataset_train = transform_data(root=train_path,
                                   classes=classes,
                                   one_hot=one_hot,
                                   target_size=target_size)
    dataloaders_train = DataLoader(dataset_train,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)
    dataset_val = transform_data(root=val_path,
                                 classes=classes,
                                 one_hot=one_hot,
                                 target_size=target_size)
    dataloaders_val = DataLoader(dataset_val,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4)

    min_epoch_loss = 1e5
    loss_list = []
    for epoch in range(max_epoch):
        print('Epoch {}/{}'.format(epoch, max_epoch - 1))
        print('-' * 10)
        train_model(model,
                    criterion,
                    optimizer,
                    dataloaders_train,
                    writer,
                    classes=classes,
                    epoch=epoch)
        epoch_loss = val_model(model,
                               criterion,
                               dataloaders_val,
                               writer,
                               classes=classes,
                               epoch=epoch)

        if epoch == 0:
            print('saving model...')
            torch.save(model.state_dict(),
                       os.path.join(ckp, 'weights_initial.pth'))
            min_epoch_loss = epoch_loss
            loss_list.append(min_epoch_loss)
        else:
            min_epoch_loss = min(loss_list)
            if epoch_loss <= min_epoch_loss:
                print(
                    'mean_val_loss = {} <= min_mean_val_loss = {}, saving model...'.format(epoch_loss, min_epoch_loss))
                torch.save(model.state_dict(),
                           os.path.join(ckp, 'weights_best.pth'))
            loss_list.append(epoch_loss)

    writer.close()


# 显示模型的输出结果
def test(batch_size=2,
         ckp=None,
         classes=1,
         one_hot='n',
         target_size=(512, 512),
         save_size = (2752,2752),
         val_path=None,
         img_folder=None,
         save_path=None):
    print('testing...')
    device = torch.device("cuda")
    model = Unet(3, classes).to(device)
    model.load_state_dict(torch.load(os.path.join(ckp, 'weights_best.pth'), map_location='cuda'))
    dataset = transform_data(root=val_path,
                             classes=classes,
                             one_hot=one_hot,
                             target_size=target_size)
    dataload = DataLoader(dataset, 1)
    model.eval()
    summary(model, (3,) + target_size)

    i = 0
    n = 0
    with torch.no_grad():  # 不进行梯度计算和反向传播
        dt_size = len(dataload.dataset)
        step = 0
        for x, y in tqdm(dataload, ncols=60):
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            if classes == 1:
                outputs = nn.Sigmoid()(outputs)
                criterion = nn.BCELoss()
                criterion = DiceLoss()
                loss = criterion(outputs, labels)
            else:
                # labels = labels.squeeze(1)  #使用内置的交叉熵函数时需要压缩维度，且不需要softmax
                outputs = nn.Softmax(dim=1)(outputs)
                # criterion = nn.CrossEntropyLoss()  #pytorch CEloss 内置softmax函数，因此network不需要添加softmax！
                criterion = MulticlassDiceLoss()
                loss = criterion(outputs, labels)
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
            img_y = outputs.cpu().numpy()  # cuda输出的tensor无法直接进行numpy操作，因此需要转换成cpu tensor
            # img_y.save(os.path.join("C:/Users/whl/python project/vessel_seg/data/val_resault","%03d.png" % i))
            img_list = os.listdir(os.path.join(val_path, img_folder))
            for n in range(classes):
                real_batch_size = img_y.shape[0]  # 图像数量可能不能被batch_size整除，需要以实际输入的batch_size为准
                for b in range(real_batch_size):
                    # cv2.imwrite(os.path.join("C:/Users/whl/python project/vessel_seg/data/val_resault","%03d.png" % i),img_y[0,:,:]*255)

                    img_y[b, n, :, :][np.where(img_y[b, n, :, :] < 0.4)] = 0
                    img_y[b, n, :, :][np.where(img_y[b, n, :, :] >= 0.4)] = 255
                    cv2.imwrite(os.path.join(save_path,
                                             img_list[i + b]), cv2.resize(img_y[b, n, :, :] * 255, save_size, interpolation=cv2.INTER_NEAREST))  # 可视化

            i = i + real_batch_size

            # plt.imshow(img_y)
            # plt.pause(0.1)
        # plt.show()


'''
parse = argparse.ArgumentParser()
parse.add_argument("--action", type=str, help="train or test",default='train')
parse.add_argument("--batch_size", type=int, default=1)
parse.add_argument("--one_hot", type=str, default='y')
parse.add_argument("--epoch", type=int, default=20)
parse.add_argument("--target_size", type=tuple, default=(512,512))
parse.add_argument("--ckp", type=str, help="the path of model weight file", default="C:/Users/whl/python project/vessel_seg/weights_18.pth")
parse.add_argument("--classes", type=int, default=2)
args = parse.parse_args()
'''


# train
# train()

# test()
# args.ckp = "C:/Users/whl/python project/vessel_seg/weights_19.pth"
def main(args):
    batch_size = args.vs_batch_size
    classes = 1
    one_hot = 'no'
    target_size = (args.vs_target_size, args.vs_target_size)
    if args.dataset == 'IDRiD':
        save_size = (2752, 2752)
    else:
        save_size = (1024, 1024)
    max_epoch = args.vs_max_epoch
    step = args.vs_step
    root = args.vs_root
    train_path = root + '/train'
    val_path = root + '/val'  # train and val in vessel seg dataset
    img_folder = 'image_zoom_hd'  # the name of image folder
    test_path = args.train_path  # test in DR seg dataset
    save_path = args.train_path+'/vessel_mask_zoom_hd'  # save path of the testing results
    log_path = root + '/tmp/ves_seg_unet_512'
    ckp_path = root + '/results' # the path of the trained model


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    elif not os.path.exists(log_path):
        os.makedirs(log_path)
    elif not os.path.exists(ckp_path):
        os.makedirs(ckp_path)
    if step == 'test':
        test(batch_size=batch_size,
             ckp=ckp_path,
             classes=classes,
             one_hot=one_hot,
             target_size=target_size,
             save_size=save_size,
             val_path=test_path,
             img_folder=img_folder,
             save_path=save_path)
    elif step == 'train':
        train(batch_size=batch_size,
              classes=classes,
              one_hot=one_hot,
              max_epoch=max_epoch,
              target_size=target_size,
              train_path=train_path,
              val_path=val_path,
              log_path=log_path,
              ckp=ckp_path)
    elif step == 'all':
        train(batch_size=batch_size,
              classes=classes,
              one_hot=one_hot,
              max_epoch=max_epoch,
              target_size=target_size,
              train_path=train_path,
              val_path=val_path,
              log_path=log_path,
              ckp=ckp_path)
        test(batch_size=batch_size,
             ckp=ckp_path,
             classes=classes,
             one_hot=one_hot,
             target_size=target_size,
             save_size=save_size,
             val_path=test_path,
             img_folder=img_folder,
             save_path=save_path)

import argparse
if __name__ == '__main__':
    paraser = argparse.ArgumentParser()
    paraser.add_argument('--dataset', type=str, default='IDRiD')
    paraser.add_argument('--vs_step', type=str, default='test', help='train, test, all')
    paraser.add_argument('--vs_batch_size', type=int, default=2, help='batch size for training and testing')
    paraser.add_argument('--vs_target_size', type=int, default=512, help='image size for training and testing')
    #paraser.add_argument('--vs_save_size', type=int, default=2752, help='image size for saving')
    paraser.add_argument('--vs_max_epoch', type=int, default=200, help='max epoch')
    paraser.add_argument('--vs_root', type=str, default='.', help='root directory of vessel seg')
    paraser.add_argument('--train_path', type=str, default='../IDRiD/train/4 classes', help='path of training set')
    args = paraser.parse_args()



    '''
    paraser = argparse.ArgumentParser()
    paraser.add_argument('--dataset', type=str, default='IDRiD')
    paraser.add_argument('--vs_step', type=str, default='test', help='train, test, all')
    paraser.add_argument('--vs_batch_size', type=int, default=2, help='batch size for training and testing')
    paraser.add_argument('--vs_target_size', type=int, default=512, help='image size for training and testing')
    paraser.add_argument('--vs_save_size', type=int, default=1024, help='image size for saving')
    paraser.add_argument('--vs_max_epoch', type=int, default=200, help='max epoch')
    paraser.add_argument('--vs_root', type=str, default='.', help='root directory of vessel seg')
    paraser.add_argument('--train_path', type=str, default='../e_ophtha/train/2 classes', help='path of training set')
    args = paraser.parse_args()
    '''
    main(args)
