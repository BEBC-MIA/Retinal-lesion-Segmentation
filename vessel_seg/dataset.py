import torch.utils.data as data
import PIL.Image as Image
import os
import torchvision.transforms as transforms
import torch
"""
def make_dataset(root):
    imgs = []
    image_dir = os.path.join(root,'image')
    mask_dir = os.path.join(root,'label')
    n = len(os.listdir(image_dir)) 
    img_list = os.listdir(image_dir)
    for name in img_list:
        img = os.path.join(image_dir, name)
        mask = os.path.join(mask_dir, name.split('.')[0]+'_mask.png')
        #img = os.path.join(image_dir, "%03d.png" % i)
        #mask = os.path.join(mask_dir, "%03d_mask.png" % i)
        
    
        imgs.append((img, mask))
    return imgs
"""
def make_dataset(root):
    imgs = []
    print(root)
    image_dir = os.path.join(root,'image_zoom_hd')
    mask_dir = os.path.join(root,'label_zoom_hd')
    n = len(os.listdir(image_dir)) 
    img_list = os.listdir(image_dir)
    for name in img_list:
        img = os.path.join(image_dir, name)
        mask = os.path.join(mask_dir, name)
        #img = os.path.join(image_dir, "%03d.png" % i)
        #mask = os.path.join(mask_dir, "%03d_mask.png" % i)
        imgs.append((img, mask))
    return imgs

class imageDataset(data.Dataset):           #pytorch 读取数据的标准格式
    def __init__(self, root, transform=None, target_transform=None,classes=1,one_hot='n'):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.one_hot = one_hot
        self.classes = classes

    def __getitem__(self, index):
        #print(index)
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)  #二分类label：0-255
            
            #img_y = self.target_transform(img_y)*255      #如果label为多分类，且已经转化成数值标签（0，1，2，3）totensor后应x255还原原数值大小。#pytorch多分类label不需转化成one-hot编码格式！
        if self.one_hot=='y':
            #img_y = img_y         
            #print('img_y',img_y.shape)
            img_y_onehot = torch.zeros((self.classes, img_y.shape[1], img_y.shape[2]))
            img_y_onehot.scatter_(0, img_y.long(), 1)  #(1,h,w)-->(classes,h,w
            img_y = img_y_onehot
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
    
def transform_data(root=None,classes=1,one_hot="n",target_size=(512,512)):   
    # 把多个步骤整合到一起, channel=（channel-mean）/std, 因为是分别对三个通道处理
    x_transforms = transforms.Compose([       #transforms.Compose()串联多个transform操作
        transforms.Resize(target_size,2),
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        
    ])
    # mask只需要转换为tensor
    print('one_hot:',one_hot)
    y_transforms = transforms.Compose([transforms.Resize(target_size,3),transforms.ToTensor()])
    # 参数解析器,用来解析从终端读取的命令
    dataset = imageDataset(root, transform=x_transforms, target_transform=y_transforms,classes=classes,one_hot=one_hot)
    return dataset