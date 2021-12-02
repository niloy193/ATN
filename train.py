#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
import numpy as np
import cv2
from torch.utils import data
from torchvision import transforms, utils
import shutil

torch.manual_seed(1338)
np.random.seed(1338)
import sys
sys.path.append('/home/agency/xai/TIP')


# In[ ]:


def train_set(source_dir, tar_dir, patch_length):
    im_num = 0
    for imgs in os.listdir(source_dir):
        if imgs.endswith('.jpg') or imgs.endswith('.png') or imgs.endswith('.tif'):
            img = cv2.imread(os.path.join(source_dir,imgs))
            w_rem = img.shape[1]%patch_length
            h_rem = img.shape[0]%patch_length
            w_tar = img.shape[1] - w_rem
            h_tar = img.shape[0] - h_rem
            img = cv2.resize(img, (w_tar, h_tar))
            img_with_border = cv2.copyMakeBorder(img, top = patch_length, bottom = patch_length, left = patch_length, right = patch_length, borderType = cv2.BORDER_CONSTANT, value=0)
            grid_img = view_as_blocks(img_with_border, (patch_length,patch_length,3))
            grid_img = np.squeeze(grid_img, axis = 2)
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)
            for i in range(grid_img.shape[0]):
                for j in range(grid_img.shape[1]):
                    cv2.imwrite(os.path.join(tar_dir,str(im_num)+'_'+str(i)+'_'+str(j)+'.png'), grid_img[i][j])
            im_num = im_num + 1

def train_gt(source_dir, tar_dir, patch_length):
    im_num = 0
    count = 0
    for imgs in os.listdir(source_dir):
        if imgs.endswith('.jpg') or imgs.endswith('.png') or imgs.endswith('.tif'):
            img = cv2.imread(os.path.join(source_dir,imgs), 0)
            w_rem = img.shape[1]%patch_length
            h_rem = img.shape[0]%patch_length
            w_tar = img.shape[1] - w_rem
            h_tar = img.shape[0] - h_rem
            img = cv2.resize(img, (w_tar, h_tar), interpolation = cv2.INTER_NEAREST)
            grid_img = view_as_blocks(img, (patch_length,patch_length))
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir) 
            for i in range(0,grid_img.shape[0]):
                for j in range(0,grid_img.shape[1]):
                    cv2.imwrite(os.path.join(tar_dir,str(im_num)+'_'+str(i+1)+'_'+str(j+1)+'.png'), grid_img[i][j]) 
                    count = count + 1
            im_num = im_num + 1


# In[ ]:


# source_dir = "/home/agency/xai/TIP/dataset/images"
# tar_dir = "/home/agency/xai/TIP/dataset/train"
# train_set(source_dir, tar_dir, 256)

# source_dir = "/home/agency/xai/TIP/dataset/masks"
# tar_dir = "/home/agency/xai/TIP/dataset/train_gt"
# train_gt(source_dir, tar_dir, 256)

# source_dir = "/home/agency/xai/TIP/dataset/masks_test"
# tar_dir = "/home/agency/xai/TIP/dataset/test_gt"
# train_gt(source_dir, tar_dir, 256)

# source_dir = "/home/agency/xai/TIP/dataset/images_test"
# tar_dir = "/home/agency/xai/TIP/dataset/test"
# train_set(source_dir, tar_dir, 256)


# In[ ]:


def get_files_names(gt_dir):
    files = os.listdir(gt_dir)
    final_files = {}
    i = 0
    for file in files:
        if file.endswith('.png'):
            final_files[i] = file
            i = i +1
    return final_files    

def get_num(ID):
    nums = []
    temp = ''
    for i in range (len(ID)):
        if ID[i] == '_' or ID[i] == '.':
            nums.append(int(temp))
            temp = ''
        else:
            temp = temp + ID[i]
    return nums     

def get_neighbor_filenames(source_dir, ID):
    nums = get_num(ID)
    index1 = nums[0]
    index2 = nums[1]
    index3 = nums[2]
    neighbor_filenames = {}
    
    neighbor_filenames[0] = os.path.join(source_dir,str(index1) + '_' + str(index2-1) + '_' + str(index3-1) + '.png')
    neighbor_filenames[1] = os.path.join(source_dir,str(index1) + '_' + str(index2-1) + '_' + str(index3) + '.png')
    neighbor_filenames[2] = os.path.join(source_dir,str(index1) + '_' + str(index2-1) + '_' + str(index3+1) + '.png')
    neighbor_filenames[3] = os.path.join(source_dir,str(index1) + '_' + str(index2) + '_' + str(index3-1) + '.png')
    neighbor_filenames[4] = os.path.join(source_dir,str(index1) + '_' + str(index2) + '_' + str(index3) + '.png')
    neighbor_filenames[5] = os.path.join(source_dir,str(index1) + '_' + str(index2) + '_' + str(index3+1) + '.png')
    neighbor_filenames[6] = os.path.join(source_dir,str(index1) + '_' + str(index2+1) + '_' + str(index3-1) + '.png')
    neighbor_filenames[7] = os.path.join(source_dir,str(index1) + '_' + str(index2+1) + '_' + str(index3) + '.png')
    neighbor_filenames[8] = os.path.join(source_dir,str(index1) + '_' + str(index2+1) + '_' + str(index3+1) + '.png')
    return neighbor_filenames
    


# In[ ]:


def load_image_I(source, shape):
    H = shape[0]
    W = shape[1]
    img = cv2.imread(source,1)
    img = cv2.resize(img, (H,W))
    img = np.float32(img)
    img=img/255.0
    img = np.moveaxis(img, 2, 0)
    img = torch.from_numpy(img)
    return img

def load_image_gt(source, shape):
    H = shape[0]
    W = shape[1]
    img = cv2.imread(source,0)
    img = np.float32(img)
    img = cv2.resize(img, (H,W), interpolation=cv2.INTER_NEAREST)
    img = torch.from_numpy(img)
    return img 

def load_neighbors(source, shape):
    H = shape[0]
    W = shape[1]
    imgs = torch.empty((0,3,H,W))
    for i in range(len(source)):
        img = cv2.imread(source[i],1)
        img = cv2.resize(img, (H,W))
        img = np.float32(img)
        img=img/255.0
        img = np.moveaxis(img, 2, 0)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0)
        imgs = torch.cat([imgs, img], dim=0)
    return imgs   


# In[ ]:


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, train_source, gt_source, file_names):
        'Initialization'
        self.gt_path = gt_source
        self.train_path = train_source
        self.file_names = file_names

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        I_file = os.path.join(self.train_path, self.file_names[index])
        neighbor_files = get_neighbor_filenames(self.train_path, self.file_names[index])
        gt_file = os.path.join(self.gt_path, self.file_names[index])
        # Load data and get label
        
        I = load_image_I(I_file, (256,256))
        neighbors = load_neighbors(neighbor_files, (256,256))
        gt = load_image_gt(gt_file, (256,256))
        
        
        
        return I, neighbors, gt


# In[ ]:


def cal_weights(I, neighbors):
    Ba,B1,C,H,W = I.size()  #Batch, 1,256,14,14
    Ba,B2,C,H,W = neighbors.size() #Batch, 8,256,14,14


    neighbors_flat = neighbors.view(-1, B2, C,W*H)
    I_flat = I.view(-1, B1,C,W*H)
    
    I_T = torch.transpose(I_flat,2,3)
    neighbors_T = torch.transpose(neighbors_flat,2,3)

    weights = torch.matmul(I_T,neighbors_flat)
#     weights = F.sigmoid(weights)
    weights = F.softmax(weights, dim=-1)
    weights_T = torch.transpose(weights,2,3)
    weighted_neighbourhood = torch.matmul(neighbors_flat,weights_T)
    weighted_neighbourhood = weighted_neighbourhood.view(Ba,B2,C,H,W)
    return weighted_neighbourhood


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=5,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.alpha1 = nn.Parameter(torch.ones(size=(1,)))
        self.alpha2 = nn.Parameter(torch.zeros(size=(64,64)))
        if freeze_bn:
            self.freeze_bn()

    def deep_encoder(self,input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        return x , low_level_feat
    
    def deep_decoder(self,x, low_level_feat, size,weights_x=None, weights_llf= None):
        if weights_x != None:
#             weights_x = self.alpha1*weights_x
            weights_x = weights_x.sum(dim=1)
            x = (self.alpha1*x) + (self.alpha2*weights_x)
            
        if weights_llf != None:   
#             weights_llf = self.alpha2*weights_llf
            weights_llf = weights_llf.sum(dim=1)
            low_level_feat = (self.alpha1*low_level_feat) + (self.alpha2*weights_llf)
            
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x
        

    def forward(self, input):
        x, llf = self.deep_encoder(input)
        x = self.deep_decoder(x,llf)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


# In[ ]:


batch_size = 2
params_train = {'batch_size': batch_size,
          'shuffle': False,
        'drop_last': True}
params_test = {'batch_size': batch_size,
          'shuffle': False,
        'drop_last': True}

source_dir = '/home/agency/xai/TIP/dataset/train/'
tar_dir = '/home/agency/xai/TIP/dataset/train_gt/'
files = get_files_names(tar_dir) #FILE NAMES MUST BE FOUND THROUGH THE GROUNDTRUTH DIRECTORY
train_set = Dataset(source_dir, tar_dir, files)
select_train_set = np.random.choice(len(train_set), len(train_set)//6, False)
# train_subset = torch.utils.data.Subset(train_set, select_train_set)
training_generator = data.DataLoader(train_set, **params_train)

source_test_dir = '/home/agency/xai/TIP/dataset/test/'
target_test_dir = '/home/agency/xai/TIP/dataset/test_gt/'
files_test = get_files_names(target_test_dir)
test_set = Dataset(source_test_dir, target_test_dir, files_test)
select_test_set = np.random.choice(len(test_set), len(test_set)//6, False)
# test_subset = torch.utils.data.Subset(test_set, select_test_set)
test_generator = data.DataLoader(test_set, **params_test)


# In[ ]:


# I1 = torch.randn((1,2,4,4))
# N1 = torch.randn((2,2,4,4))
# first_weights = cal_weights2(I1, N1)
# I2 = torch.randn((1,2,4,4))
# N2 = torch.randn((2,2,4,4))
# second_weights = cal_weights2(I2, N2)

# I = torch.cat([I1, I2], dim=0)
# I = torch.unsqueeze(I, dim=1)
# N = torch.cat([N1, N2], dim=0)
# N = N.view(2,2,2,4,4)
# final_weights = cal_weights(I,N)


# In[ ]:


def one_hot(label, n_class, device):
    B,H,W = label.shape
    encoded = torch.zeros(size=(B,n_class,H,W)) #6 is the number of classes, background, aeroplane and bicycle
    encoded = encoded.to(device)
    encoded = encoded.scatter_(1, label.unsqueeze(1), 1)
    return encoded

def iou_cal(enc1, enc2):
    #enc1 and enc2 both should be B*C*H*W shaped
    B,C,H,W = enc1.shape
    
    enc1 = enc1.int()
    enc2 = enc2.int()

    enc3 = enc1*enc2
    intersection = enc3.sum(dim=(2,3))
    intersection = intersection.float()
    
    enc3 = enc1 | enc2
    union = enc3.sum(dim=(2,3))
    union = union.float()
    
    return intersection, union


# In[ ]:


def train_heart(I, N, G):
    _,_,_,H,W = N.shape
    N = N.view(-1,3,H,W)
    with torch.no_grad():
        x_n, l_n = model.deep_encoder(N)
    x_i, l_i = model.deep_encoder(I)

    _, C_x, H_x, W_x = x_n.shape
    x_n = x_n.view(-1,9,C_x,H_x,W_x)
    _, C_n, H_n, W_n = l_n.shape
    l_n = l_n.view(-1,9,C_n,H_n,W_n)

    x_i = torch.unsqueeze(x_i, dim=1)
    l_i = torch.unsqueeze(l_i, dim=1)

#     weights1 = cal_weights(x_i, x_n)
    weights2 = cal_weights(l_i, l_n)

    x_i = torch.squeeze(x_i, dim=1)
    l_i = torch.squeeze(l_i, dim=1)
    out = model.deep_decoder(x_i, l_i, (H,W), None, weights2)
    return out


# In[ ]:


with open('results_train.txt','w'): 
    pass


# In[ ]:


import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from multiprocessing import Pool
from tqdm import tqdm
from IPython.display import clear_output

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model = DeepLab()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    total_inter = 0
    total_union = 0
    for I, N, G in tqdm(training_generator):
        optimizer.zero_grad()
        I, N, G = I.to(device), N.to(device), G.to(device)
        out = train_heart(I,N,G)
        loss = criterion(out,G.long())
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        for I, N, G in tqdm(test_generator):
            I, N, G = I.to(device), N.to(device), G.to(device)
            out = train_heart(I,N,G)
            G = one_hot(G.long(), 5, device)
            _, out = torch.max(out,1)
            out = one_hot(out.long(), 5, device)
            intersection, union = iou_cal(out,G)
            total_inter = total_inter + intersection
            total_union = total_union + union
        IOU = total_inter.sum(dim=0)/total_union.sum(dim=0)
        IOU = IOU.detach().cpu().numpy()
        print("EPOCH:", epoch+1)
        print("IOU:", IOU)
        mIOU = IOU.sum()/5
        print("mIOU:", mIOU)
        
        f = open("results_train.txt", "a")
        f.write(str(epoch+1)+" "+str(IOU)+" "+ str(mIOU)
                + ",\n")
        f.close()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'train.pth.tar')
        
    
#duita matrix niye batch chara alada cal_weight kore ar duita matrix eki sathe batch e niye kore dekbo same ashe kina


# In[ ]:


# checkpoint = torch.load('checkpoint.pth.tar')
# start_epoch = checkpoint['epoch']
# scheduler = checkpoint['scheduler']
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# for state in optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.to(device)


# In[ ]:


# list(model.named_parameters())[1]


# In[ ]:


# a =  list(model.named_parameters())[1][1]
# a = torch.relu(a)
# a = a.detach().cpu().numpy()
# a = a/a.max()
# a = a*255
# a = cv2.resize(a,(256,256))
# cv2.imwrite("/home/agency/xai/TIP/saliency3.png", a)


# In[ ]:




