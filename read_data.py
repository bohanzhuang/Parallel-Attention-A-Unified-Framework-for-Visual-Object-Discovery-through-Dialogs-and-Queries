import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import os
from torch.autograd import Variable
import h5py
from utils import sen_to_fea


def feature_loader(path):
    img_data = h5py.File(path, 'r')
    return img_data['img_data'][:].astype('float32')


class MyDataset(data.dataset.Dataset):
    
    def __init__(self, data_source, feature_dir, patch_dir, transform=None):
    	
    	self.data_source = data_source
        self.feature_dir = feature_dir
        self.patch_dir = patch_dir    

    def __getitem__(self, idx):

        img_name = self.data_source[idx]['image_name']
        num_patches = len(self.data_source[idx]['category'])
        proposal_feature_container = torch.rand(20, 512, 7, 7).float()
        img_feature = torch.from_numpy(feature_loader(os.path.join(self.feature_dir, img_name[:-4] + '.h5'))).float()
        proposal_features = torch.from_numpy(feature_loader(os.path.join(self.patch_dir, '%06d' % (idx) + '.h5')))
        proposal_feature_container[0:num_patches,:,:,:] = proposal_features
        
        return img_feature, proposal_feature_container, idx


    def __len__(self):
        return len(self.data_source)

