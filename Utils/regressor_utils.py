import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images, targets):
        self.images_list = images
        self.target_list = targets
        assert (len(self.images_list) == len(self.target_list))
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, key):
        image_idx = self.images_list[key]
        target_idx = self.target_list[key]
        # Correct datatype here
        return [image_idx.astype(np.float32), target_idx.astype(np.float32)]
    
def prepare_data(directory):
    train_path = directory + "/train.npz"
    val_path = directory + "/val.npz"

    print("Loading data....................")
    train = np.load(train_path)
    val = np.load(val_path)
    print(train['train_input_images'].shape)

    return train['train_input_images'], train['train_target_angles'], val['val_input_images'], val['val_target_angles'] 

    #one = train['train_images'][0:int(train['train_images'].shape[0]*0.8)]
    #two = train['train_targets'][0:int(train['train_images'].shape[0]*0.8)]
    #three = train['train_images'][int(train['train_images'].shape[0]*0.8):train['train_images'].shape[0]]
    #four = train['train_targets'][int(train['train_images'].shape[0]*0.8):train['train_images'].shape[0]]
    #print(one.shape)
    #print(two.shape)
    #print(three.shape)
    #print(four.shape)
    #return one ,two , three, four