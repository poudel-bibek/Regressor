import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
#from torchvision.transforms.transforms import ToTensor 

class DriveDataset(Dataset):
    def __init__(self, images, targets, transformation):
        self.images_list = images
        self.target_list = targets
        self.transform = transformation
        assert (len(self.images_list) == len(self.target_list))
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, key):
        image_idx = self.transform(self.images_list[key])
        target_idx = self.target_list[key]
        # Correct datatype here
        return [image_idx.astype(np.float32), target_idx.astype(np.float32)]
 
class Normalize(nn.Module):
    # Use GPU for normalization (as suggested by NVIDIA paper)
    def __init__(self, mean, std, device):
        super(Normalize, self).__init__()
        self.device = device
        self.norm = transforms.Compose([
            #transforms.ToTensor(), 
            # This makes each batch a tensor --> We get that from dataloader
            # This also divides by 255.0 --> Need to do this manually, or maybe not do it at all
            # Also does H x W x C to C x H X W (see docs) --> we did this manually during data preparation
            transforms.Normalize(mean = mean, std = std)])

    def forward(self, x):
        x = x/255.0 # This works if used
        x = self.norm(x)
        return x
    
def prepare_data(directory):
    train_path = directory + "/new_train.npz"
    val_path = directory + "/val.npz"
    #val_path = directory + "/train.npz"
    print("Loading data....................")
    train = np.load(train_path)
    val = np.load(val_path)
    print(train['train_images'].shape)
    return train['train_images'], train['train_targets'], val['val_images'], val['val_targets'] 

    #one = train['train_images'][0:int(train['train_images'].shape[0]*0.8)]
    #two = train['train_targets'][0:int(train['train_images'].shape[0]*0.8)]
    #three = train['train_images'][int(train['train_images'].shape[0]*0.8):train['train_images'].shape[0]]
    #four = train['train_targets'][int(train['train_images'].shape[0]*0.8):train['train_images'].shape[0]]
    #print(one.shape)
    #print(two.shape)
    #print(three.shape)
    #print(four.shape)
    #return one ,two , three, four