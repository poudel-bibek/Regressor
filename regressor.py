import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from resnet50 import ResNet
from resnet50 import ResNet50

from Utils.regressor_utils  import DriveDataset
from Utils.regressor_utils  import prepare_data
# from Utils.regressor_utils import Normalize
#from torch.optim.lr_scheduler import MultiStepLR # This is not reaally required for Adam

class Regressor:
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter()
        print("\n--------------------------------")
        print("Seed: ", self.args.seed)

        # Set seeds
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        
        ## Identify device and acknowledge
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device Assigned to: ", self.device)

        # Constants + Hyperparams : Not set by terminal args
        self.TRAIN_BATCH_SIZE = 128
        self.VAL_BATCH_SIZE = 128

        print("Data Directory: ", self.args.train_data_dir)
        train_images, train_targets, val_images, val_targets = prepare_data(self.args.train_data_dir)
        print("\nLoaded:\nTraining: {} Images, {} Targets\nValidation: {} Images, {} Targets".format(train_images.shape[0],
                                                                                                    train_targets.shape[0],
                                                                                                    val_images.shape[0],
                                                                                                    val_targets.shape[0]))
        print("Each image has shape:{}".format(train_images.shape[-3:]))                                                                                       
        self.train_dataset = DriveDataset(train_images, train_targets)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                    batch_size=self.TRAIN_BATCH_SIZE,
                                                    collate_fn=None,
                                                    shuffle=False)

        self.val_dataset = DriveDataset(val_images, val_targets)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                    batch_size=self.VAL_BATCH_SIZE,
                                                    collate_fn=None,
                                                    shuffle=False)

        """
        Normalize inside the model: Later when the RL agents wants decisions on images
        We dont have to worry about it, the saved model has it
        """
        self.cnn_model = ResNet50()
        self.net = self.cnn_model.to(self.device)

        # self.net = nn.Sequential(
        #             Normalize([87.3387902, 95.2747195, 107.87840699],
        #             [59.92092777, 65.32244491, 76.10364479],
        #             self.device),
        #             self.cnn_model).to(self.device)
        
        print("\n--------------------------------")
        print("Total No. of Trainable Parameters: ",sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        self.criterion = nn.MSELoss() # reduction = "mean"
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.args.lr)

    def train(self):
        # Train & Val
        # Val set to every train epoch
        train_loss_collector = np.zeros(self.args.train_epochs)
        val_loss_collector = np.zeros(self.args.train_epochs)

        start = time.time()
        best_loss = float('inf')
        print("\n#### Started Training ####")
        for i in range(self.args.train_epochs):
            #since we have dropout
            self.net.train()
        
            batch_loss = 0
            print("Epoch :",i, end="\t")
            for bi, data in enumerate(self.train_dataloader):
                #print(data[0].shape)
                inputs_batch, targets_batch = data 
                # Load data to device (may require type conversion on both or single)
                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)
                #targets_batch = targets_batch.view(targets_batch.size(0), -1).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs_batch)
                loss = self.criterion(torch.squeeze(outputs), targets_batch)

                loss.backward()
                self.optimizer.step()

                loss_np = loss.cpu().detach().numpy() #This gives a single number
                self.writer.add_scalar("Regressor: Batch Loss/train", loss_np, bi) #train is recorded per batch
                batch_loss +=loss_np
                
            # Average Batch train loss per epoch (length of trainloader = number of batches?)
            avg_batch_loss_train = round(batch_loss / len(self.train_dataloader),3)
            print("Average Batch loss: Training {}".format(avg_batch_loss_train), end="\t")

            # Val loss per epoch
            avg_batch_loss_val = self.validate(self.net)
            val_loss_collector[i] = avg_batch_loss_val

            print("Validation: {}".format(avg_batch_loss_val))
            # If val loss smaller, save model
            if avg_batch_loss_val < best_loss:
                best_loss = avg_batch_loss_val
                # Save the entire model
                print("#### New Model Saved #####")
                # This is terminal dependent (Change it to os, listdir (relative path))
                # When accessed from NFQ main is ../Saved_models, whereas from regressor file its ./saved_models
                torch.save(self.net, './Saved_models/regressor.pt')

            train_loss_collector[i] = avg_batch_loss_train
            

        print("#### Ended Training ####")
        elapsed = (time.time() - start)
        print("Training Stats: \nTime: {} seconds".format(round(elapsed,2)))
        self.writer.flush() 
        self.writer.close()
        # Draw loss plot (both train and val)
        fig, ax = plt.subplots(figsize=(16,5), dpi = 100)
        xticks= np.arange(0,self.args.train_epochs,5)
        ax.set_ylabel("MSE Loss (Training & Validation)") 
        ax.plot(np.array(train_loss_collector))
        ax.plot(np.array(val_loss_collector))
        ax.set_xticks(xticks) #;
        ax.legend(["Validation", "Training"])
        fig.savefig('./Regressor/training_result.png')

    def validate(self, current_model):
        self.net.eval()  
        batch_val_loss=0
        with torch.no_grad():
            for bi, data in enumerate(self.val_dataloader):
                #print(data[0].shape)
                inputs_batch, targets_batch = data 
                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)
                outputs = current_model(inputs_batch)
                loss = self.criterion(torch.squeeze(outputs), targets_batch)
                self.writer.add_scalar("Regressor: Batch Loss/val", loss_np, bi)
                loss_np =  loss.cpu().detach().numpy()
                #loss_np = loss.item()
                batch_val_loss +=loss_np
        avg_batch_loss_val = round(batch_val_loss / len(self.val_dataloader),3)
        return avg_batch_loss_val


# Example use case
# 1. 
#regressor_1 = Regressor(args)
#regressor_1.train()