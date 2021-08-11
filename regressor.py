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
        self.val_dataset = DriveDataset(val_images, val_targets)

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

        train_loss_collector = np.zeros(self.args.train_epochs)
        val_loss_collector = np.zeros(self.args.train_epochs)

        best_loss = float('inf')
        print("\n#### Started Training ####")
        for i in range(self.args.train_epochs):
            self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                    batch_size=self.TRAIN_BATCH_SIZE,
                                                    collate_fn=None,
                                                    shuffle=True)

            self.val_dataloader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                    batch_size=self.VAL_BATCH_SIZE,
                                                    collate_fn=None,
                                                    shuffle=True)

            self.net.train()
            start = time.time()

            batch_loss_train = 0
            print("Ep. {}/{}:".format(i+1, self.args.train_epochs), end="\t")

            ground_truths_train =[]
            predictions_train =[]

            for bi, data in enumerate(self.train_dataloader):

                inputs_batch, targets_batch = data 
                ground_truths_train.extend(targets_batch.numpy())

                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs_batch)

                predictions_train.extend(outputs.cpu().detach().numpy())
                loss = self.criterion(torch.squeeze(outputs), targets_batch)

                loss.backward()
                self.optimizer.step()

                loss_np = loss.cpu().detach().numpy() #This gives a single number
                self.writer.add_scalar("Regressor: Batch Loss/train", loss_np, bi) #train is recorded per batch
                batch_loss_train +=loss_np
                
            # Average Batch train loss per epoch (length of trainloader = number of batches?)
            avg_batch_loss_train = round(batch_loss_train / len(self.train_dataloader),3)

            # Get train accuracy here
            acc_train = self.mean_accuracy(ground_truths_train, predictions_train)
            print("Train: ABL {}, Acc. {}%".format(avg_batch_loss_train, round(acc_train,2) ), end="\t")

            # Val loss per epoch
            acc_val, avg_batch_loss_val = self.validate(self.net)
            val_loss_collector[i] = avg_batch_loss_val

            print("Val: ABL {}, Acc. {}%".format(avg_batch_loss_val, round(acc_val,2) ), end = "\t")
            print("Time: {} s".format(round(time.time() - start, 1)))

            if avg_batch_loss_val < best_loss:

                best_loss = avg_batch_loss_val
                print("#### New Model Saved #####")
                torch.save(self.net, './Saved_models/regressor.pt')

            train_loss_collector[i] = avg_batch_loss_train
            
        self.writer.flush() 
        self.writer.close()

        # Draw loss plot (both train and val)
        fig, ax = plt.subplots(figsize=(16,5), dpi = 100)
        xticks= np.arange(0,self.args.train_epochs,5)

        ax.set_ylabel("MSE Loss (Training & Validation)") 
        ax.plot(np.asarray(train_loss_collector))
        ax.plot(np.asarray(val_loss_collector))

        ax.set_xticks(xticks) #;
        ax.legend(["Validation", "Training"])
        fig.savefig('./Regressor/training_result.png')

        print("#### Ended Training ####")

        # Plot AMA as well

    def validate(self, current_model):

        current_model.eval()  
        batch_loss_val=0

        ground_truths_val =[]
        predictions_val =[]

        with torch.no_grad():
            for bi, data in enumerate(self.val_dataloader):

                inputs_batch, targets_batch = data 
                ground_truths_val.extend(targets_batch.numpy())

                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)

                outputs = current_model(inputs_batch)
                predictions_val.extend(outputs.cpu().detach().numpy())

                loss = self.criterion(torch.squeeze(outputs), targets_batch)
                loss_np =  loss.cpu().detach().numpy()

                self.writer.add_scalar("Regressor: Batch Loss/val", loss_np, bi)
                batch_loss_val +=loss_np


        acc_val = self.mean_accuracy(ground_truths_val, predictions_val)
        avg_batch_loss_val = round(batch_loss_val / len(self.val_dataloader),3)

        return acc_val, avg_batch_loss_val

    def mean_accuracy(self, ground_truths, predictions):

        ground_truths = np.asarray(ground_truths)
        predictions = np.asarray(predictions).reshape(-1)
        error = 15.0*(ground_truths - predictions) # Accuracy is measured in steering angles

        # print("GTS,", ground_truths.shape)
        # print("PREDS,", predictions.shape)
        # print("ER", error.shape)

        # Error in 1.5,3,7,15,30,75
        er_1 = np.asarray([ 1.0 if er <=1.5 else 0.0 for er in error]) 
        er_2 = np.asarray([ 1.0 if er <=3.0 else 0.0 for er in error])
        er_3 = np.asarray([ 1.0 if er <=7.5 else 0.0 for er in error]) 
        er_4 = np.asarray([ 1.0 if er <=15.0 else 0.0 for er in error]) 
        er_5 = np.asarray([ 1.0 if er <=30.0 else 0.0 for er in error])  
        er_6 = np.asarray([ 1.0 if er <=75.0 else 0.0 for er in error]) 

        # count each and Mean
        acc_1 = np.sum(er_1)/error.shape[0]
        acc_2 = np.sum(er_2)/error.shape[0]
        acc_3 = np.sum(er_3)/error.shape[0]
        acc_4 = np.sum(er_4)/error.shape[0]
        acc_5 = np.sum(er_5)/error.shape[0]
        acc_6 = np.sum(er_6)/error.shape[0]

        mean_acc = (acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6)/6
        return 100*mean_acc # In percentage