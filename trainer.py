
import argparse
from regressor import Regressor

def main(args):
    reg = Regressor(args)
    reg.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", default = "./Data", help = "Data Directory")
    parser.add_argument("--seed", type = int, default = 42, help = "Randomization Seed")

    parser.add_argument("--train_epochs", type = int, default = 1000, help = "Number of epochs to do training")
    parser.add_argument("--lr", type=float, default = 0.01, help = "Learning Rate")

    parser.add_argument("--val_dataset_src", default="./Data/val.npz", help="The source for creating dataset for validating predictor")
    
    main(parser.parse_args())
