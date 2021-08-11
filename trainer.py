
import argparse
from regressor import Regressor

def main(args):
    reg = Regressor(args)
    reg.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", default = "./Data", help = "Data Directory")
    parser.add_argument("--seed", type = int, default = 42, help = "Randomization Seed")

    parser.add_argument("--train_epochs", type = int, default = 200, help = "Number of epochs to do training")
    parser.add_argument("--lr", type=float, default = 0.01, help = "Learning Rate")

    parser.add_argument("--ld_method", default="vae", help = "LDR method")
    parser.add_argument("--load_ldr", action="store_true", help = "Whether to load pre-run LD images")
    parser.add_argument("--ldr_img_src", default="./Data/train.npz", help = "If converting images to ldr, what is the source")

    parser.add_argument("--env_type", default="nearest_neighbor", help = "What environment to use")
    parser.add_argument("--train_regressor", action="store_true", help = "Whether to train a new regressor or load a trained one")

    parser.add_argument("--val_dataset_src", default="./Data/val.npz", help="The source for creating dataset for validating predictor")
    
    main(parser.parse_args())
