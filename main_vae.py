import torch
from torch import nn
from VAE.model import VAE
from Dataset.transformation import JointTransform

from train import train
from Dataset.dataloader import get_dataLoader
from utils.visual_analysis import visual_analysis
from config import root_dir, mask_dir, num_epochs, batch_size, embedding_dim, learning_rate, l2, gamma, milestones

import warnings
warnings.filterwarnings("ignore")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    # Add arguments for each hyperparameter
    parser.add_argument('--path', type=str, default=root_dir, help="Path to the root of the dataset")
    parser.add_argument('--mask_dir', type=str, default=mask_dir, help="Path to the root of the masked version of the dataset")

    parser.add_argument('--shape', type=int, default=256, help="Shape of the image") 
    parser.add_argument('--num_workers', type=int, default=4, help="Shape of the image") 


    parser.add_argument('--embedding_dim', type=int, default=embedding_dim, help="Embedding dimension")
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help="Learning rate")

    parser.add_argument('--num_epochs', type=int, default=num_epochs, help="Number of epochs")
    parser.add_argument('--l2', type=float, default=l2, help="L2 regularization")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="Batch size")
    parser.add_argument('--gamma', type=float, default=gamma, help="Gamma")
    parser.add_argument('--optim', type=str, default="Adam", help="Optimizer")

    # Boolean flags
    parser.add_argument('--use_scheduler', action='store_true', help="Use scheduler")

    return parser.parse_args()

args = parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

joint_transform = JointTransform(resize=(218, 178), horizontal_flip_prob=0.5, rotation_range=(-30, 30))
train_loader, test_loader = get_dataLoader(args, joint_transform, joint_transform)

model = VAE(input_channels=3, latent_dim=args.embedding_dim).to(device)
model = nn.DataParallel(model)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, gamma=args.gamma, milestones=milestones)

train(model, optimizer, scheduler, train_loader, test_loader, device, args)
visual_analysis(model, test_loader, device)