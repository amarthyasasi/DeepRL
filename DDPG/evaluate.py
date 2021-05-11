import sys, os
import argparse
import torch
from models.DQN import DQN
from dataset.RLdataset import *
from sensors.heat_sensor import HeatSensor
from agents.drone import Drone
import os, hydra, logging, glob
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from torch.utils.data import DataLoader
import torch.nn as nn
from trainer import *


if __name__=="__main__":

    ################################# Argument Parser #####################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help = "path to model", required=True)
    args = parser.parse_args()
    
    # Initialize model
    model = AgentTrainer.load_from_checkpoint(args.model).cuda()
    model.eval()

    # Play trajectory
    model.playTrajectory()

        