import os
import argparse

import wandb

import torch.nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam

from configs import Configs as cfg
from src import set_seed
from src import dataset
from src import image_augment
from src.models import load_model
from src import loss_func

from src import train_core


def prediction(configs):
    os.makedirs(os.path.join(configs.Homepath, 'data', 'Prediction'), exist_ok=True)
    os.makedirs(os.path.join(configs.Homepath, 'data', 'Prediction_view'), exist_ok=True)
    set_seed.torch_fix_seed(seed=configs.random_seed)
    Normalize = image_augment.ImageNorm()

    test_dataset = dataset.SegDataset(config=configs, mode='pseudo', transform=Normalize)
    test_loader = DataLoader(test_dataset,
                             batch_size=configs.Segmentation_BatchSize_test,
                             shuffle=False,
                             num_workers=configs.Segmentation_NumWorkers,
                             pin_memory=True,
                             drop_last=False)


    model = load_model.load_deplaboV3plus(configs=configs)
    model_name = f'model_weight_name'                       # ←here
    model_path = os.path.join(configs.Homepath, "models", model_name + '.pth')
    model.load_state_dict(torch.load(model_path))
    model.to(configs.device)

    none = train_core.prediction(configs, model, test_loader)


def main():
    CFG = cfg()
    prediction(CFG)


if __name__ == '__main__':
    main()
