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


def main_loop(configs):
    set_seed.torch_fix_seed(seed=configs.random_seed)

    transform = image_augment.SegTransform(mode='train')
    transform_ps = image_augment.SegTransform(mode='pseudo')
    Normalize = image_augment.ImageNorm()

    data_set = dataset.SegDatasetPos(config=configs, transform=transform, transform_pseudo=transform_ps)
    train_loader = DataLoader(data_set,
                              batch_size=configs.Segmentation_BatchSize_train,
                              shuffle=True,
                              num_workers=configs.Segmentation_NumWorkers,
                              pin_memory=True,
                              drop_last=True)

    valid_dataset = dataset.SegDataset(config=configs, mode='valid', transform=transform)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=configs.Segmentation_BatchSize_valid,
                              shuffle=True,
                              num_workers=configs.Segmentation_NumWorkers,
                              pin_memory=True,
                              drop_last=False)

    test_dataset = dataset.SegDataset(config=configs, mode='test', transform=Normalize)
    test_loader = DataLoader(test_dataset,
                             batch_size=configs.Segmentation_BatchSize_test,
                             shuffle=False,
                             num_workers=configs.Segmentation_NumWorkers,
                             pin_memory=True,
                             drop_last=False)

    model = load_model.load_deplaboV3plus(configs=configs)
    model.to(configs.device)

    optimizer = Adam(model.parameters(),
                     lr=configs.Segmentation_LearningRate, betas=(0.9, 0.999),
                     eps=1e-08, weight_decay=2e-5, amsgrad=False
                     )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=configs.Segmentation_Epoch, eta_min=configs.Segmentation_LearningRate_min, last_epoch=-1
    )

    # criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_CE = loss_func.LabelNormLoss(configs, eta=configs.label_Norm_eta)

    criterion_KL = loss_func.ClassKL()
    # criterion_KL = loss_func.CosSimilarityLoss()

    valid_best_loss = 100
    valid_best_mIoU = 0
    for epoch in range(configs.Segmentation_Epoch):
        configs.now_epoch = epoch+1
        print(f"{epoch+1} / {configs.Segmentation_Epoch}")

        # ====== train ======
        if epoch < configs.Segmentation_Normal_Train:
            print("通常のセグメンテーション")
            logs = train_core.train_normal(configs, model, train_loader, optimizer, criterion_CE)
        elif configs.Segmentation_Normal_Train <= epoch < configs.Segmentation_Self_pseudo_Train:
            print("通常のセグメンテーション　＋　自己学習")
            logs = train_core.train_pseudo(configs, model, train_loader, optimizer, criterion_CE)
        elif configs.Segmentation_Self_pseudo_Train <= configs.Segmentation_Self_pseudo_class_Train:
            print("通常のセグメンテーション　＋　自己学習 + 疑似ラベルの面積比")
            logs = train_core.train_pseudo_dataset_ckl(configs, model, train_loader, optimizer, criterion_CE, criterion_KL)
        # ==================

        # === validation ===
        valid_loss, valid_miou = train_core.validation(configs, model, valid_loader, criterion_CE)

        if valid_best_loss > valid_loss:
            torch.save(
                model.state_dict(),
                os.path.join(configs.Homepath, "models",
                             configs.save_folder_name + '_loss.pth')
            )
            wandb.save(os.path.join(os.path.join(configs.Homepath, "models",
                             configs.save_folder_name + '_loss.pth')))
            if configs.wandb_logging:
                wandb.run.summary["best_valid_loss_epoch"] = epoch
                wandb.run.summary["best_valid_loss"] = valid_loss
            valid_best_loss = valid_loss

        if valid_best_mIoU < valid_miou:
            torch.save(
                model.state_dict(),
                os.path.join(configs.Homepath, "models",
                             configs.save_folder_name + '_mIoU.pth')
            )
            wandb.save(os.path.join(configs.Homepath, "models",
                             configs.save_folder_name + '_mIoU.pth'))
            if configs.wandb_logging:
                wandb.run.summary["best_valid_mIoU_epoch"] = epoch
                wandb.run.summary["best_valid_mIoU"] = valid_miou
            valid_best_mIoU = valid_miou

        if configs.wandb_logging:
            wandb_log = {**logs, **{
                "valid loss": valid_loss,
                "valid mIoU": valid_miou,
                "LR": scheduler.get_last_lr()[0]
            }}
            wandb.log(wandb_log)
        # ==================
        scheduler.step()

    # ==== test =====
    print("てすと　もーど")
    for best_model in ['mIoU', 'loss']:
        model_name = configs.save_folder_name + f'_{best_model}'
        model_path = os.path.join(configs.Homepath, "models", model_name+'.pth')
        model.load_state_dict(torch.load(model_path))
        log_path = os.path.join(configs.Log_Folder, model_name)
        os.makedirs(log_path, exist_ok=True)
        loss_mean, iou_mean, iou_obj = train_core.test(configs, model, test_loader, criterion_CE)
        iou_obj.save_csv_data(os.path.join(log_path, f"iou_fold_{configs.Segmentation_Fold}_{best_model}.csv"))
        iou_obj.save_all_dict(os.path.join(log_path, f"iou_fold_{configs.Segmentation_Fold}_{best_model}.yaml"))
        if configs.wandb_logging:
            wandb.log({
                f"test mIoU ={best_model}=": iou_mean,
            })
            wandb.save(os.path.join(log_path, f"iou_fold_{configs.Segmentation_Fold}_{best_model}.csv"))
            wandb.save(os.path.join(log_path, f"iou_fold_{configs.Segmentation_Fold}_{best_model}.yaml"))

    # ===============


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--Segmentation_Fold', type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument('-n', '--Name', type=str, default=None)
    parser.add_argument('-e', '--Segmentation_Epoch', type=int, default=None)
    args = parser.parse_args()

    CFG = cfg()
    if args.Segmentation_Epoch != None:
        CFG.Segmentation_Epoch = args.Segmentation_Epoch
    if args.Segmentation_Fold != None:
        CFG.Segmentation_Fold = args.Segmentation_Fold
        CFG.save_folder_name += "_Fold_"+str(args.Segmentation_Fold)
    if args.Name != None:
        CFG.Name = args.Name
    return CFG


if __name__ == '__main__':
    CFG = get_args()
    if CFG.wandb_logging:
        run = wandb.init(
            project="{Project Name}",       # ← here
            config={k: v for k, v in dict(vars(CFG)).items() if '__' not in k},
            name=f"{CFG.Name}",             # ← here
            entity="{User Name}"            # ← here
        )
        wandb.save(os.path.join(CFG.Homepath, 'src', 'train_core.py'))
    main_loop(CFG)

    if CFG.wandb_logging:
        run.finish()
