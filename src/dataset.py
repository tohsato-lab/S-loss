
import os
import pandas as pd
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.transforms.functional as F
import torchvision.transforms as T


class SegDatasetPos(torch.utils.data.Dataset):
    def __init__(self, config, transform, transform_pseudo):
        self.config = config
        self.transform = transform
        self.transform_pseudo = transform_pseudo
        fold_id = self.config.Segmentation_Fold

        self.data_table = self.config.label_table
        self.name_list = self.data_table.loc[(self.data_table['folds'] > 0)
                                                 & (self.data_table['folds'] != fold_id)][['Name', 'NormPos']]

        self.data_table_pseudo = self.config.image_table
        self.name_list_pseudo = self.data_table[['Name', 'NormPos']]

        # print(self.image_name_list)
    def get_image(self, name):
        image_path = os.path.join(self.config.image_folder, name)
        image = read_image(image_path) / 255
        return image

    def get_label(self, name):
        label_path = os.path.join(self.config.label_folder, name)
        label = read_image(label_path)
        label = F.resize(img=label, size=(512, 512), interpolation=T.InterpolationMode.NEAREST)
        label = label.to(torch.float32)
        return label

    def __getitem__(self, idx):
        data = self.name_list.iloc[idx]
        name = data['Name']
        train_pos = data['NormPos']

        # ポジティブなペアの選択
        pseudo_Norm_datas = self.name_list_pseudo.loc[
            (self.name_list_pseudo['NormPos'] < train_pos + self.config.Segmentation_Self_pseudo_pos_Pos) &
            (train_pos - self.config.Segmentation_Self_pseudo_pos_Pos < self.name_list_pseudo['NormPos'])
        ]['Name'].to_numpy()
        name_pseudo_Pos = np.random.choice(pseudo_Norm_datas)

        # ネガティブなペアの選択
        pseudo_Norm_datas = self.name_list_pseudo.loc[
            (self.name_list_pseudo['NormPos'] > train_pos + self.config.Segmentation_Self_pseudo_pos_Neg) |
            (train_pos - self.config.Segmentation_Self_pseudo_pos_Neg > self.name_list_pseudo['NormPos'])
        ]['Name'].to_numpy()
        name_pseudo_Neg = np.random.choice(pseudo_Norm_datas)

        # 通常のソース画像の取り出し
        image_train = self.get_image(name)
        label_train = self.get_label(name)
        if self.transform is not None:
            image_train, label_train = self.transform(image_train, label_train)
        if isinstance(image_train, torch.Tensor):
            image_train = torch.squeeze(image_train)
        if isinstance(label_train, torch.Tensor):
            label_train = torch.squeeze(label_train)

        # ポジティブなターゲット画像の取り出し
        image_pseudo_Pos = self.get_image(name_pseudo_Pos)
        label_pseudo_Pos = 0
        if self.transform_pseudo is not None:
            image_pseudo_Pos, label_pseudo = self.transform_pseudo(image_pseudo_Pos, label_pseudo_Pos)
        if isinstance(image_pseudo_Pos, torch.Tensor):
            image_pseudo_Pos = torch.squeeze(image_pseudo_Pos)
        if isinstance(label_pseudo_Pos, torch.Tensor):
            label_pseudo_Pos = torch.squeeze(label_pseudo_Pos)

        # ネガティブなターゲット画像の取り出し
        image_pseudo_Neg = self.get_image(name_pseudo_Neg)
        label_pseudo_Neg = 0
        if self.transform_pseudo is not None:
            image_pseudo_Neg, label_pseudo = self.transform_pseudo(image_pseudo_Neg, label_pseudo_Neg)
        if isinstance(image_pseudo_Neg, torch.Tensor):
            image_pseudo_Neg = torch.squeeze(image_pseudo_Neg)
        if isinstance(label_pseudo_Neg, torch.Tensor):
            label_pseudo_Neg = torch.squeeze(label_pseudo_Neg)

        return (image_train, label_train, name), (image_pseudo_Pos, label_pseudo_Pos, name_pseudo_Pos), (image_pseudo_Neg, label_pseudo_Neg, name_pseudo_Neg)

    def __len__(self):
        return len(self.name_list)

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode, transform):
        self.config = config
        self.mode = mode
        self.transform = transform
        fold_id = self.config.Segmentation_Fold
        if self.mode == 'train':
            self.data_table = self.config.label_table
            self.name_list = self.data_table.loc[(self.data_table['folds'] > 0)
                                                 & (self.data_table['folds'] != fold_id)]['Name'].to_list()
        elif self.mode == 'pseudo':
            self.data_table = self.config.image_table
            self.name_list = self.data_table['Name'].to_list()
        elif self.mode == 'valid':
            self.data_table = self.config.label_table
            self.name_list = self.data_table.loc[(self.data_table['folds'] == fold_id)]['Name'].to_list()
        elif self.mode == 'test':
            self.data_table = self.config.label_table
            self.name_list = self.data_table['Name'].to_list()
        else:
            self.name_list = None

        # print(self.image_name_list)
    def get_image(self, name):
        image_path = os.path.join(self.config.image_folder, name)
        image = read_image(image_path) / 255
        return image

    def get_label(self, name):
        if self.mode in ['pred', 'pseudo']:
            label = 0.
        elif self.mode == 'test':
            label_path = os.path.join(self.config.label_folder, name)
            label = read_image(label_path)
            label = F.resize(img=label, size=(512, 512), interpolation=T.InterpolationMode.NEAREST)
            label = label.to(torch.float32)
        else:
            label_path = os.path.join(self.config.label_folder, name)
            label = read_image(label_path)
            label = F.resize(img=label, size=(512, 512), interpolation=T.InterpolationMode.NEAREST)
            label = label.to(torch.float32)
        return label

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image = self.get_image(name)
        label = self.get_label(name)

        if self.transform is not None:
            image, label = self.transform(image, label)

        if isinstance(image, torch.Tensor):
            image = torch.squeeze(image)
        if isinstance(label, torch.Tensor):
            label = torch.squeeze(label)
        return image, label, name

    def __len__(self):
        return len(self.name_list)


class AllImageDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform):
        self.config = config
        self.transform = transform
        self.data_table = self.config.image_table
        self.name_list = self.data_table['Name'].to_list()

    def get_image(self, name):
        image_path = os.path.join("./inputs/images/", name)
        image = read_image(image_path) / 255
        return image
    def __getitem__(self, item):
        image_name = self.name_list[item]
        image = self.get_image(image_name)
        if self.transform is not None:
            image = self.transform(image)
        image = torch.squeeze(image)
        return image, image_name
    def __len__(self):
        return len(self.name_list)


if __name__ == '__main__':
    from work import configs
    from src import set_seed
    from src import image_augment
    set_seed.torch_fix_seed(1)
    config = configs.Configs(mode='test')
    transform = image_augment.SegTransform(mode='train')
    transform_pseudo = image_augment.SegTransform(mode='pseudo')
    data_set = SegDatasetPos(config=config, transform=transform, transform_pseudo=transform_pseudo)
    train_loader = DataLoader(data_set,
                              batch_size=config.Segmentation_BatchSize_train,
                              shuffle=True,
                              num_workers=config.Segmentation_NumWorkers,
                              pin_memory=True,
                              drop_last=True)
    for i in train_loader:
        train, pseudo = i
        print(train[0].shape)
        print(train[1].shape)
        print(train[2])
        break