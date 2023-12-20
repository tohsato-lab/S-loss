import torch
import os
import pandas as pd

class Configs:
    def __init__(self, mode=None):
        #  ===== labels information ======
        self.mode = mode
        self.N_classes = 9
        self.mouse_area_data = {
            "BG": dict(id=0, color=[0, 0, 0]),
            "Isocortex":dict(id=1, color=[0, 0, 255]),
            "Olfactory area":dict(id=2, color=[0, 255, 0]),
            "Hippocampal formation": dict(id=3, color=[255, 0, 0]),
            "Cerebral nuclei": dict(id=4, color=[0, 255, 255]),
            "Interbrain": dict(id=5, color=[255, 255, 0]),
            "Midbrain": dict(id=6, color=[255, 0, 255]),
            "Hindbrain": dict(id=7, color=[255, 255, 255]),
            "Cerebellum": dict(id=8, color=[255, 150, 45]),
        }

        # === data setting =======
        self.Homepath = os.path.join("/", "home", "challenger", "ascender")
        self.label_folder = os.path.join(self.Homepath, "data", "labels")
        self.image_folder = os.path.join(self.Homepath, "data", "images")
        self.image_table = pd.read_csv(os.path.join(self.Homepath, "data", "ImageDataTable.csv"))
        self.label_table = pd.read_csv(os.path.join(self.Homepath, "data", "LabelDataTable.csv"))

        # ==== device setting =======
        self.random_seed = 1
        self.wandb_logging = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # == backbone model params ==
        self.backbone = "efficientnetv2_rw_m"  # "resnet101" #"efficientnetv2_rw_m" # "timm-efficientnet-b8"
        self.dim = 1024
        self.pred_dim = 512
        self.pretrain_pretrained = True

        # ====== segmentation params ======
        self.mode = None
        self.Segmentation_MODEL_NAME = "DeepLabV3+" # "MCD" # "DeepLabV3+"
        self.Segmentation_Version = f"{self.backbone}"
        self.Segmentation_Fold = 1
        self.Segmentation_LearningRate = 5e-4
        self.Segmentation_LearningRate_min = 1e-8
        self.Segmentation_BatchSize_train = 2
        self.Segmentation_BatchSize_valid = 16
        self.Segmentation_BatchSize_test = 64
        self.Segmentation_NumWorkers = 8
        self.Segmentation_Epoch = 1000
        self.Segmentation_Normal_Train = 300
        self.Segmentation_Self_pseudo_Train = 600
        self.Segmentation_Self_pseudo_class_Train = 1001
        self.now_epoch = 0
        self.Segmentation_Self_pseudo_pos_Pos = 0.16
        self.Segmentation_Self_pseudo_pos_Neg = 0.45
        self.label_Norm_eta = 0.05

        self.save_folder_name = f'{self.Segmentation_MODEL_NAME}_V_{self.Segmentation_Version}'
        self.Log_Folder = os.path.join(self.Homepath, "outputs", "Segmentation_result")
        self.Log_ViewFolder = os.path.join(self.Log_Folder, self.save_folder_name, "view")
        # self.Log_labelImageFolder = os.path.join(self.Log_Folder, f"labels")
        if self.mode == None:
           self.set_folders()

        self.Name = f'BTIO : r = 16% re'
        # self.Name = 'test'

    def set_folders(self):
        os.makedirs(self.Log_Folder, exist_ok=True)
        os.makedirs(self.Log_ViewFolder, exist_ok=True)
        # os.makedirs(self.Log_labelImageFolder, exist_ok=True)
