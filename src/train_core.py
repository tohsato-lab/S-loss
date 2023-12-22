import wandb
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.io import read_image

from src import iou
from src import image_overlap


class log():
    def __init__(self):
        self.train_loss = []
        self.train_loss_CE = []
        self.train_loss_PS = []
        self.train_loss_CKL = []
        self.train_loss_CKL_neg = []
        self.train_loss_SKL = []
        self.train_mIoU = 0

    def empty_append_zero(self):
        self_dict = vars(self)
        for keys in self_dict:
            if isinstance(self_dict[keys], list):
                if len(self_dict[keys]) == 0:
                    self_dict[keys].append(0)
    def get_log(self):
        self.empty_append_zero()
        logs = {
            "train loss": np.array(self.train_loss).mean().item(),
            "train loss CE": np.array(self.train_loss_CE).mean().item(),
            "train loss PS": np.array(self.train_loss_PS).mean().item(),
            "train loss CKL": np.array(self.train_loss_CKL).mean().item(),
            "train loss CKL negative": np.array(self.train_loss_CKL_neg).mean().item(),
            "train loss SKL": np.array(self.train_loss_SKL).mean().item(),
            "train mIoU": self.train_mIoU,
        }
        return logs


def train_normal(configs, model, train_loader, optimizer, criterion_CE):
    iou_obj = iou.IoU(n_classes=configs.N_classes)
    model.train()
    logs = log()
    for step, ((image, label, name), _, _) in enumerate(train_loader):
        # ---- normal train ------
        image = image.to(configs.device)
        label = label.to(configs.device)

        outputs = model(image)
        loss = criterion_CE(outputs, label)

        logs.train_loss_CE.append(loss.item())
        # ------------------------
        logs.train_loss.append(loss.item())

        # ======= check IoU ===== #
        outputs = outputs.to('cpu').detach().numpy().copy()
        outputs = np.argmax(outputs, axis=1)
        label = label.to('cpu').detach().numpy().copy()
        iou_obj.cal_iou_from_img(outputs, label, name)
        # ======================= #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    iou_mean = iou_obj.get_m_iou()
    logs.train_mIoU = iou_mean
    return logs.get_log()


def train_pseudo(configs, model, train_loader, optimizer, criterion_CE):
    iou_obj = iou.IoU(n_classes=configs.N_classes)
    logs = log()
    model.train()
    for step, ((image, label, name), (image_ps, _, name_ps), _) in enumerate(train_loader):
        # ---- normal train ------
        image = image.to(configs.device)
        label = label.to(configs.device)
        outputs = model(image)
        loss_CE = criterion_CE(outputs, label)
        logs.train_loss_CE.append(loss_CE.item())
        # ------------------------

        # ---- train pseudo ------
        image_ps = image_ps.to(configs.device)
        outputs_ps = model(image_ps)
        # outputs_ps = torch.nn.Softmax(dim=1)(outputs_ps)
        label_ps = outputs_ps.clone().detach()
        # label_ps[:, 0] = 0
        label_ps_shape = label_ps.shape

        percent = torch.quantile(torch.nn.Softmax(dim=1)(label_ps).view(label_ps_shape[0], -1), torch.tensor(0.7, device=configs.device), dim=1)
        # percent = torch.quantile(label_ps.view(label_ps_shape[0], -1), torch.tensor(0.7, device=configs.device), dim=1)
        percent = percent.repeat(torch.prod(torch.tensor(label_ps_shape[1:])), 1).permute(1, 0).reshape(label_ps_shape)
        label_ps = torch.where(label_ps > percent, label_ps, torch.tensor(0., device=configs.device)).argmax(1)
        loss_ps = criterion_CE(outputs_ps, label_ps)
        logs.train_loss_PS.append(loss_ps.item())
        # ------------------------


        loss = loss_CE + loss_ps*0.1
        logs.train_loss.append(loss.item())

        # ======= check IoU ===== #
        outputs = outputs.to('cpu').detach().numpy().copy()
        outputs = np.argmax(outputs, axis=1)
        label = label.to('cpu').detach().numpy().copy()
        iou_obj.cal_iou_from_img(outputs, label, name)
        # ======================= #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    iou_mean = iou_obj.get_m_iou()
    logs.train_mIoU = iou_mean
    return logs.get_log()


def train_pseudo_dataset_ckl(configs, model, train_loader, optimizer, criterion_CE, criterion_KL):
    iou_obj = iou.IoU(n_classes=configs.N_classes)
    logs = log()
    model.train()
    train_data_classes = torch.zeros((1, configs.N_classes), requires_grad=True).to(configs.device)
    pseudo_data_classes = torch.zeros((1, configs.N_classes), requires_grad=True).to(configs.device)
    for step, ((image, label, name), (image_ps, _, name_ps), _) in enumerate(train_loader):
        image_shape = image.shape
        # ---- normal train ------
        image = image.to(configs.device)
        label = label.to(configs.device)
        outputs = model(image)
        loss_CE = criterion_CE(outputs, label)
        logs.train_loss_CE.append(loss_CE.item())
        # ------------------------

        step_i = step + 1
        # ---- cal obj class sum ---
        train_class_sum = torch.nn.Softmax(dim=1)(outputs).sum(dim=(0, 2, 3))
        train_class_sum = train_class_sum / (image_shape[0] * image_shape[2] * image_shape[3])
        train_class_sum = train_class_sum.view((1, configs.N_classes)).detach()
        train_data_classes = train_data_classes*((step_i-1)/step_i) + train_class_sum*(1/step_i)
        # --------------------------

        # ---- train pseudo ------
        image_ps = image_ps.to(configs.device)
        outputs_ps = model(image_ps)
        # outputs_ps = torch.nn.Softmax(dim=1)(outputs_ps)
        label_ps = outputs_ps.clone().detach()
        # label_ps[:, 0] = 0
        label_ps_shape = label_ps.shape
        percent = torch.quantile(torch.nn.Softmax(dim=1)(label_ps).view(label_ps_shape[0], -1), torch.tensor(0.7, device=configs.device), dim=1)
        # percent = torch.quantile(label_ps.view(label_ps_shape[0], -1), torch.tensor(0.7, device=configs.device), dim=1)
        percent = percent.repeat(torch.prod(torch.tensor(label_ps_shape[1:])), 1).permute(1, 0).reshape(label_ps_shape)
        label_ps = torch.where(label_ps > percent, label_ps, torch.tensor(0., device=configs.device)).argmax(1)
        loss_ps = criterion_CE(outputs_ps, label_ps)
        logs.train_loss_PS.append(loss_ps.item())
        # ------------------------
        if configs.now_epoch == configs.Segmentation_Epoch:
            image_ps = image_ps.to('cpu').numpy()
            label_ps = label_ps.to('cpu').numpy()
            image_overlap.save_overlap_image(configs, image_ps, label_ps, name_ps)

        # ---- cal pseudo class sum ---
        train_class_sum = torch.nn.Softmax(dim=1)(outputs_ps).sum(dim=(0, 2, 3))
        train_class_sum = train_class_sum / (image_shape[0] * image_shape[2] * image_shape[3])
        train_class_sum = train_class_sum.view((1, configs.N_classes)).detach()
        pseudo_data_classes = pseudo_data_classes * ((step_i - 1) / step_i) + train_class_sum * (1 / step_i)
        # --------------------------
        DeepLabV3 + _V_efficientnetv2_rw_m_F_1_mIoU
        # ---- class area -----
        outputs = torch.nn.Softmax(dim=1)(outputs)
        outputs_ps = torch.nn.Softmax(dim=1)(outputs_ps)
        outputs_ps_area = outputs_ps.sum(dim=(2, 3)) / (image_shape[2] * image_shape[3])
        outputs_area = outputs.sum(dim=(2, 3)) / (image_shape[2] * image_shape[3])
        loss_C_KL = torch.mean(criterion_KL(outputs_ps_area, outputs_area))
        logs.train_loss_CKL.append(loss_C_KL.item())
        # ---------------------


        loss = loss_CE + loss_ps*0.1 + loss_C_KL*0.1
        # loss = loss_CE + loss_C_KL*0.1
        logs.train_loss.append(loss.item())

        # ======= check IoU ===== #
        outputs = outputs.to('cpu').detach().numpy().copy()
        outputs = np.argmax(outputs, axis=1)
        label = label.to('cpu').detach().numpy().copy()
        iou_obj.cal_iou_from_img(outputs, label, name)
        # ======================= #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- data pseudo KL -----

    loss_sum_KL = torch.mean(criterion_KL(train_data_classes, pseudo_data_classes))
    logs.train_loss_SKL.append(loss_sum_KL.item())
    loss_sum_KL = loss_sum_KL*0.1
    optimizer.zero_grad()
    loss_sum_KL.backward()
    optimizer.step()
    # ------------------------

    iou_mean = iou_obj.get_m_iou()
    logs.train_mIoU = iou_mean
    return logs.get_log()



def validation(configs, model, valid_loader, criterion_CE):
    loss_list = []
    iou_obj = iou.IoU(n_classes=configs.N_classes)
    model.eval()
    with torch.no_grad():
        for step, (image, label, name) in enumerate(valid_loader):
            # ---- normal train ------
            image = image.to(configs.device)
            label = label.to(configs.device)

            outputs = model(image)
            loss = criterion_CE(outputs, label)
            loss_list.append(loss.item())
            # ------------------------

            # ======= check IoU ===== #
            outputs = outputs.to('cpu').detach().numpy().copy()
            outputs = np.argmax(outputs, axis=1)
            label = label.to('cpu').detach().numpy().copy()
            iou_obj.cal_iou_from_img(outputs, label, name)
            # ======================= #
    loss_mean = np.array(loss_list).mean().item()
    iou_mean = iou_obj.get_m_iou()
    return loss_mean, iou_mean


def test(configs, model, test_loader, criterion_CE):
    loss_list = []
    iou_obj = iou.IoU(n_classes=configs.N_classes)
    model.eval()
    with torch.no_grad():
        for step, (image, label_img, name) in enumerate(test_loader):
            label_size = label_img.shape
            image = image.to(configs.device)
            outputs = model(image)
            outputs = outputs.to('cpu').detach()
            for b_n in range(label_size[0]):
                label_path = os.path.join(configs.label_folder, name[b_n])
                label_bn = read_image(label_path)
                label_bn = label_bn.long()

                # ---- normal train ------
                outputs_bn = F.resize(img=outputs[b_n], size=label_bn.shape[1:], interpolation=T.InterpolationMode.NEAREST)
                outputs_bn = outputs_bn.unsqueeze(dim=0)
                loss = criterion_CE(outputs_bn, label_bn)
                loss_list.append(loss.item())
                # ------------------------

                # ======= check IoU ===== #
                outputs_bn = outputs_bn.to('cpu').detach().numpy().copy()
                outputs_bn = np.argmax(outputs_bn, axis=1)
                label_bn = label_bn.to('cpu').detach().numpy().copy()
                iou_obj.cal_iou_from_img(outputs_bn, label_bn, [name[b_n]])
                # ======================= #
    loss_mean = np.array(loss_list).mean().item()
    iou_mean = iou_obj.get_m_iou()
    return loss_mean, iou_mean, iou_obj


def prediction(configs, model, test_loader):
    model.eval()
    with torch.no_grad():
        for step, (image, label_img, name) in enumerate(test_loader):
            image = image.to(configs.device)
            outputs = model(image)
            outputs = outputs.to('cpu').detach()
            outputs = np.argmax(outputs, axis=1)
            outputs = np.uint8(outputs)
            for b_n in range(image.shape[0]):
                outputs_bn = outputs[b_n]
                save_img_path = os.path.join(configs.Homepath, 'data', 'Prediction', name[b_n])
                outputs_bn = Image.fromarray(outputs_bn)
                outputs_bn.save(save_img_path)

                view_img = image_overlap.create_pred_one_image(configs, outputs_bn)
                view_img.save(os.path.join(configs.Homepath, 'data', 'Prediction_view', name[b_n]))

    return None
