import torch
import torch.nn as nn

class ClassKL(nn.Module):
    def __init__(self, mode="mean"):
        super(ClassKL, self).__init__()
        self.mode = mode
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs, labels):
        inputs = self.softmax(inputs)
        labels = self.softmax(labels)
        loss_kl = labels * torch.log(labels) - labels * torch.log(inputs)
        if self.mode == "mean":
            loss = torch.sum(loss_kl, dim=1).mean()
        elif self.mode == 'sum':
            loss = torch.sum(loss_kl, dim=1).sum()
        return loss


class CosSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosSimilarityLoss, self).__init__()
        self.cos = torch.nn.CosineSimilarity()

    def forward(self, a, b):
        loss = 1 - self.cos(a, b)
        loss = torch.mean(loss)
        return loss


# https://arxiv.org/abs/1512.00567
class LabelNormLoss(nn.Module):
    def __init__(self, configs, eta=0.0005):
        super(LabelNormLoss, self).__init__()
        self.KLDiv = torch.nn.KLDivLoss(reduction='mean')
        self.configs = configs
        self.N_classes = self.configs.N_classes
        self.eta = eta

    def onehot(self, image_tensor, num_classes):
        b, h, w = image_tensor.size()
        one_hot = torch.LongTensor(b, num_classes, h, w).zero_()
        image_tensor = image_tensor.unsqueeze_(1).to(int)
        one_hot = one_hot.scatter_(1, image_tensor, 1)
        return one_hot

    def KLdivLoss(self, inputs, labels):
        inputs = torch.nn.Softmax(dim=1)(inputs)
        self_info = labels * labels.log()
        self_info = torch.nan_to_num(self_info)
        KL_loss = self_info - labels * torch.log(inputs)
        KL_loss = KL_loss.sum(dim=1).mean()
        return KL_loss

    def forward(self, inputs, label):
        label_oneHot = label.to('cpu')
        label_oneHot = self.onehot(label_oneHot, num_classes=self.N_classes).to(float)
        label_oneHot = label_oneHot * (1-self.eta - self.eta/(self.N_classes-1))
        label_oneHot = label_oneHot + self.eta/(self.N_classes-1)
        loss = self.KLdivLoss(inputs.to('cpu'), label_oneHot)
        return loss
