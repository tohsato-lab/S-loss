import torch.nn as nn
import segmentation_models_pytorch as smp

from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.base import SegmentationHead, SegmentationModel, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder

from . import efficientnet
from . import memorymod

def load_MemCLST(configs):
    if configs.backbone is not None:
        encoder_name = configs.backbone
    if configs.pretrain_pretrained is not None:
        pretrained = 'imagenet'
    if configs.backbone == "efficientnetv2_rw_m":
        backbone = efficientnet.EfficientNetBaseEncoder(encoder_name=encoder_name, pretrained=pretrained)
        MemoryMod = memorymod.MemModule(configs=configs)
        models = MemCLST(backbone=backbone, memorymod=MemoryMod, encoder_name=encoder_name)
    else:
        backbone = None
        encoder_name = configs.backbone
        MemoryMod = memorymod.MemModule(configs=configs)
        models = MemCLST_othermodel(backbone=backbone, memorymod=MemoryMod, encoder_name=encoder_name)
    return models


def load_deplaboV3plus(configs):
    if configs.backbone is not None:
        encoder_name = configs.backbone
    if configs.pretrain_pretrained is not None:
        pretrained = 'imagenet'
    if configs.backbone == "efficientnetv2_rw_m":
        backbone = efficientnet.EfficientNetBaseEncoder(encoder_name=encoder_name, pretrained=pretrained)
        models = DeepLaboV3plus(backbone=backbone, encoder_name=encoder_name)
        return models

class MemCLST_othermodel(SegmentationModel):
    def __init__(self,
                 backbone=None,
                 memorymod=None,
                 encoder_name=None,
                 encoder=None,
                 pretrained: bool = False,
                 encoder_depth=5,
                 encoder_weights='imagenet',
                 encoder_output_stride: int = 16,
                 decoder_channels: int = 256,
                 decoder_atrous_rates: tuple = (12, 24, 36),
                 in_channels: int = 3,
                 classes: int = 9):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        self.MemoryMod = memorymod

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

    def forward(self, x):
        features = self.encoder(x)
        # memory_feature = self.MemoryMod(features[-1])

        # normal segmentation
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        # memory mod
        # features[-1] = memory_feature
        # decoder_memory_output = self.decoder(*features)
        # memory_masks = self.segmentation_head(decoder_memory_output)

        memory_masks = masks
        return masks, memory_masks


class MemCLST(SegmentationModel):
    def __init__(self, backbone,
                 memorymod,
                 encoder_name=None,
                 pretrained: bool = False,
                 encoder_output_stride: int = 16,
                 decoder_channels: int = 256,
                 decoder_atrous_rates: tuple = (12, 24, 36),
                 in_channels: int = 3,
                 classes: int = 9):
        super().__init__()
        self.backbone = backbone
        self.MemoryMod = memorymod

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.backbone.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

    def forward(self, x):
        features = self.backbone(x)
        memory_feature = self.MemoryMod(features[-1])

        # normal segmentation
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        # memory mod
        features[-1] = memory_feature
        decoder_memory_output = self.decoder(*features)
        memory_masks = self.segmentation_head(decoder_memory_output)
        # memory_masks = masks
        return masks, memory_masks


class DeepLaboV3plus(SegmentationModel):
    def __init__(self, backbone,
                 encoder_name=None,
                 pretrained: bool = False,
                 encoder_output_stride: int = 16,
                 decoder_channels: int = 256,
                 decoder_atrous_rates: tuple = (12, 24, 36),
                 in_channels: int = 3,
                 classes: int = 9):
        super().__init__()
        self.backbone = backbone

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.backbone.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

    def forward(self, x):
        features = self.backbone(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks
