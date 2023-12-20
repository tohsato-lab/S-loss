import torch
import torch.nn as nn
import timm


class EfficientNetBaseEncoder(nn.Module):
    def __init__(self, encoder_name, encoder=None, in_channels=3, output_stride=16, pretrained=False):
        super().__init__()

        self.output_stride = output_stride
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = timm.create_model(
                encoder_name, pretrained=pretrained, output_stride=output_stride)
        out_channels = [in_channels]
        for i in self.encoder.feature_info:
            out_channels.append(i["num_chs"])

        self._stage_idxs = [self.encoder.feature_info[i]["stage"]
                            for i in range(1, len(self.encoder.feature_info)-1)]
        self.out_channels = out_channels
        self._depth = len(self._stage_idxs) + 2
        self._in_channels = 3

        self.classifier = self.encoder.classifier
        del self.encoder.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.encoder.conv_stem,
                          self.encoder.bn1,
                          ),
            self.encoder.blocks[: self._stage_idxs[0]],
            self.encoder.blocks[self._stage_idxs[0]: self._stage_idxs[1]],
            self.encoder.blocks[self._stage_idxs[1]: self._stage_idxs[2]],
            self.encoder.blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

