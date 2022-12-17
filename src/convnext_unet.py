from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import convnext_base
from .unetvgg16 import Up, OutConv


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class U_ConvNext(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(U_ConvNext, self).__init__()
        backbone = convnext_base(pretrained=pretrain_backbone)

        # if pretrain_backbone:
        #     # 载入mobilenetv3 large backbone预训练权重
        #     # https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
        #     backbone.load_state_dict(torch.load("mobilenet_v3_large.pth", map_location='cpu'))

        backbone = backbone.features
        
        stage_indices = [0,1,3,5,7]
        self.stage_out_channels = [128, 128,256,512,1024]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        
        x = self.up1(backbone_out['stage7'], backbone_out['stage5'])
        x = self.up2(x, backbone_out['stage3'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return {"out": x}
