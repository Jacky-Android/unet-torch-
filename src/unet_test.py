from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        
        
        self.downsample = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.layernorm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.link = nn.Conv2d(dim,dim,kernel_size=1,stride = 2)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        shortcut = self.link(x)
        x = self.downsample(x)
        x = self.layernorm(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        
        
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net_convnextblock(nn.Module):   
    def __init__(self,img_ch=3,num_classes:int=2,layer_scale_init_value:float=1e-6):
        super(U_Net_convnextblock,self).__init__()
        self.relu=nn.ReLU()
        #self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64) #64
        self.block1 = Block(dim=64,layer_scale_init_value=layer_scale_init_value)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)  #64 128
        self.block2 = Block(dim=128,layer_scale_init_value=layer_scale_init_value)
        self.Conv3 = conv_block(ch_in=128,ch_out=256) #128 256
        self.block3 = Block(dim=256,layer_scale_init_value=layer_scale_init_value)
        self.Conv4 = conv_block(ch_in=256,ch_out=512) #256 512
        self.block4 = Block(dim=512,layer_scale_init_value=layer_scale_init_value)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024) #512 1024

        
        self.Up5 = up_conv(ch_in=1024,ch_out=512)  #1024 512
        self.Up_conv5 = conv_block(ch_in=512, ch_out=512)  

        self.Up4 = up_conv(ch_in=512,ch_out=256)  #512 256
        self.Up_conv4 = conv_block(ch_in=256, ch_out=256)  
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)  #256 128
        self.Up_conv3 = conv_block(ch_in=128, ch_out=128) 
        
        self.Up2 = up_conv(ch_in=128,ch_out=64) #128 64
        self.Up_conv2 = conv_block(ch_in=64, ch_out=64)  

        self.Conv_1x1 = nn.Conv2d(64,num_classes,kernel_size=1,stride=1,padding=0)  #64


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
       
        

        x2 = self.block1(x1)
        
        x2 = self.Conv2(x2)
       
        
        x3 = self.block2(x2)
        
        x3 = self.Conv3(x3)
        

        x4 = self.block3(x3)
        x4 = self.Conv4(x4)
        

        x5 = self.block4(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        #d5 = torch.cat((x4,d5),dim=1)
        d5 = self.relu(d5+x4)
        print(d5.shape)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        #d4 = torch.cat((x3,d4),dim=1)
        d4 = self.relu(d4+x3)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        #d3 = torch.cat((x2,d3),dim=1)
        d3 = self.relu(d3+x2)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        #d2 = torch.cat((x1,d2),dim=1)
        d2 = self.relu(d2+x1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return {"out":d1}
