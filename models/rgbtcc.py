import torch
import torch.nn as nn
from torchvision import models
import logging
from models.utils import KeypointNorm, KeypointEncoder, ChannelAttn, ResidualDown
from models.fuse_ssm import FuseSSM
from models.cross_ssm import CrossSSM
from kornia.utils import create_meshgrid
from einops import rearrange


class SONet(nn.Module):
    def __init__(self, pretrained=True):
        super(SONet, self).__init__()
        logging.info(f"ssm_dim_opt")
        in_dim = 256
        self.down = ResidualDown(512, in_dim)
        self.chan = ChannelAttn(in_dim)
        
        # regress  
        self.last_layer_rgbt = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, output_padding=0, bias=True),
        )

        self._weight_init_()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        features = list(vgg.features.children())
        self.rgb1 = nn.Sequential(*features[0:6])
        self.rgb2 = nn.Sequential(*features[6:13])
        self.rgb4 = nn.Sequential(*features[13:23])
        self.rgb8 = nn.Sequential(*features[23:33])

        self.cssm = CrossSSM(in_dim)
        self.fssm = FuseSSM(in_dim)
        self.kpe = KeypointEncoder(in_dim, [32, 64, 128, in_dim])

    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def keypoint(self, batch, h, w, dh, dw, device):
        reduce = h // dh
        gd = [rearrange((create_meshgrid(dh, dw, False, device) * reduce).squeeze(0), 'h w t->(h w) t')] * batch
        kp = torch.stack(gd, 0)
        kp = KeypointNorm(kp, (h, w))
        kp = kp.transpose(1, 2)
        kp = self.kpe(kp)
        return kp

    def forward(self, RGBT):
        rgb = RGBT[0]
        tir = RGBT[1]

        rgbt = torch.cat([rgb, tir], 0)
        rgbt = self.rgb1(rgbt)
        rgbt = self.rgb2(rgbt)
        rgbt = self.rgb4(rgbt)
        rgbt = self.rgb8(rgbt)
        rgbt = self.down(rgbt)

        batch, _, h, w = rgb.shape
        _, _, dh, dw = rgbt.shape       
        rgb, tir = self.cssm(rgbt, batch, dh, dw, self.keypoint(batch, h, w, dh, dw, rgb.device))
        
        rgbt = self.chan(rgb, tir)
        rgbt = rgbt.permute(0, 2, 3, 1)
        rgbt = self.fssm(rgbt)
        rgbt = rgbt.permute(0, 3, 1, 2)

        rgbt = self.last_layer_rgbt(rgbt)

        return rgbt


def get_model(train=False):
    model = SONet()
    return model
