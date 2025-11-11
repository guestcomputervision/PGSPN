import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv_bn_relu, conv_shuffle_bn_relu, convt_bn_relu
from .stodepth_lineardecay import se_resnet34_StoDepth_lineardecay, se_resnet18_StoDepth_lineardecay,se_resnet68_StoDepth_lineardecay


class BaseModelv2_Encoder(nn.Module):
    def __init__(self, sto=True, res="res34", suffle_up=False, norm_layer='bn'):
        # def __init__(self):
        super(BaseModelv2_Encoder, self).__init__()
    
        assert norm_layer in ['bn', 'in']

        # 1/1
        self.conv1_rgb = conv_bn_relu(3, 64, kernel=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=2, padding=1, norm_layer=norm_layer)
        if sto == True:
            if res == "res34":
                net = se_resnet34_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=False, norm_layer=norm_layer)
            else:
                net = se_resnet68_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=False, norm_layer=norm_layer)
                
        else:
            if res == "res34":
                net = torchvision.models.resnet34(pretrained=False)
            else:
                net = torchvision.models.resnet50(pretrained=False)
        # 1/2 64
        self.conv2 = net.layer1
        # 1/4 128
        self.conv3 = net.layer2
        # 1/8 256
        self.conv4 = net.layer3
        # 1/16 512
        self.conv5 = net.layer4

        del net

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb):
        
        fe1 = self.conv1_rgb(rgb)

        fe2 = self.conv1(fe1)
        fe2 = self.conv2(fe2)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)

        feats = []
        feats.append(fe5)
        feats.append(fe4)
        feats.append(fe3)
        feats.append(fe2)
        feats.append(fe1)

        return feats