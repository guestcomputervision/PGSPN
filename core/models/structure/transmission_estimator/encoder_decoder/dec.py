import torch.nn as nn
from .common import conv_bn_relu, conv_shuffle_bn_relu, convt_bn_relu
from .stodepth_lineardecay import se_resnet34_StoDepth_lineardecay, se_resnet18_StoDepth_lineardecay,se_resnet68_StoDepth_lineardecay
from torch.nn import functional as F
from .layers import CombinedUpsample

class Eh_Decoder(nn.Module):
    def __init__(
        self,
        out_channels=64,
        prior_channels=0,
        single_channel_output=False,
    ) -> None:
        super(Eh_Decoder, self).__init__()

        self.fe5_conv = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        
        # upsampling layers
        # in_channels equals current channels + concat channels + prior channels
        # Added upsample_in_channels and additional_channels parameters
        self.up0 = CombinedUpsample(
            1024 + prior_channels, 512 // 2, upsample_in_channels=512,
            additional_channels=prior_channels if prior_channels > 0 else None
        )
        self.up1 = CombinedUpsample(
            256 + 256 + prior_channels, 128, upsample_in_channels=256,
            additional_channels=prior_channels if prior_channels > 0 else None
        )
        self.up2 = CombinedUpsample(
            128 + 128 + prior_channels, 64, upsample_in_channels=128,
            additional_channels=prior_channels if prior_channels > 0 else None
        )
        self.up3 = CombinedUpsample(
            64 + 64 + prior_channels, 64, upsample_in_channels=64,
            additional_channels=prior_channels if prior_channels > 0 else None
        )
        self.up4 = CombinedUpsample(
            64 + 64 + prior_channels, 64, upsample_in_channels=64,
            additional_channels=prior_channels if prior_channels > 0 else None
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # optional: 3x3 convolution, 1 channel output
        # for estimating depth directly from encoder-decoder
        self.single_channel_output = single_channel_output
        if self.single_channel_output:
            self.conv3x3 = nn.Conv2d(
                64 - prior_channels,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, features, depth_prior):

        skip5 = features[0]
        skip4 = features[1]
        skip3 = features[2]
        skip2 = features[3]
        skip1 = features[4]

        out = self.fe5_conv(skip5)
        
        # upsample together with skip connections and depth prior
        out = self.up0(out, skip5, depth_prior)  
        out = self.up1(out, skip4, depth_prior)  
        out = self.up2(out, skip3, depth_prior)  
        out = self.up3(out, skip2, depth_prior)  
        out = self.up4(out, skip1, depth_prior)  
        out = self.up5(out) 

        # optional: final convolution to achieve single channel
        if self.single_channel_output:
            out = self.conv3x3(out) 

        return out