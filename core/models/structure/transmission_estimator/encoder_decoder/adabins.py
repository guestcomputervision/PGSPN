import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.functional as F

from .dec import Eh_Decoder
from .mViT import mViT



class UDFNet(nn.Module):
    """Underwater Depth Fusion Net"""

    def __init__(self, n_bins=128, ambient_light_channel=16):
        super(UDFNet, self).__init__()
                
        # decoder
        prior_channels = 1  # channels of prior parametrization
        self.decoder = Eh_Decoder(
            out_channels=(64 - 1),
            prior_channels=prior_channels,
        )

        # mViT
        self.mViT = mViT(
            in_channels=64,  # decoder output plus prior parametrization
            embedding_dim=64,
            patch_size=16,
            num_heads=4,
            num_query_kernels=64,
            n_bins=n_bins,
            ambient_light_channel=ambient_light_channel
        )

        # regression for bin scores
        self.conv_out = nn.Sequential(
            nn.Conv2d(48, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )

        self.ambient_light_out = nn.Sequential(
            nn.Conv2d(ambient_light_channel, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        # params
        self.n_bins = n_bins

    # Process each channel with the same algorithm
    def process_channel(self, bin_widths_normed_channel):
        # bin edges - torch.Size([1, n_bins])
        bin_edges_normed = torch.cumsum(bin_widths_normed_channel, dim=1)
        # torch.Size([1, n_bins+1]) - add edge at zero
        bin_edges_normed = functional.pad(bin_edges_normed, (1, 0), value=0.0)
        
        # scale to max depth
        bin_edges = bin_edges_normed * 1.0
        
        # bin centers - torch.Size([1, n_bins])
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
        return bin_centers

    def forward(self, encoder_out, sparse_depth):
        
        # decode
        decoder_out = self.decoder(encoder_out, sparse_depth)
        
        # concat prior parametrization
        mvit_in = torch.cat((decoder_out, sparse_depth), dim=1)

        # normed bin widths, range attention maps
        bin_widths_normed, range_attention_maps, ambient_light_embeddings = self.mViT(mvit_in)

        # Process all RGB channels using the same algorithm
        # bin_widths_normed: tuple of (red, green, blue) tensors
        bin_widths_normed_channels = bin_widths_normed
        
        # Apply the same processing to all channels
        bin_centers_channels = [self.process_channel(channel) for channel in bin_widths_normed_channels]
        
        # depth classification scores - 48 x H x W
        depth_scores = self.conv_out(range_attention_maps)

        # linear combination of centers and scores for each channel
        # torch.Size([1, n_bins, H, W]) * torch.Size([1, n_bins, 1, 1]) = torch.Size([1, n_bins, H, W])
        transmission_channels = []
        for bin_centers in bin_centers_channels:
            transm_channel = torch.sum(
                depth_scores * bin_centers.view(bin_centers.size(0), self.n_bins, 1, 1),
                dim=1,
                keepdim=True,
            )
            transmission_channels.append(transm_channel)
        
        transmission = torch.cat(transmission_channels, dim=1)
        ambient_light = self.ambient_light_out(ambient_light_embeddings)
        
        return transmission, ambient_light


def test_udfnet():

    print("Testing UDFNet with random input ...")

    # instantiate model
    udfnet = UDFNet(n_bins=100)

    # generate random input
    random_rgb = torch.rand(4, 3, 480, 640)
    random_prior = torch.rand(4, 2, 240, 320)

    # inference
    out = udfnet(random_rgb, random_prior)

    print("Ok")