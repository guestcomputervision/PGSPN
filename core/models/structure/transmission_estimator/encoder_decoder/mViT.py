import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PatchTransformerEncoder, PixelWiseDotProduct


class mViT(nn.Module):
    """
    - in_channels: number of input channels per pixel
    - embedding_dimension: dimension of patch embeddings for transformer input
    - patch_size: size of patch used for each embedding is patch_size x patch_size
    - num_heads: number of parallel heads for attention
    - num_query_kernels: number of output kernels used to compute range attention maps
    """

    def __init__(
        self,
        in_channels,
        embedding_dim=64,
        patch_size=16,
        num_heads=4,
        num_query_kernels=64,
        n_bins=128,
        ambient_light_channel=16
    ) -> None:
        super(mViT, self).__init__()

        # patch transformer
        self.patch_transformer_encoder = PatchTransformerEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,  # E (embeddings dimension)
            patch_size=patch_size,
            num_heads=num_heads,
        )
        self.num_query_kernels = num_query_kernels  # NK (num kernels)

        # multi layer perceptron for bin widths
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_bins * 3),  # n bins plus max depth
            # nn.ReLU(),
            nn.Softplus(),  # replace ReLU because it causes dead neurons
        )
        self.n_bins = n_bins
        self.ambient_light_channel = ambient_light_channel
        # dot product layer for computing range attention map from decoder output and learned 1x1 kernels
        self.dot_product_layer = PixelWiseDotProduct()

    def forward(self, x):
        
        # embedding_dim x H x W
        out = self.patch_transformer_encoder(
            x.clone()  # TODO test if it works without clone?
        )  # S x E x N (S: num patches aka. sequence length)

        # patch num x 1 x embedding_dim
        # regression head for adaptive bins, size N x E
        
        # 1 x embedding_dim
        bins_head = out[0, ...]
        # out[1 : self.num_query_kernels + 1, ...] -> num_query_kernels x 1 x embedding_dim
        # kernels for attention maps, size N x NK x E
        attention_kernels = out[1 : self.num_query_kernels + 1, ...].permute(1, 0, 2)
        # print("attention_kernels.shape:", attention_kernels.shape)
        # estimating max depth and normed bin_widths
        eps = 0.1
        # 1 x n_bins * 3
        mlp_out = self.mlp(bins_head) + eps
        # max_depth = mlp_out[:, 0].unsqueeze(1)
        # bin_widths_normed = mlp_out[:, 1:]
        bin_widths_normed_red = mlp_out[:, :self.n_bins]
        bin_widths_normed_green = mlp_out[:, self.n_bins:self.n_bins * 2]
        bin_widths_normed_blue = mlp_out[:, self.n_bins * 2:]
        
        bin_widths_normed_red = bin_widths_normed_red / bin_widths_normed_red.sum(
            dim=1, keepdim=True
        )
        bin_widths_normed_green = bin_widths_normed_green / bin_widths_normed_green.sum(
            dim=1, keepdim=True
        )
        bin_widths_normed_blue = bin_widths_normed_blue / bin_widths_normed_blue.sum(
            dim=1, keepdim=True
        )

        # range attention maps, size N x NK x h x w
        attention_maps = self.dot_product_layer(x, attention_kernels)
        # aff_matrix = dc_out.narrow(1,0,48)
        # init_depth = F.relu(dc_out.narrow(1,48,1))
        # var_map = F.sigmoid(dc_out.narrow(1,48+1, self.prop_time)) * self.var_map_max   # (B, self.prop_time, H, W)

        # ambient_light_channel x H x W
        ambient_light_embeddings = attention_maps[:, :self.ambient_light_channel, ...]
        # embedding_dim - ambient_light_channel x H x W
        range_attention_maps = attention_maps[:, self.ambient_light_channel:, ...]

        # print("range_attention_maps.shape: ", range_attention_maps.shape)
        # print("ambient_light_embeddings.shape: ", ambient_light_embeddings.shape)

        bin_widths_normed = [bin_widths_normed_red, bin_widths_normed_green, bin_widths_normed_blue]
        return bin_widths_normed, range_attention_maps, ambient_light_embeddings