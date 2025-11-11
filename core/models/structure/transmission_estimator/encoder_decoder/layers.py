import torch
import torch.nn as nn
import torch.nn.functional as functional


class CombinedUpsample(nn.Module):
    """Upsample an input x by deconvolution, concatenate with a skip_input and additional_input
    followed by two convolutional layers. Additional input (can be None) is upsampled to match
    resolution of skip_input and also concatenated."""

    def __init__(self, in_channels, out_channels, upsample_in_channels, additional_channels=None):
        super(CombinedUpsample, self).__init__()

        # Store channels info for dynamic computation
        self.upsample_in_channels = upsample_in_channels
        self.additional_channels = additional_channels

        # deconvolution layer for upsampling main input
        self.deconv_main = nn.ConvTranspose2d(
            upsample_in_channels,
            upsample_in_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )

        # deconvolution layer for additional input (if needed)
        if additional_channels is not None:
            self.deconv_additional = nn.ConvTranspose2d(
                additional_channels,
                additional_channels,
                kernel_size=2,
                stride=2,
                padding=0
            )
        else:
            self.deconv_additional = None

        # We'll create conv layers dynamically based on actual input channels
        self.convA = None
        self.convB = None
        self.out_channels = out_channels


        self.convA = nn.Conv2d(upsample_in_channels + upsample_in_channels + additional_channels, 
                               self.out_channels, 
                               kernel_size=3, stride=1, padding=1)

        self.convB = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        # leaky relu layers
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def _create_conv_layers_if_needed(self, actual_in_channels):
        """Create conv layers dynamically based on actual input channels"""
        if self.convA is None:
            self.convA = nn.Conv2d(
                actual_in_channels, self.out_channels, kernel_size=3, stride=1, padding=1
            ).to(next(self.parameters()).device)

        if self.convB is None:
            self.convB = nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
            ).to(next(self.parameters()).device)

    def forward(self, x, skip_input, additional_input=None):

        # upscale x using deconvolution
        x_upsampled = self.deconv_main(x)

        # If sizes don't match exactly after deconv, use interpolation to fine-tune
        if x_upsampled.size(2) != skip_input.size(2) or x_upsampled.size(3) != skip_input.size(3):
            x_upsampled = functional.interpolate(
                x_upsampled,
                size=[skip_input.size(2), skip_input.size(3)],
                mode="bilinear",
                align_corners=True,
            )

        # depth wise concatenate skip input
        out = torch.cat([x_upsampled, skip_input], dim=1)

        # additional input handling
        if additional_input is not None:
            # Use deconvolution if available, otherwise use interpolation
            if self.deconv_additional is not None:
                additional_input_upsampled = self.deconv_additional(additional_input)

                # Fine-tune size if needed
                if (additional_input_upsampled.size(2) != skip_input.size(2) or
                    additional_input_upsampled.size(3) != skip_input.size(3)):
                    additional_input_upsampled = functional.interpolate(
                        additional_input_upsampled,
                        size=[skip_input.size(2), skip_input.size(3)],
                        mode="bilinear",
                        align_corners=True,
                    )
            else:
                # Fallback to interpolation
                additional_input_upsampled = functional.interpolate(
                    additional_input,
                    size=[skip_input.size(2), skip_input.size(3)],
                    mode="bilinear",
                    align_corners=True,
                )

            # depth wise concatenate additional input
            out = torch.cat([out, additional_input_upsampled], dim=1)

        # convolutional layers and activation
        out = self.convA(out)
        out = self.leakyreluA(out)
        out = self.convB(out)
        out = self.leakyreluB(out)

        return out



class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, patch_size=16, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()

        # convolution to prepare features for patch embeddings
        self.embedding_convPxP = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # create 500 positional encodings as unique vectors with length embedding_dim
        self.positional_encodings = nn.Parameter(
            torch.rand(embedding_dim, 2160), requires_grad=True
        )

        # transformer layer used by encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=1024,
        )

        # transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=4,
        )

    def forward(self, x):

        # patch embeddings of length embedding_dim, size batch x dim x n_patches
        patch_embeddings = self.embedding_convPxP(x).flatten(2)

        # add positional encodings batchwise
        patch_embeddings += self.positional_encodings[
            :, : patch_embeddings.size(2)
        ].unsqueeze(
            0
        )  # unsqueeze at dim 0 to add batch dimension

        # encode patch embeddings with transformer
        patch_embeddings = patch_embeddings.permute(
            2, 0, 1
        )  # transformer expects n_patches x emb_dim x n_batches
        out = self.transformer_encoder(
            patch_embeddings
        )  # output is n_patches x emb_dim x n_batches

        return out


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, nk = K.size()
        assert (
            c == nk
        ), "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        out = torch.matmul(
            x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1)
        )  # .shape = n, hw, cout
        return out.permute(0, 2, 1).view(n, cout, h, w)