from abc import ABC

import torch
import torch.nn as nn
from scipy.stats import truncnorm
import math

from mmcv.ops import ModulatedDeformConv2d

from ...network_builder import STRUCTURE


# Visual Affinity Creator for Deformable Convolution (PGPS mechanism)
class Visual_Affinity_Creator_Deformable(nn.Module):
    def __init__(self, kernel_size=3):
        super(Visual_Affinity_Creator_Deformable, self).__init__()
        self.kernel_size = kernel_size
        self.ch_num = kernel_size * kernel_size  # All pixels including center

        self.depth_sampler = ModulatedDeformConv2d(
            in_channels=1,
            out_channels=self.ch_num,  # K^2 outputs, one for each kernel position
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            deform_groups=1
        )

        with torch.no_grad():
            self.depth_sampler.weight.zero_()
            for i in range(self.ch_num):
                ky = i // kernel_size
                kx = i % kernel_size
                self.depth_sampler.weight[i, 0, ky, kx] = 1.0
            self.depth_sampler.bias.zero_()

        # Freeze the weights - we don't want them to change during training
        self.depth_sampler.weight.requires_grad = False
        self.depth_sampler.bias.requires_grad = False

    def forward(self, depth, offset, temperature=1.0, normalize_mode='threshold',
                decay_rate=4.0, relative_threshold=0.3):

        B, _, H, W = depth.shape
        K2 = self.ch_num

        # Create mask (all ones - we want full weight from each position)
        mask = torch.ones(B, K2, H, W, dtype=depth.dtype, device=depth.device)

        sampled_depth = self.depth_sampler(depth, offset, mask)

        center_idx = self.kernel_size * self.kernel_size // 2  # Center index
        center_depth = sampled_depth[:, center_idx:center_idx+1, :, :]  # (B, 1, H, W)

        depth_diff = torch.abs(sampled_depth - center_depth)  # (B, K^2, H, W)

        if normalize_mode != 'threshold' and temperature != 1.0:
            depth_diff = depth_diff / temperature

        # Normalize to get affinity weights
        if normalize_mode == 'softmax':
            affinity = torch.nn.functional.softmax(-depth_diff, dim=1)
        elif normalize_mode == 'l1':
            affinity = torch.exp(-depth_diff)
            affinity = affinity / (affinity.sum(dim=1, keepdim=True) + 1e-8)
        elif normalize_mode == 'threshold':
            relative_diff = depth_diff / (center_depth + 1e-8)  # (B, K^2, H, W)

            within_threshold = relative_diff <= relative_threshold

            affinity = torch.exp(-decay_rate * (depth_diff - (center_depth * relative_threshold)))  # (B, K^2, H, W)

            affinity[within_threshold] = 1.0
        else:
            affinity = torch.exp(-depth_diff)
            affinity = affinity / (affinity.sum(dim=1, keepdim=True) + 1e-8)

        return affinity


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Basic2d(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out

class Basic2dTrans(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module, ABC):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out

class StoDepth_BasicBlock(nn.Module, ABC):
    expansion = 1

    def __init__(self, prob, m, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = m
        self.multFlag = multFlag

    def forward(self, x):

        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out

class Guide(nn.Module, ABC):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=1, input_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = Basic2d(input_planes*2, input_planes, norm_layer)

    def forward(self, feat, weight):
        weight = torch.cat((feat, weight), dim=1)
        weight = self.conv(weight)
        return weight

class BasicDepthEncoder(nn.Module, ABC):

    def __init__(self, kernel_size, block=BasicBlock, bc=16, norm_layer=nn.BatchNorm2d,
                 use_depth_similarity=False, similarity_temperature=1.5,
                 normalize_mode='threshold', decay_rate=4.0, relative_threshold=0.3):
        super(BasicDepthEncoder, self).__init__()
        self._norm_layer = norm_layer
        self.kernel_size = kernel_size
        self.num = kernel_size*kernel_size - 1
        self.idx_ref = self.num // 2
        self.use_depth_similarity = use_depth_similarity
        self.similarity_temperature = similarity_temperature
        self.normalize_mode = normalize_mode
        self.decay_rate = decay_rate
        self.relative_threshold = relative_threshold

        self.convd1 = Basic2d(1, bc * 2, norm_layer=None, kernel_size=3, padding=1)
        self.convd2 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)

        self.convf1 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)
        self.convf2 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)

        self.conv = Basic2d(bc * 4, bc * 4, norm_layer=None, kernel_size=3, padding=1)
        self.ref = block(bc * 4, bc * 4, norm_layer=norm_layer, act=False)
        self.conv_weight = nn.Conv2d(bc * 4, self.kernel_size**2, kernel_size=1, stride=1, padding=0)
        self.conv_offset = nn.Conv2d(bc * 4, 2*(self.kernel_size**2 - 1), kernel_size=1, stride=1, padding=0)

        # Visual affinity creator for PGPS
        if self.use_depth_similarity:
            self.visual_affinity_creator = Visual_Affinity_Creator_Deformable(kernel_size=kernel_size)

    def forward(self, depth, context, best_corrected_depth=None):
        """
        Forward pass with optional PGPS
        Args:
            depth: (B, 1, H, W) current depth estimate
            context: (B, C, H, W) context features
            best_corrected_depth: (B, 1, H, W) reference depth for similarity (optional, for PGPS)
        Returns:
            weight: (B, K^2, H, W) learned affinity weights (potentially modulated by visual affinity)
            offset: (B, 2*K^2, H, W) offsets for deformable convolution
        """
        B, _, H, W = depth.shape

        d1 = self.convd1(depth)
        d2 = self.convd2(d1)

        f1 = self.convf1(context)
        f2 = self.convf2(f1)

        input_feature = torch.cat((d2, f2), dim=1)
        input_feature = self.conv(input_feature)
        feature = self.ref(input_feature)
        weight = torch.sigmoid(self.conv_weight(feature))
        offset = self.conv_offset(feature)

        # Add zero reference offset FIRST (before computing visual affinity)
        # This is necessary because visual affinity needs the complete offset tensor
        offset_temp = offset.view(B, self.num, 2, H, W)
        list_offset = list(torch.chunk(offset_temp, self.num, dim=1))
        list_offset.insert(self.idx_ref,
                           torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

        # Apply PGPS: modulate learned weight with visual affinity
        if self.use_depth_similarity and best_corrected_depth is not None:
            # Compute visual affinity based on depth similarity at deformed positions
            visual_affinity = self.visual_affinity_creator(
                best_corrected_depth,
                offset,  # Pass offset so affinity is computed at deformed positions
                temperature=self.similarity_temperature,
                normalize_mode=self.normalize_mode,
                decay_rate=self.decay_rate,
                relative_threshold=self.relative_threshold
            )

            weight = weight * visual_affinity.detach()
        return weight, offset

class Post_process_deconv(nn.Module, ABC):

    def __init__(self, dkn_residual = True, kernel_size = 3):
        super().__init__()

        self.dkn_residual = dkn_residual
        self.kernel_size = kernel_size

        # ModulatedDeformConv2d: in_channels=1 (depth), out_channels=1, kernel_size=3
        self.dcn = ModulatedDeformConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=int((kernel_size - 1) / 2),
            dilation=1,
            deform_groups=1
        )

        # Initialize weights to ones and bias to zeros (same as original)
        nn.init.ones_(self.dcn.weight)
        nn.init.zeros_(self.dcn.bias)

    def forward(self, depth, weight, offset):
        # weight normalization (same as original)
        if self.dkn_residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        # ModulatedDeformConv2d expects: input, offset, mask
        # offset: (B, 2*kernel_size^2, H, W) - already in correct format
        # mask (weight): (B, kernel_size^2, H, W) - already in correct format
        output = self.dcn(depth, offset, weight)

        # Add residual connection
        if self.dkn_residual:
            output = output + depth

        return output

@STRUCTURE.register_module()
class OursLRRU_1(nn.Module, ABC):

    def __init__(self, bc=16, prob= 0.5, kernel_size=3, dkn_residual=True, block=StoDepth_BasicBlock, multFlag=True, layers=(2, 2, 2, 2, 2),
                 norm_layer=nn.BatchNorm2d, guide=Guide, weight_ks=1,
                 use_depth_similarity=False, similarity_temperature=1.5, relative_error_threshold=0.3,
                 normalize_mode='threshold', decay_rate=4.0, relative_threshold=0.5):
        super().__init__()
        self.dep_max = None
        self.kernel_size = kernel_size
        self._norm_layer = norm_layer
        self.preserve_input = True
        self.use_depth_similarity = use_depth_similarity
        self.similarity_temperature = similarity_temperature
        self.relative_error_threshold = relative_error_threshold
        self.normalize_mode = normalize_mode
        self.decay_rate = decay_rate
        self.relative_threshold = relative_threshold

        prob_0_L = (1, prob)
        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(layers) - 1)

        self.conv_img = Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)
        self.conv_lidar = Basic2d(1, bc * 2, norm_layer=None, kernel_size=5, padding=2)
        in_channels = bc * 2

        self.inplanes = in_channels
        self.layer1_img, self.layer1_lidar = self._make_layer(block, in_channels * 2, layers[0], stride=1)
        self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)

        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img, self.layer2_lidar = self._make_layer(block, in_channels * 4, layers[1], stride=2)
        self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)

        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img, self.layer3_lidar = self._make_layer(block, in_channels * 8, layers[2], stride=2)
        self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)

        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img, self.layer4_lidar = self._make_layer(block, in_channels * 8, layers[3], stride=2)
        self.guide4 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)

        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_img, self.layer5_lidar = self._make_layer(block, in_channels * 8, layers[4], stride=2)


        self.layer4d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.upproj0 = nn.Sequential(
            Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer),
            Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer),
            Basic2dTrans(in_channels * 2, in_channels, norm_layer)
        )
        self.weight_offset0 = BasicDepthEncoder(kernel_size=3, block=BasicBlock, bc=bc, norm_layer=nn.BatchNorm2d,
                                                use_depth_similarity=use_depth_similarity,
                                                similarity_temperature=similarity_temperature,
                                                normalize_mode=normalize_mode,
                                                decay_rate=decay_rate,
                                                relative_threshold=relative_threshold)

        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.upproj1 = nn.Sequential(
            Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer),
            Basic2dTrans(in_channels * 4, in_channels, norm_layer)
        )
        self.weight_offset1 = BasicDepthEncoder(kernel_size=3, block=BasicBlock, bc=bc, norm_layer=nn.BatchNorm2d,
                                                use_depth_similarity=use_depth_similarity,
                                                similarity_temperature=similarity_temperature,
                                                normalize_mode=normalize_mode,
                                                decay_rate=decay_rate,
                                                relative_threshold=relative_threshold)

        self.layer2d = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.upproj2 = nn.Sequential(
            Basic2dTrans(in_channels * 4, in_channels, norm_layer)
        )
        self.weight_offset2 = BasicDepthEncoder(kernel_size=3, block=BasicBlock, bc=bc, norm_layer=nn.BatchNorm2d,
                                                use_depth_similarity=use_depth_similarity,
                                                similarity_temperature=similarity_temperature,
                                                normalize_mode=normalize_mode,
                                                decay_rate=decay_rate,
                                                relative_threshold=relative_threshold)

        self.layer1d = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.conv = Basic2d(in_channels * 2, in_channels, norm_layer)
        self.weight_offset3 = BasicDepthEncoder(kernel_size=3, block=BasicBlock, bc=bc, norm_layer=nn.BatchNorm2d,
                                                use_depth_similarity=use_depth_similarity,
                                                similarity_temperature=similarity_temperature,
                                                normalize_mode=normalize_mode,
                                                decay_rate=decay_rate,
                                                relative_threshold=relative_threshold)

        self.Post_process = Post_process_deconv(dkn_residual=dkn_residual, kernel_size=kernel_size)

        self._initialize_weights()


    def forward(self, input):

        depth = input['sparse_input']
        img = input['rgb']
        lidar = input['best_corrected_depth']

        # PGPS: Get reference depth for similarity computation
        best_corrected_depth = input.get('best_corrected_depth', None) if self.use_depth_similarity else None

        # PGPS: Filter out sparse depth points with large relative error
        if best_corrected_depth is not None and self.use_depth_similarity:
            mask_valid = depth.sign()  # (B, 1, H, W) - binary mask of valid sparse depth points
            relative_error = torch.abs(depth - best_corrected_depth) / (best_corrected_depth + 1e-8)
            mask_keep = (relative_error <= self.relative_error_threshold) | (mask_valid == 0)
            depth = depth * mask_keep.float()  # zero out high-error sparse depth points

        c0_img = self.conv_img(img)
        c0_lidar = self.conv_lidar(depth)

        c1_img = self.layer1_img(c0_img)
        c1_lidar = self.layer1_lidar(c0_lidar)
        c1_lidar_dyn = self.guide1(c1_lidar, c1_img)

        c2_img = self.layer2_img(c1_img)
        c2_lidar = self.layer2_lidar(c1_lidar_dyn)
        c2_lidar_dyn = self.guide2(c2_lidar, c2_img)

        c3_img = self.layer3_img(c2_img)
        c3_lidar = self.layer3_lidar(c2_lidar_dyn)
        c3_lidar_dyn = self.guide3(c3_lidar, c3_img)

        c4_img = self.layer4_img(c3_img)
        c4_lidar = self.layer4_lidar(c3_lidar_dyn)
        c4_lidar_dyn = self.guide4(c4_lidar, c4_img)

        c5_img = self.layer5_img(c4_img)
        c5_lidar = self.layer5_lidar(c4_lidar_dyn)

        list_feat_depth = []
        c5 = c5_img + c5_lidar
        dc4 = self.layer4d(c5)
        c4 = dc4 + c4_lidar_dyn
        c4_up = self.upproj0(c4)
        if self.preserve_input:
            # Upsample lidar to match depth tensor size
            lidar_upsampled = torch.nn.functional.interpolate(lidar, size=(depth.shape[2], depth.shape[3]), mode='bilinear', align_corners=False)
            # Take only the first channel if lidar has 2 channels
            if lidar_upsampled.shape[1] > 1:
                lidar_upsampled = lidar_upsampled[:, 0:1, :, :]
            
            mask = torch.sum(depth > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(depth)
            lidar = (1.0 - mask) * lidar_upsampled + mask * depth
        else:
            lidar = lidar
        lidar = lidar.detach()
        # PGPS: Pass best_corrected_depth to weight_offset for visual affinity computation
        weight0, offset0 = self.weight_offset0(lidar, c4_up, best_corrected_depth)
        output = self.Post_process(lidar, weight0, offset0)
        list_feat_depth.append(torch.relu(output))

        dc3 = self.layer3d(c4)
        c3 = dc3 + c3_lidar_dyn
        c3_up = self.upproj1(c3)
        if self.preserve_input:
            mask = torch.sum(depth > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(depth)
            output = (1.0 - mask) * output + mask * depth
        else:
            output = output
        output = output.detach()
        # PGPS: Pass best_corrected_depth to weight_offset for visual affinity computation
        weight1, offset1 = self.weight_offset1(output, c3_up, best_corrected_depth)
        output = self.Post_process(output, weight1, offset1)
        list_feat_depth.append(torch.relu(output))

        dc2 = self.layer2d(c3)
        c2 = dc2 + c2_lidar_dyn
        c2_up = self.upproj2(c2)
        if self.preserve_input:
            mask = torch.sum(depth > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(depth)
            output = (1.0 - mask) * output + mask * depth
        else:
            output = output
        output = output.detach()
        # PGPS: Pass best_corrected_depth to weight_offset for visual affinity computation
        weight2, offset2 = self.weight_offset2(output, c2_up, best_corrected_depth)
        output = self.Post_process(output, weight2, offset2)
        list_feat_depth.append(torch.relu(output))

        dc1 = self.layer1d(c2)
        c1 = dc1 + c1_lidar_dyn
        c1 = self.conv(c1)
        c0 = c1 + c0_lidar
        if self.preserve_input:
            mask = torch.sum(depth > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(depth)
            output = (1.0 - mask) * output + mask * depth
        else:
            output = output
        output = output.detach()
        # PGPS: Pass best_corrected_depth to weight_offset for visual affinity computation
        weight3, offset3 = self.weight_offset3(output, c0, best_corrected_depth)
        output = self.Post_process(output, weight3, offset3)

        list_feat_depth.append(torch.relu(output))
        depth_pred = torch.relu(output)
        
        output = {'depth_pred': depth_pred,
                  'list_feat_depth':list_feat_depth}

        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        img_downsample, depth_downsample = None, None
        if stride != 1 or self.inplanes != planes * block.expansion:
            img_downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            depth_downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
        img_layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, img_downsample)]
        depth_layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, depth_downsample)]
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
            img_layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            depth_layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*img_layers), nn.Sequential(*depth_layers)


    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)