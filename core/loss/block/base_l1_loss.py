
import torch.nn as nn
import torch
from ..loss_builder import LOSS_BLOCK


@LOSS_BLOCK.register_module()
class L1_loss(nn.Module):
    
    def __init__(self, lambda_l1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        
    def forward(self, input):
        generated = input[0]
        image_gt = input[1]

        torch_l1_dist = torch.nn.PairwiseDistance(p=1)
        loss = self.lambda_l1 * torch.mean(torch_l1_dist(generated, image_gt))
        
        return loss

@LOSS_BLOCK.register_module()
class Iter_L1_loss(nn.Module):
    
    def __init__(self, lambda_l1, gamma=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.gamma = gamma
        
    def forward(self, input):
        generated_list = input[0]
        image_gt = input[1]
        
        len_generated_list = len(generated_list)
        total_loss = 0.0
        for i, generated in enumerate(generated_list):
            torch_l1_dist = torch.nn.PairwiseDistance(p=1)
            loss = torch.mean(torch_l1_dist(generated, image_gt))
            loss = (self.gamma ** (len_generated_list - i - 1)) * loss 
            total_loss += loss
            
        return self.lambda_l1 * total_loss / len_generated_list


@LOSS_BLOCK.register_module()
class Depth_Estimation_L1_loss(nn.Module):
    
    def __init__(self, lambda_l1, depth_min_eval):
        super().__init__()
        self.lambda_l1 = lambda_l1
        
    def forward(self, x):
        depth_est = x[0]
        depth_gt = x[1]
        mask = x[2]
        
        depth_loss = torch.mean(torch.abs(depth_est[mask] - depth_gt[mask]))
        return self.lambda_l1 * depth_loss


@LOSS_BLOCK.register_module()
class Iter_Depth_Estimation_L1_loss(nn.Module):
    
    def __init__(self, lambda_l1, depth_min_eval, gamma=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.gamma = gamma
        
    def forward(self, x):
        depth_est_list = x[0]
        depth_gt = x[1]
        mask = x[2]
        
        len_depth_est_list = len(depth_est_list)
        total_loss = 0.0
        for i, depth_est in enumerate(depth_est_list):
            loss = torch.mean(torch.abs(depth_est[mask] - depth_gt[mask]))
            loss = (self.gamma ** (len_depth_est_list - i - 1)) * loss 
            total_loss += loss
            
        return self.lambda_l1 * total_loss / len_depth_est_list