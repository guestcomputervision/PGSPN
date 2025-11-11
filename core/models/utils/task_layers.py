

import torch.nn as nn

from ..network_builder import TASK


@TASK.register_module()
class DepthEstimation_Task(nn.Module):
    def __init__(self, max_depth):
        super(DepthEstimation_Task, self).__init__()
        
        self.max_depth = max_depth
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(x)*self.max_depth
        return x


@TASK.register_module()
class DepthCompletion_Task(nn.Module):
    def __init__(self, max_depth):
        super(DepthCompletion_Task, self).__init__()
        
    def forward(self, x):
        return x


@TASK.register_module()
class Enhancement_Task(nn.Module):
    def __init__(self):
        super(Enhancement_Task, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(x)
        return x