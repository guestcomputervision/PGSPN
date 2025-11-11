import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder.enc import BaseModelv2_Encoder
from .encoder_decoder.adabins import UDFNet
from .depth_scale_corrector import DepthScaleCorrector

from ...network_builder import STRUCTURE

@STRUCTURE.register_module()
class Learning_04_04_step1(nn.Module):
    def __init__(
        self, 
        enc='res34',        # res18, res34, res68
        suffle_up=False,
        sto=True,
        iteration=5
    ):
        super(Learning_04_04_step1, self).__init__()
        
        self.iteration = iteration
        self.var_map_max = 5.0
        self.sto = sto
        self.suffle_up = suffle_up
        
        self.enc = BaseModelv2_Encoder(sto=sto, res=enc, suffle_up=suffle_up, norm_layer='bn')
        self.depth_scale_corrector = DepthScaleCorrector(max_depth=25.0,
                                                         valid_threshold = 1e-6,
                                                         min_valid_points = 10,
                                                         enable_logging = False
                                                         )
        self.eh_dec = UDFNet(n_bins=128)
    
            
    def _metric_aware_restoration(self, original, transmission, ambient):
        clean = torch.clamp((original - ambient) / transmission + ambient, min=0.0, max=1.0)
        return clean
    
    def forward(self, x):
        rgb = x['rgb']
        sparse_dep = x['sparse_input']
        
        feats = self.enc(rgb)
        transmission, ambient_light = self.eh_dec(feats, sparse_dep)
        pseudo_depth = -torch.log(transmission + 1e-8)
        
        corrected_depth, best_corrected, best_channel_indices = self.depth_scale_corrector(pseudo_depth, sparse_dep)
        restoration = self._metric_aware_restoration(x['origin'], transmission, ambient_light)            
        best_transmission = self.depth_scale_corrector.extract_best_channels(transmission, best_channel_indices)
        
        return {'transmission': transmission,
                'ambient_light': ambient_light,
                'restoration': restoration,
                'corrected_depth': corrected_depth,
                'best_corrected_depth': best_corrected,
                'best_transmission': best_transmission
                }        