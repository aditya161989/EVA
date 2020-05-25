import torch
import torch.nn as nn
from torchvision import models
import model
import os

class Hybrid(nn.Module):
    def __init__(self):
        super(Hybrid, self).__init__()
        self.segment = model.ResNetUNet(n_class=1)
        self.segment.load_state_dict(torch.load(os.path.join(os.getcwd(),'epoch-300weights.pth')))
        self.depth = model.ResNetUNet(n_class=1)
        self.depth.load_state_dict(torch.load(os.path.join(os.getcwd(),'epoch-300dweights.pth')))
    
    def forward(self, input):
        segment = self.segment(input)
        depth = self.depth(input)

        return segment, depth
