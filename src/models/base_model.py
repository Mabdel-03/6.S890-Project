"""Base model architecture"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model class for Echo(I) project.
    
    Extend this class to implement your specific model architecture.
    """
    
    def __init__(self, config):
        """
        Args:
            config (dict): Model configuration parameters
        """
        super(BaseModel, self).__init__()
        self.config = config
        
        # TODO: Define your model architecture here
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # TODO: Implement forward pass
        raise NotImplementedError
        
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


