"""Representation learning encoder modules"""

import torch
import torch.nn as nn


class RepresentationEncoder(nn.Module):
    """
    Encoder for learning representations.
    
    This module can be used for self-supervised learning, contrastive learning,
    or any other representation learning approach.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output representations
        """
        super(RepresentationEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Encode input into representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Representation tensor of shape (batch_size, output_dim)
        """
        return self.encoder(x)


