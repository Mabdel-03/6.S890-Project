"""Training loop implementation"""

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """
    Trainer class for model training.
    """
    
    def __init__(self, model, optimizer, criterion, device='cuda'):
        """
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            criterion: Loss function
            device (str): Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.model.to(self.device)
        
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)


