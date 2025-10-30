"""Dataset implementations for Echo(I) project"""

from torch.utils.data import Dataset
import torch


class EchoDataset(Dataset):
    """
    Base dataset class for Echo(I) project.
    
    Customize this class based on your specific data requirements.
    """
    
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (str): Path to the data directory
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.transform = transform
        # TODO: Load your data here
        
    def __len__(self):
        """Return the size of dataset"""
        # TODO: Implement this
        raise NotImplementedError
        
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            sample: Data sample at the given index
        """
        # TODO: Implement data loading logic
        raise NotImplementedError


