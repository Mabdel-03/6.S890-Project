"""DataLoader utilities"""

from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: PyTorch Dataset object
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


