# src/nanoLLM/data/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    """
    A PyTorch Dataset that uses a memory-mapped file for efficient handling of large datasets.
    The dataset is expected to be a single binary file of token IDs.
    """
    def __init__(self, data_path: str, block_size: int, stride: int = None):
        """
        Args:
            data_path (str): Path to the memory-mapped binary file.
            block_size (int): The sequence length for the model.
            stride (int): The step size to move between sequences. If None, defaults to block_size.
        """
        super().__init__()
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        
        # Memory-map the file
        # The file contains a sequence of uint16 tokens
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # Calculate the number of possible sequences
        num_tokens = len(self.data)
        self.length = (num_tokens - self.block_size) // self.stride + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        """
        Returns a tuple of (input_ids, labels) for the given index.
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.block_size
        
        # Input sequence
        x = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        
        # Target sequence (shifted by one)
        y = torch.from_numpy(self.data[start_idx + 1 : end_idx + 1].astype(np.int64))
        
        return x, y