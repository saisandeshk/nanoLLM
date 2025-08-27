# src/nanoLLM/data/collator.py
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class DataCollator:
    """
    A simple data collator that takes a list of (input, target) tuples
    and stacks them into batches.
    """
    def __call__(self, features: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # `features` is a list of (x, y) tuples from the dataset
        
        # Stack the tensors from each tuple into a single batch tensor
        input_ids = torch.stack([f[0] for f in features])
        labels = torch.stack([f[1] for f in features])
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }