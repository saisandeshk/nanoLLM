import torch
from dataclasses import dataclass

@dataclass
class DataCollatorForLanguageModeling:
    pad_token_id: int
    
    def __call__(self, features):
        # features is a list of tuples (x, y) where x and y are tensors
        input_ids = [f[0] for f in features]
        labels = [f[1] for f in features]
        
        # Pad input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100  # Use -100 for labels to ignore in loss calculation
        )
        
        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }