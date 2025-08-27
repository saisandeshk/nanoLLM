import os
import pickle
from typing import List, Tuple, Optional
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tqdm import tqdm

class Tokenizer:
    def __init__(self, tokenizer_path: Optional[str] = None, vocab_size: int = 32000):
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = HFTokenizer.from_file(tokenizer_path)
        else:
            # Create a new tokenizer
            self.tokenizer = HFTokenizer(models.BPE(unk_token="[UNK]"))
            self.tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFD(), 
                normalizers.Lowercase(), 
                normalizers.StripAccents()
            ])
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            self.tokenizer.decoder = decoders.ByteLevel()
            self.trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                show_progress=True
            )
        
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.eos_token = "[SEP]"
        self.bos_token = "[CLS]"
    
    def train(self, texts: List[str], save_path: str):
        print("Training tokenizer...")
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)
        self.tokenizer.save(save_path)
        print(f"Tokenizer saved to {save_path}")
    
    def encode(self, text: str, max_length: int = None, truncation: bool = True, add_special_tokens: bool = True):
        if add_special_tokens:
            # Add [CLS] at the beginning and [SEP] at the end
            text = self.bos_token + " " + text + " " + self.eos_token
        
        encoded = self.tokenizer.encode(text)
        if max_length and truncation:
            encoded = encoded.ids[:max_length]
        else:
            encoded = encoded.ids
        
        # Create attention mask
        attention_mask = [1] * len(encoded)
        
        return {"input_ids": encoded, "attention_mask": attention_mask}
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True):
        if skip_special_tokens:
            # Remove special tokens
            token_ids = [tid for tid in token_ids if tid not in [
                self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id
            ]]
        return self.tokenizer.decode(token_ids)
    
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id(self.pad_token)
    
    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id(self.eos_token)
    
    @property
    def bos_token_id(self):
        return self.tokenizer.token_to_id(self.bos_token)
    
    @property
    def unk_token_id(self):
        return self.tokenizer.token_to_id(self.unk_token)