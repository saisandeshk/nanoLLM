# src/nanoLLM/data/tokenizer.py
import os
from typing import List
from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders, normalizers

class Tokenizer:
    def __init__(self, vocab_size: int = 32000):
        # Create a new BPE tokenizer
        self.tokenizer = HFTokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(), 
            normalizers.StripAccents()
        ])
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.tokenizer.decoder = decoders.ByteLevel()
        
        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            show_progress=True
        )

    @classmethod
    def from_file(cls, tokenizer_path: str):
        """Loads a tokenizer from a saved file."""
        instance = cls()
        instance.tokenizer = HFTokenizer.from_file(tokenizer_path)
        return instance

    def train(self, texts_iterator, save_path: str):
        """Trains the tokenizer from an iterator of texts."""
        print("Training tokenizer...")
        self.tokenizer.train_from_iterator(texts_iterator, trainer=self.trainer)
        
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        self.tokenizer.save(save_path)
        print(f"Tokenizer trained and saved to {save_path}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes text into a list of token IDs."""
        # Note: The tokenizer library automatically handles the logic
        # based on what it learned during training. We don't manually add tokens.
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back to text."""
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id("<pad>")

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.token_to_id("<eos>")