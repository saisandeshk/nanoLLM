import os
import pickle
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
from ..data.tokenizer import Tokenizer

class TextTokenDataset(Dataset):
    def __init__(self, tokens: list, seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

def load_and_cache_data(config, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(config.tokenizer_output_dir, exist_ok=True)  # Ensure tokenizer output dir exists
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens
    
    print(f"🔄 Processing new data (will cache for future use)")
    
    # Train tokenizer from scratch
    tokenizer = Tokenizer(vocab_size=config.vocab_size)
    
    # Load dataset
    dataset = load_dataset(config.dataset_name, config.dataset_subset, split="train", streaming=True)
    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])
    print(f"Loaded {len(texts)} documents")
    
    # Train tokenizer
    tokenizer.train(texts, os.path.join(config.tokenizer_output_dir, "tokenizer.json"))
    
    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        # Note: We don't add special tokens during training data tokenization
        tokens = tokenizer.encode(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    
    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    print(f"💾 Cached data to {cache_file}")
    
    return texts, tokenizer, tokens