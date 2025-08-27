# scripts/prepare_dataset.py
import os
import argparse
import numpy as np
from datasets import load_dataset
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Add src to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.nanoLLM.data import Tokenizer

def get_tokenizer(vocab_size, data_dir):
    """Initializes and trains a tokenizer on the dataset."""
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    
    if os.path.exists(tokenizer_path):
        print(f"Tokenizer already exists at {tokenizer_path}, loading it.")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("Training a new tokenizer...")
        tokenizer = Tokenizer(vocab_size=vocab_size)
        # Create a Python generator to stream text data for training
        dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
        
        def text_iterator():
            for example in dataset:
                yield example['text']
        
        tokenizer.train(text_iterator(), tokenizer_path)
        
    return tokenizer

def tokenize_chunk(chunk, tokenizer, eos_token_id):
    """Tokenizes a chunk of text examples."""
    tokens = []
    for example in chunk:
        text = example['text']
        if text:
            # Tokenize and add the EOS token to separate documents
            tokens.extend(tokenizer.encode(text, add_special_tokens=False) + [eos_token_id])
    return tokens

def process_and_save_split(split_name, tokenizer, data_dir, num_workers):
    """Processes a single split of the dataset and saves it to a binary file."""
    print(f"Processing '{split_name}' split...")
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split=split_name)
    
    output_path = os.path.join(data_dir, f"{split_name}.bin")
    eos_token_id = tokenizer.eos_token_id

    # Use ProcessPoolExecutor for parallel tokenization
    with ProcessPoolExecutor(max_workers=num_workers) as executor, open(output_path, "wb") as f:
        # Create chunks of the dataset to distribute to workers
        chunk_size = 1000  # Number of documents per chunk
        futures = []
        chunk = []
        for example in dataset:
            chunk.append(example)
            if len(chunk) == chunk_size:
                futures.append(executor.submit(tokenize_chunk, chunk, tokenizer, eos_token_id))
                chunk = []
        if chunk:
            futures.append(executor.submit(tokenize_chunk, chunk, tokenizer, eos_token_id))

        # Setup rich progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            "ETA:",
            TimeRemainingColumn(),
        )
        
        total_tokens = 0
        with progress:
            task = progress.add_task(f"[cyan]Tokenizing {split_name}...", total=len(futures))
            for future in as_completed(futures):
                tokens = future.result()
                if tokens:
                    # Write tokens to the binary file as uint16
                    f.write(np.array(tokens, dtype=np.uint16).tobytes())
                    total_tokens += len(tokens)
                progress.update(task, advance=1)

    print(f"Finished processing '{split_name}'. Total tokens: {total_tokens:,}")
    print(f"Saved tokenized data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare the smollm-corpus for training.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save tokenizer and tokenized data.")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--num_workers", type=int, default=max(1, psutil.cpu_count(logical=False) - 1), help="Number of worker processes for tokenization.")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Step 1: Get or train the tokenizer
    tokenizer = get_tokenizer(args.vocab_size, args.data_dir)
    print(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    
    # Step 2: Process the 'train' and 'validation' splits
    # Note: smollm-corpus only has 'train', so we'll just process that.
    # In a real scenario, you would create a validation split.
    process_and_save_split("train", tokenizer, args.data_dir, args.num_workers)
    
    # As there is no validation split, we will create a small one from the end of the train split
    print("Creating validation split from train data...")
    train_file = os.path.join(args.data_dir, "train.bin")
    val_file = os.path.join(args.data_dir, "validation.bin")
    
    with open(train_file, 'rb') as f_train:
        all_tokens = np.fromfile(f_train, dtype=np.uint16)
        
    # Use last 5 million tokens for validation
    val_split_size = 5_000_000
    train_tokens = all_tokens[:-val_split_size]
    val_tokens = all_tokens[-val_split_size:]
    
    with open(train_file, 'wb') as f:
        f.write(train_tokens.tobytes())
    with open(val_file, 'wb') as f:
        f.write(val_tokens.tobytes())
        
    print(f"Train split size: {len(train_tokens):,} tokens")
    print(f"Validation split size: {len(val_tokens):,} tokens")
    print("Data preparation complete.")

if __name__ == "__main__":
    main()