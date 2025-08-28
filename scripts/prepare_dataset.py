# scripts/prepare_dataset.py
import os
import argparse
import numpy as np
from datasets import load_dataset
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add src to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.nanoLLM.data import Tokenizer

def get_tokenizer(vocab_size, data_dir, num_docs_for_training):
    """Initializes and trains a tokenizer on a subset of the dataset."""
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    
    if os.path.exists(tokenizer_path):
        print(f"✅ Tokenizer already exists at {tokenizer_path}, loading it.")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print(f"⏳ Training a new tokenizer on {num_docs_for_training:,} documents...")
        tokenizer = Tokenizer(vocab_size=vocab_size)
        dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
        
        # MODIFICATION 2: Create a limited text iterator for training
        def limited_text_iterator():
            count = 0
            for example in dataset:
                if count >= num_docs_for_training:
                    break
                yield example['text']
                count += 1
        
        tokenizer.train(limited_text_iterator(), tokenizer_path)
        
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

def process_and_save_split(split_name, tokenizer, data_dir, num_workers, max_docs: int = 0):
    """
    Processes a single split, with an optional limit on the number of documents.
    max_docs=0 means no limit.
    """
    print(f"Processing '{split_name}' split...")
    if max_docs > 0:
        print(f"⚠️  Limiting to a maximum of {max_docs:,} documents.")
        
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split=split_name, streaming=True)
    
    output_path = os.path.join(data_dir, f"{split_name}.bin")
    eos_token_id = tokenizer.eos_token_id

    # The rest of this function is from the "Killed" fix, with the added max_docs logic
    with ProcessPoolExecutor(max_workers=num_workers) as executor, open(output_path, "wb") as f:
        chunk_size = 1000
        futures = []
        docs_processed = 0
        
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[cyan]Docs processed: {task.completed:,}",
            TimeElapsedColumn(),
        )
        
        with progress:
            task = progress.add_task(f"[cyan]Tokenizing {split_name}...", total=None)
            
            chunk = []
            for example in dataset:
                # --- THIS IS THE NEW LOGIC ---
                if max_docs > 0 and docs_processed >= max_docs:
                    break # Stop if we've reached the document limit
                
                chunk.append(example)
                docs_processed += 1
                
                if len(chunk) == chunk_size:
                    futures.append(executor.submit(tokenize_chunk, chunk, tokenizer, eos_token_id))
                    chunk = []
                    
                    if len(futures) > num_workers * 2:
                        tokens = futures.pop(0).result()
                        if tokens:
                            f.write(np.array(tokens, dtype=np.uint16).tobytes())
                        progress.update(task, advance=chunk_size)

            if chunk:
                futures.append(executor.submit(tokenize_chunk, chunk, tokenizer, eos_token_id))

            for future in futures:
                tokens = future.result()
                if tokens:
                    f.write(np.array(tokens, dtype=np.uint16).tobytes())
                progress.update(task, advance=len(chunk) if len(chunk)<chunk_size else chunk_size) # Adjust for last chunk

    print(f"Finished processing '{split_name}'. Total documents: {docs_processed:,}")
    print(f"Saved tokenized data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare the smollm-corpus for training.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save tokenizer and tokenized data.")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--num_workers", type=int, default=max(1, psutil.cpu_count(logical=False) - 1), help="Number of worker processes for tokenization.")
    parser.add_argument("--tokenizer_docs", type=int, default=1_000_000, help="Number of documents to use for training the tokenizer.")
    
    # --- MODIFICATION 2: Add the new argument for controlling FINAL dataset size ---
    parser.add_argument(
        "--max_docs", 
        type=int, 
        default=2_000_000, # A reasonable default for a quick test run.
        help="Maximum number of documents to process for the final .bin files. Set to 0 for no limit."
    )
    
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    
    tokenizer = get_tokenizer(args.vocab_size, args.data_dir, args.tokenizer_docs)
    print(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    
    # --- MODIFICATION 3: Pass the new argument to the function ---
    process_and_save_split("train", tokenizer, args.data_dir, args.num_workers, args.max_docs)
    
    # --- MODIFICATION 4: Make the validation split logic more robust ---
    print("Creating validation split from train data...")
    train_file = os.path.join(args.data_dir, "train.bin")
    val_file = os.path.join(args.data_dir, "validation.bin")
    
    with open(train_file, 'rb') as f_train:
        all_tokens = np.fromfile(f_train, dtype=np.uint16)
        
    # Use 5% of the data for validation, but cap it at 5M tokens for very large runs
    val_split_size = min(5_000_000, int(len(all_tokens) * 0.05))
    
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