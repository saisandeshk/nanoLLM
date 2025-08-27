import hydra
from omegaconf import DictConfig
from src.nanoLLM.data.dataset import load_and_cache_data
from src.nanoLLM.utils.config import set_seed

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    print("Preparing dataset...")
    texts, tokenizer, tokens = load_and_cache_data(cfg.data)
    print(f"Dataset prepared with {len(texts)} documents and {len(tokens)} tokens")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

if __name__ == "__main__":
    main()