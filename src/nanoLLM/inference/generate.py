# src/nanoLLM/inference/generate.py
import torch
import torch.nn.functional as F
from tqdm import trange
# NOTE: This version still re-computes the full sequence each time. A future optimization would be to add a KV Cache.
class Generator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device).eval() # Ensure model is on correct device and in eval mode
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50):
        """Generate text using the trained model."""
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        tokens = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate tokens one by one
        for _ in trange(max_new_tokens, desc="Generating"):
            # Get logits for the next token
            logits = self.model(tokens)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)

            # Stop if EOS token is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode the generated sequence
        generated_text = self.tokenizer.decode(tokens[0].tolist())
        return generated_text