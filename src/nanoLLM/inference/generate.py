import torch
import torch.nn.functional as F
from typing import Optional

class Generator:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
        """Generate text using the trained model"""
        self.model.eval()
        
        # Tokenize prompt with special tokens
        input_data = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_data["input_ids"]], dtype=torch.long).to(self.device)
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits = self.model(generated_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ensure next_token is 2D (batch_size, 1) for concatenation
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
                elif next_token.dim() == 1:
                    next_token = next_token.unsqueeze(0)  # [1, 1]
                elif next_token.dim() == 2:
                    if next_token.size(0) == 1:
                        next_token = next_token.unsqueeze(1)  # [1, 1]
                    elif next_token.size(1) == 1:
                        next_token = next_token.unsqueeze(0)  # [1, 1]
                elif next_token.dim() == 3:
                    if next_token.size(0) == 1 and next_token.size(2) == 1:
                        next_token = next_token.squeeze(2)  # [1, 1]
                    elif next_token.size(0) == 1 and next_token.size(1) == 1:
                        next_token = next_token.squeeze(1)  # [1, 1]
                
                # Ensure generated_ids is 2D
                if generated_ids.dim() == 3:
                    generated_ids = generated_ids.squeeze(0)  # Remove batch dimension if it's 3D
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if we reach the end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        return generated_text
    
    def interactive(self):
        """Interactive inference session"""
        print("🤖 Starting interactive inference session")
        print("Type 'quit' to exit")
        
        while True:
            try:
                prompt = input("\nEnter your prompt: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                if not prompt.strip():
                    continue
                
                print("🔄 Generating...")
                generated_text = self.generate(
                    prompt,
                    max_length=150,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
                print(f"\nGenerated text:")
                print(f"📝 {generated_text}")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def demo(self):
        """Run a quick demo of the model's capabilities"""
        print("🎭 Running inference demo")
        
        # Demo prompts
        demo_prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The most important thing to remember is",
            "In the year 2050, technology will",
            "The best way to learn programming is"
        ]
        
        for i, prompt in enumerate(demo_prompts, 1):
            print(f"\nDemo {i}: '{prompt}'")
            print("-" * 50)
            generated_text = self.generate(
                prompt,
                max_length=100,
                temperature=0.7,
                top_k=40,
                top_p=0.85
            )
            print(f"📝 {generated_text}")
            print()