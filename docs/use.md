### Prerequisites

Before you begin, make sure you have the following installed:
*   Git
*   Python 3.10+
*   NVIDIA GPU with CUDA drivers installed (for training)

---

### Step 1: Setup and Installation

First, clone the repository and install the required dependencies. We'll install the project in "editable" mode (`-e`), which means any changes you make to the source code will be immediately available without reinstalling.

```bash
# 1. Clone your repository
git clone <your_repository_url>
cd nanoLLM

# 2. Install PyTorch (adjust for your CUDA version if necessary)
# This example is for CUDA 12.1. Check https://pytorch.org/ for your specific setup.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install the project and all other dependencies
pip install -e .
```

### Step 2: Prepare the Dataset

This is a **one-time step** that downloads the `smollm-corpus`, trains a BPE tokenizer on it, and then tokenizes the entire corpus into efficient binary files (`.bin`).

This process will take some time and can use a significant amount of CPU and disk space (~15-20 GB for the tokenized data).

```bash
# Run the data preparation script
python scripts/prepare_dataset.py
```

After this command finishes, you will have the following crucial files in your `data/` directory:
*   `data/tokenizer.json` (Your trained tokenizer)
*   `data/train.bin` (The tokenized training data)
*   `data/validation.bin` (The tokenized validation data)

### Step 3: Pre-train the 0.6B Model

Now you're ready to start training. The main script will use Hydra to read all the configurations from the `configs/` directory.

#### (Optional but Recommended) Setup Weights & Biases

For long training runs, it's essential to monitor your model's performance. We've integrated Weights & Biases (W&B) for this.

```bash
# 1. Install W&B
pip install wandb

# 2. Log in to your W&B account (you'll be prompted for an API key)
wandb login
```
Logging is enabled by default with `wandb: true` in `configs/train/default.yaml`. You will see a link to your W&B dashboard when you start training.

#### Running the Training

This command will start the training process for the 0.6B model as defined in `configs/model/nanoLLM_0.6B.yaml`. It will automatically use your GPU if available.

```bash
# Start the training process
python scripts/run_training.py
```
You will see a progress bar and log messages in your terminal. During training, checkpoints will be saved to the `outputs/` directory:
*   `outputs/best_model.pt`: Saved whenever the model achieves a new best validation loss.
*   `outputs/final_model.pt`: The final state of the model when training finishes.

#### (Advanced) Overriding Configuration

Thanks to Hydra, you can easily change any training parameter from the command line without editing YAML files. For example, to train with a smaller batch size:
```bash
python scripts/run_training.py train.batch_size=4 train.gradient_accumulation_steps=32
```

### Step 4: Run Inference

Once you have a trained model checkpoint, you can use it to generate text.

#### Interactive Mode (Chatbot)

This is the easiest way to test your model. The script will load the `best_model.pt` by default and prompt you for input.

```bash
# Start the interactive generation script
python scripts/run_generation.py
```
**Example Session:**
```
> The future of artificial intelligence is
🤖: The future of artificial intelligence is a complex and multifaceted topic...
>
```

#### Single Prompt Mode

If you just want to generate text for a single prompt without entering interactive mode, you can pass it as an argument.

```bash
python scripts/run_generation.py inference.prompt="Once upon a time in a faraway land,"
```

#### Using a Different Checkpoint or Parameters

You can easily specify which checkpoint to use or change generation parameters like temperature.

```bash
# Use the final model instead of the best one
python scripts/run_generation.py inference.model_path="outputs/final_model.pt"

# Change temperature and max tokens for a single prompt
python scripts/run_generation.py \
  inference.prompt="The recipe for a perfect day is:" \
  inference.temperature=0.9 \
  inference.max_new_tokens=50
```

---

### Summary of the Workflow

Here is the entire process in a nutshell:

1.  **Setup:** `git clone ...` & `pip install -e .`
2.  **Data Prep:** `python scripts/prepare_dataset.py` (Run once)
3.  **Train:** `python scripts/run_training.py` (Wait for it to finish)
4.  **Infer:** `python scripts/run_generation.py` (Chat with your model!)

Congratulations! You now have a complete workflow for training and running your own small language model from scratch.