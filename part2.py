#!/usr/bin/env python
# coding: utf-8

# # Part 2: Supervised Fine-Tuning (SFT)

# ## Introduction
# 
# This notebook focuses on performing Supervised Fine-Tuning (SFT) on the base model used in Part 1. We will use the `mbpp-rkt-train` dataset to fine-tune the model on the task of Racket code generation. The key steps are:
# 
# 1.  **Configuration**: Set up different hyperparameter configurations for our training experiments.
# 2.  **Data Preparation**: Load and format the training dataset into a prompt-completion structure.
# 3.  **W&B Logging**: Integrate Weights & Biases to log metrics like loss and learning rate.
# 4.  **Training**: Implement a training loop using PyTorch and the `transformers` library.
# 5.  **Model Saving**: Save the fine-tuned model and tokenizer for evaluation in Part 3.
# 
# We will run at least two experiments to observe how different parameters affect the training process.

# In[1]:


# Cell 1: Import Libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
import wandb

print("Libraries imported successfully")


# ## Configuration
# 
# We define our base model, dataset details, and create two different training configurations to experiment with. `Config1` is a quick, single-epoch run. `Config2` is a longer, multi-epoch run with a different learning rate and a learning rate scheduler.

# In[2]:


# Cell 2: Configuration Settings

# --- General Settings ---
BASE_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
DATASET_NAME = "nuprl/engineering-llm-systems"
DATASET_CONFIG = "mbpp-rkt-correct-executions" # Use the training split for SFT
WANDB_NOTEBOOK_NAME = "llm-sft-racket-finetuning"

# --- Experiment 1 Configuration ---
class Config1:
    RUN_NAME = "exp1_lr5e-5_1epoch"
    OUTPUT_DIR = f"./models/{RUN_NAME}"
    NUM_EPOCHS = 1
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 4
    MAX_SEQ_LENGTH = 1024
    USE_SCHEDULER = False

# --- Experiment 2 Configuration ---
class Config2:
    RUN_NAME = "exp2_lr2e-5_3epochs_scheduler"
    OUTPUT_DIR = f"./models/{RUN_NAME}"
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 4
    MAX_SEQ_LENGTH = 1024
    USE_SCHEDULER = True
    WARMUP_STEPS = 50


# --- SELECT CONFIGURATION TO RUN ---

# We explicitly switch between Config1 and Config2 to run different experiments
CURRENT_CONFIG = Config2 # Change to Config2 for the second experiment

print(f"Selected Configuration: {CURRENT_CONFIG.RUN_NAME}")
print(f"  - Base Model: {BASE_MODEL_NAME}")
print(f"  - Output Directory: {CURRENT_CONFIG.OUTPUT_DIR}")
print(f"  - Epochs: {CURRENT_CONFIG.NUM_EPOCHS}")
print(f"  - Learning Rate: {CURRENT_CONFIG.LEARNING_RATE}")


# ## W&B Login
# 
# You need to log in to your Weights & Biases account to track the experiments. You'll be prompted to enter your API key.

# In[4]:


# Cell 3: Login to Weights & Biases
wandb.login()


# ## Load Model, Tokenizer, and Dataset
# 
# Here, we load the pre-trained model and tokenizer. We also load the training dataset and define a formatting function. This function creates a single string for each data point, combining the problem description and the solution. This is the text the model will be trained on to learn the task format.

# In[5]:


# Cell 4: Load Model and Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Set padding token for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model ({BASE_MODEL_NAME}) and tokenizer loaded successfully.")


# In[6]:


# Cell 5: Load and Prepare the Dataset (Corrected for KeyError)

# Load the dataset from Hugging Face
train_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

# Define a single function to handle both formatting and tokenization for a batch
def format_and_tokenize_batch(batch):
    """
    This function takes a batch (a dictionary of lists), formats each example
    into a single string, and then tokenizes the list of strings.
    """
    formatted_texts = []
    num_examples = len(batch['description'])

    for i in range(num_examples):
        text = (
            f"; {batch['description'][i]}\n"
            f"; Input format: {batch['input_format'][i]}\n"
            f"; Output format: {batch['output_format'][i]}\n\n"
            f"{batch['code'][i]}"
        )
        formatted_texts.append(text)

    return tokenizer(
        formatted_texts,
        truncation=True,
        max_length=CURRENT_CONFIG.MAX_SEQ_LENGTH,
        padding="max_length"
    )

tokenized_dataset = train_dataset.map(
    format_and_tokenize_batch,
    batched=True,
    num_proc=4, 
    remove_columns=train_dataset.column_names 
)

tokenized_dataset.set_format("torch")

print(f"Dataset loaded and tokenized. Total examples: {len(tokenized_dataset)}")
print("\nSample decoded tokens from the first example:")
print(tokenizer.decode(tokenized_dataset[0]['input_ids'], skip_special_tokens=True))


# ## Training Loop
# 
# This is the core of our SFT process. The loop iterates through the specified number of epochs. In each step:
# 
# 1.  We get a batch of data from the `DataLoader`.
# 2.  We perform a **forward pass**. By passing `labels=batch['input_ids']`, the model automatically calculates the causal language modeling loss (cross-entropy loss) for us.
# 3.  We perform a **backward pass** to compute gradients (`loss.backward()`).
# 4.  The optimizer updates the model's weights (`optimizer.step()`).
# 5.  We log the `loss` and `learning_rate` to W&B for monitoring.
# 6.  At the end of each epoch, we save a checkpoint of the model and tokenizer.

# In[8]:


def train(config):
    """Main training function."""
    wandb.init(
        project=WANDB_NOTEBOOK_NAME,
        name=config.RUN_NAME,
        config={
            "model_name": BASE_MODEL_NAME,
            "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "max_seq_length": config.MAX_SEQ_LENGTH
        }
    )

    train_loader = DataLoader(tokenized_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    if config.USE_SCHEDULER:
        num_training_steps = len(train_loader) * config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=num_training_steps
        )
    
    print("Starting training...")
    
    model.train()
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            if config.USE_SCHEDULER:
                scheduler.step()
            optimizer.zero_grad()

            wandb.log({
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0] if config.USE_SCHEDULER else config.LEARNING_RATE,
                "epoch": epoch + 1,
                "step": global_step
            })
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1
        
        epoch_output_dir = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch+1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        print(f"Saving model checkpoint to {epoch_output_dir}")
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)

    wandb.finish()
    print("\nTraining complete!")

train(CURRENT_CONFIG)

