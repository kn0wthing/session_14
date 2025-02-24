import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AdamW
)

# ---------------------------
# 1. Define the DeepSeekV3 model
# ---------------------------
class DeepSeekV3Model(nn.Module):
    """A simple Transformer-like model for demonstration purposes."""
    def __init__(self, vocab_size=30522, hidden_dim=256, num_layers=4, num_heads=4):
        super().__init__()
        # Example embedding layer
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Simple feed-forward block repeated for demonstration
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads, 
                dim_feedforward=4*hidden_dim
            ) for _ in range(num_layers)
        ])

        # Final linear layer to project to vocabulary size
        self.linear_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embedding lookup
        x = self.embeddings(input_ids)

        # Pass through a stack of TransformerEncoderLayers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)

        # Project to vocab-size
        logits = self.linear_out(x)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        return {"loss": loss, "logits": logits}

# ---------------------------
# 2. Load the dataset with streaming
# ---------------------------
dataset_name = "cosmopedia-v2"  # e.g., "username/cosmopedia-v2" if it's under a user namespace
dataset = load_dataset(dataset_name, streaming=True, split="train")

# ---------------------------
# 3. Load cosmo2-tokenizer
# ---------------------------
tokenizer_name = "cosmo2-tokenizer"  # e.g., "username/cosmo2-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

# ---------------------------
# 4. Tokenize the dataset
# ---------------------------
def tokenize_function(examples):
    # Adjust "text" key as appropriate if your dataset uses another key name
    return tokenizer(examples["text"])  

# Since we're streaming, we typically apply map with batched=False or handle tokenization in the collate function.
# For demonstration, tokenization is shown here:
tokenized_dataset = dataset.map(tokenize_function)

# ---------------------------
# 5. Prepare data collator
# ---------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# 6. Initialize model
# ---------------------------
# Adjust vocab_size, hidden_dim, etc. to match tokenizer & desired architecture
model = DeepSeekV3Model(vocab_size=tokenizer.vocab_size)

# ---------------------------
# 7. Define training arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./deepseek_v3_ckpt",
    per_device_train_batch_size=8,
    logging_steps=100,
    save_steps=500,
    num_train_epochs=1,
    evaluation_strategy="no",  # or "steps" if you have a validation set
    report_to="none"  # or "tensorboard", "wandb", etc.
)

# ---------------------------
# 8. Setup Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # streaming dataset
    data_collator=data_collator
)

# ---------------------------
# 9. Train the model
# ---------------------------
trainer.train() 