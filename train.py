import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
import wandb
import os, sys
import time
from deepseek_config import DeepSeekConfig, LatentAttentionConfig
from typing import List, Dict, Any, Optional, Tuple
from model import get_model

wandb.init(project="smollm-training", name="llama-smollm-corpus", mode="offline")


class TrainingConfig:
    BATCH_SIZE: int = 4
    ACCUMULATION_STEPS: int = 8
    SEQ_LEN: int = 256
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 5
    WARMUP_STEPS: int = 1000
    GRADIENT_CLIP_VAL: float = 0.5
    CHECKPOINT_DIR: str = "checkpoints"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

os.makedirs(TrainingConfig.CHECKPOINT_DIR, exist_ok=True)


def generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = TrainingConfig.DEVICE
) -> str:
    """Generate text using the model."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    path: str
) -> None:
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
) -> Tuple[int, int]:
    """Load model checkpoint."""
    if os.path.exists(path):
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"], checkpoint["step"]
    return 0, 0


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.resize_token_embeddings(len(tokenizer))

dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True, split="train"
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=TrainingConfig.SEQ_LEN, padding="max_length"
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)


def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor(
        [item["attention_mask"] for item in batch], dtype=torch.long
    )
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_loader = DataLoader(
    tokenized_dataset, batch_size=TrainingConfig.BATCH_SIZE, collate_fn=collate_fn
)

# Initialize model, optimizer, and scheduler
config = DeepSeekConfig()
model = get_model(config)
model.to(TrainingConfig.DEVICE)

# Print model parameters
total_params = count_parameters(model)
print(f"\nModel Statistics:")
print(f"Total Parameters: {total_params:,}")
print(f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB")  # Assuming float32 (4 bytes)
print(f"Device: {TrainingConfig.DEVICE}")
print(f"Batch Size: {TrainingConfig.BATCH_SIZE}")
print(f"Accumulation Steps: {TrainingConfig.ACCUMULATION_STEPS}")
print(f"Sequence Length: {TrainingConfig.SEQ_LEN}")
print(f"Learning Rate: {TrainingConfig.LEARNING_RATE}")
print("-" * 50 + "\n")


optimizer = AdamW(model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=TrainingConfig.LEARNING_RATE,
    total_steps=20000,
    pct_start=0.1,
    anneal_strategy="cos",
    cycle_momentum=False,
)

# Load checkpoint if exists
start_epoch, global_step = load_checkpoint(
    f"{TrainingConfig.CHECKPOINT_DIR}/latest_checkpoint.pt",
    model, 
    optimizer, 
    lr_scheduler
)

# Sample prompts for evaluation
sample_prompts = [
    "Particles in Action. Have you ever imagined being able to see tiny particles that zoom around us at incredible speeds? Welcome to the world of particle physics! ",
    "Developing number sense is a critical aspect of mathematics education that involves helping students understand numbers, their relationships, and operations involving them. ",
    "All parts of the coriander plant are edible - including its leaves, its fruits, its seeds and its roots. However, the fresh leaves and the dried seeds score over the other two, and are the most commonly employed in cooking. ",
    "There are several foods that can help boost your metabolism and promote calorie burning, thanks to their unique nutritional profiles. ",
    "Are you looking for vegan sandwich recipes? We've rounded up 21 of our favorite vegan sandwich ideas that you will want to make right now. ",
]

model.train()
try:
    for epoch in range(start_epoch, TrainingConfig.EPOCHS):
        print(f"Epoch {epoch + 1}/{TrainingConfig.EPOCHS}")
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for step, batch in enumerate(train_loader, start=global_step):
            # Move batch to device
            input_ids = batch["input_ids"].to(TrainingConfig.DEVICE)
            attention_mask = batch["attention_mask"].to(TrainingConfig.DEVICE)
            labels = batch["labels"].to(TrainingConfig.DEVICE)

            # Calculate expert load and update routing bias every 100 steps
            if step % 100 == 0:
                with torch.no_grad():
                    # Get initial hidden states
                    hidden_states = model.model.embed_tokens(input_ids)
                    
                    # Update expert load for each layer
                    for layer_idx, layer in enumerate(model.model.layers):
                        # Initialize expert load tensor
                        expert_load = torch.zeros(layer.mlp.moe.num_routed_experts, device=TrainingConfig.DEVICE)
                        
                        # Get routing probabilities
                        routing_logits = layer.mlp.moe.router(hidden_states)
                        routing_probs = torch.sigmoid(routing_logits + layer.mlp.moe.routing_bias)
                        
                        # Get top-k expert indices
                        _, indices = torch.topk(routing_probs, k=layer.mlp.moe.top_k, dim=-1)
                        
                        # Calculate load for each expert
                        for i in range(layer.mlp.moe.num_routed_experts):
                            expert_load[i] = (indices == i).sum().float()
                        
                        # Normalize the expert load
                        total_tokens = input_ids.size(0) * input_ids.size(1)
                        expert_load = expert_load / (total_tokens * layer.mlp.moe.top_k)
                        
                        # Update routing bias
                        layer.mlp.moe.update_bias_terms(expert_load)
                        
                        # Log expert utilization
                        for i, load in enumerate(expert_load):
                            wandb.log({
                                f"layer_{layer_idx}_expert_{i}_load": load.item(),
                                "step": step
                            })
                        
                        # Process hidden states through the layer for next iteration
                        hidden_states = layer.input_layernorm(hidden_states)
                        hidden_states = layer.self_attn(hidden_states)
                        hidden_states = layer.post_attention_layernorm(hidden_states)

            # Forward pass
            outputs = model(input_ids)
            logits = outputs.view(-1, tokenizer.vocab_size)

            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                logits, labels.view(-1), label_smoothing=0.1
            )
            
            # Scale loss by accumulation steps
            scaled_loss = loss / TrainingConfig.ACCUMULATION_STEPS
            
            # Backward pass
            scaled_loss.backward()

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Logging (show unscaled loss for monitoring)
            if step % 10 == 0:
                print(
                    f"Step {step}, Loss: {loss.item():.4f}, "  # Original loss
                    f"Scaled Loss: {scaled_loss.item():.4f}, "  # Scaled loss
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}, "
                    f"Accumulation Step: {(step + 1) % TrainingConfig.ACCUMULATION_STEPS}/{TrainingConfig.ACCUMULATION_STEPS}, "
                    f"Current Time: {current_time} "
                )
                wandb.log({
                    "loss": loss.item(),  # Log original loss
                    "scaled_loss": scaled_loss.item(),  # Log scaled loss
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": step,
                    "epoch": epoch,
                })

            # Update weights after accumulation steps
            if (step + 1) % TrainingConfig.ACCUMULATION_STEPS == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRADIENT_CLIP_VAL)
                
                # Optimizer step
                optimizer.step()
                
                # Update learning rate
                lr_scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # print(f"\nCompleted gradient accumulation step {step + 1}")

            # Save checkpoint every 100 steps (actual updates)
            if step % 100 == 0 and step != 0:
                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    step,
                    loss.item(),  # Save original loss
                    f"{TrainingConfig.CHECKPOINT_DIR}/latest_checkpoint.pt",
                )

            # Generate sample text every 500 steps
            if step != 0 and step % 500 == 0:
                model.eval()
                print("\n=== Generating Sample Texts ===")
                
                # Save model state for generation
                generation_state = model.state_dict()
                
                for temp in [1.0]:  # Added back temperature variation
                    for prompt in sample_prompts:
                        generated = generate_text(
                            model,
                            tokenizer,
                            prompt,
                            temperature=temp,
                            max_length=100,
                        )
                        print(f"\nPrompt: {prompt}")
                        print(f"Temperature: {temp}")
                        print(f"Generated: {generated}")
                        wandb.log({
                            f"generated_text_temp_{temp}_{prompt[:20]}": wandb.Html(generated),
                            "step": step
                        })
                
                print("\n=== End of Samples ===\n")
                model.train()

        # Save epoch checkpoint
        save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            epoch,
            step,
            loss.item(),  # Save original loss
            f"{TrainingConfig.CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pt",
        )

except KeyboardInterrupt:
    print("\nTraining interrupted! Saving checkpoint...")
    save_checkpoint(
        model,
        optimizer,
        lr_scheduler,
        epoch,
        step,
        loss.item(),  # Save original loss
        f"{TrainingConfig.CHECKPOINT_DIR}/interrupted_checkpoint.pt",
    )

print("Training complete!")
wandb.finish()