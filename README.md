# DeepSeek Language Model Implementation

A PyTorch implementation of a modified DeepSeek architecture with Mixture of Experts (MoE) and latent attention mechanisms. This project aims to create an efficient and scalable language model while maintaining strong performance.


## Architecture Overview

The model is based on the LLaMA architecture with several key modifications:

- **Base Configuration**:
  - Vocabulary Size: 49,152 tokens
  - Hidden Size: 128 dimensions
  - Number of Layers: 4
  - Attention Heads: 8
  - Context Length: 2048 tokens

### Key Components

1. **Latent Attention Mechanism**
   - Implements a compressed attention space using projection matrices
   - Uses rotary positional embeddings (RoPE)
   - Components:
     - KV and Q projections to latent space
     - Decompression projections back to model dimension
     - Separate rotary embeddings for keys and queries

2. **Mixture of Experts (MoE)**
   - 1 shared expert + 7 routed experts
   - Router-based dynamic expert selection
   - Components per Expert:
     - Gate projection
     - Up projection
     - Down projection
     - SiLU activation

3. **Normalization and Residual Connections**
   - Uses RMSNorm for layer normalization
   - Implements skip connections throughout

## Implementation Details
The model is trained using the following configuration and setup:

1. **Training Configuration**:
   - Batch Size: 4 with gradient accumulation steps of 8
   - Sequence Length: 256 tokens
   - Learning Rate: 1e-4
   - Training Epochs: 5
   - Warmup Steps: 1000
   - Gradient Clipping: 0.5

2. **Text Generation**:
   - Implements temperature-based sampling
   - Uses top-k filtering for token selection
   - Supports variable length generation
   - Handles special tokens appropriately

3. **Optimization**:
   - Uses AdamW optimizer
   - Implements learning rate scheduling
   - Gradient accumulation for effective larger batch sizes
   - Automatic mixed precision training

4. **Checkpointing**:
   - Regular model state saving
   - Saves optimizer and scheduler states
   - Supports training resumption
   - Maintains best model checkpoints

5. **Monitoring**:
   - Integration with Weights & Biases
   - Tracks training metrics
   - Generates sample text periodically
   - Monitors resource usage

### Model Architecture

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 128)
    (layers): ModuleList(
      (0-3): 4 x DeepSeekBlock(
        (self_attn): MultiHeadLatentAttention(
          (kv_proj_d): Linear(in_features=128, out_features=16, bias=False)
          (q_proj_d): Linear(in_features=128, out_features=16, bias=False)
          (k_proj_u): Linear(in_features=16, out_features=64, bias=False)
          (q_proj_u): Linear(in_features=16, out_features=64, bias=False)
          (v_proj_u): Linear(in_features=16, out_features=128, bias=False)
          (rope_k): Linear(in_features=128, out_features=64, bias=False)
          (rope_q): Linear(in_features=16, out_features=64, bias=False)
          (o_proj): Linear(in_features=128, out_features=128, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (moe): DeepSeekMoE(
            (shared_experts): ModuleList(
              (0): DeepSeekExpert(
                (gate_proj): Linear(in_features=128, out_features=256, bias=False)
                (up_proj): Linear(in_features=128, out_features=256, bias=False)
                (down_proj): Linear(in_features=256, out_features=128, bias=False)
                (act): SiLU()
              )
            )
            (routed_experts): ModuleList(
              (0-6): 7 x DeepSeekExpert(
                (gate_proj): Linear(in_features=128, out_features=256, bias=False)
                (up_proj): Linear(in_features=128, out_features=256, bias=False)
                (down_proj): Linear(in_features=256, out_features=128, bias=False)
                (act): SiLU()
              )
            )
            (router): Linear(in_features=128, out_features=7, bias=False)
          )
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=128, out_features=49152, bias=False)
)

### Key Features
1. **Latent Attention**:
   - Compression ratio: 8
   - Rotary embeddings with custom base
   - Efficient key-value processing

2. **MoE Implementation**:
   - Dynamic routing with load balancing
   - Auxiliary loss for router training
   - Expert capacity: 64 tokens
   - Top-k routing: k=4

3. **Training Optimizations**:
   - Gradient checkpointing
   - Mixed precision training
   - Efficient memory management


### Training Metrics
- Training Time: ~2 hours on CUDA GPU
- Total Parameters: 1,346,471,936

### Expert Utilization
- Layer 0: Balanced distribution across experts
- Layer 1: Specialized routing patterns
- Layer 2-3: Dynamic expert selection

## Setup and Usage

1. **Installation**:
```bash
git clone https://kn0wthing@github.com/kn0wthing/session_14.git
cd session_14
pip install -r requirements.txt
```

2. **Training**:
```bash
python train.py
```

3. **Generation**:
```python
from model import get_model
from deepseek_config import DeepSeekConfig

config = DeepSeekConfig()
model = get_model(config)
```

## Requirements
```text
torch>=2.0.0
numpy>=1.24.0
transformers>=4.30.0
accelerate>=0.20.0
wandb>=0.15.0
```

## License
MIT

## Acknowledgments
- Based on the DeepSeek architecture
- Inspired by LLaMA implementation
- Uses Hugging Face Transformers

```

