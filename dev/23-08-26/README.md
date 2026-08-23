Total model parameters (N) = 111,024,640

Training tokens per parameter (TPP) = r = 20 (chosen based on Chinchilla scaling laws)

Training tokens (D) = TPP × N = 2,220,492,800 (≈2.22B)

Max FLOPs - training compute budget (C) = 6ND ≈ 1.479 × 10¹⁸ FLOPs (≈1.48 exaFLOPs)

# Experiment Report

- Experiment ID: 
- Date:
- Description:
- Hypothesis:
- Git commit:
- Command:
- Random seed:

## Hardware and Software

- GPU model:
- GPU count:
- VRAM per GPU:
- CPU:
- System RAM:
- Python version:
- PyTorch version:
- CUDA version:
- Data type:

## Data

- Dataset name:
- Dataset version:
- License:
- Training shards:
- Validation shards:
- Number of documents:
- Raw characters:
- Training tokens:
- Validation tokens:
- Duplicate removal:
- Document-length statistics:

## Tokenizer

- Tokenizer type: Byte-level BPE
- Vocabulary size: 16,384
- Training characters: 
- Regex pattern:
- Special tokens:
- Characters per token:
- Bytes per token: 
- Tokenizer training time:

## Model

- Layers: 12
- Width: 768
- Attention heads: 144
- Head dimension: 64
- Context length: 1,024
- Vocabulary size: 16,384
- Parameter count: 111,024,640
- Dropout: 0.1

## Scaling and Compute Budget

- Training tokens per parameter (TPP): 20 
- Training token budget: 2,220,492,800 tokens (≈2.22B)
- Training compute budget: 1.479 × 10¹⁸ FLOPs (≈1.48 exaFLOPs)

## Optimization

- Optimizer: AdamW
- Peak learning rate:
- Minimum learning rate:
- Weight decay:
- Betas:
- Warmup steps:
- Schedule:
- Micro-batch size:
- Gradient accumulation steps:
- Global batch tokens:
- Gradient clipping:
- Total optimizer steps: 4,235
- Target tokens: 

## Runtime Measurements

| Step | Train loss | Validation loss | Validation BPB | Learning rate | Gradient norm | Tokens/sec | Step duration | GPU utilization | Peak VRAM | Elapsed time | ETA |
|-----:|-----------:|----------------:|---------------:|--------------:|--------------:|-----------:|--------------:|----------------:|----------:|-------------:|----:|
|      |            |                 |                |               |               |            |               |                 |           |              |     |

## Results

- Best validation BPB:
- Final validation BPB:
- Total tokens seen:
- Total training time:
- Estimated FLOPs:
- Checkpoint path:
- Sample outputs:
- Known failures:
- Conclusion:
- Next experiment: