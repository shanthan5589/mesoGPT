Total model parameters (N) = 111,024,640
Total_trainable_parameters = 111,024,640

Total model params after weight tying = 98,441,728  (Reduction of 12,582,912 params)
Reduction in parameters due to weight tying = 11.33%

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

- Dataset name: karpathy/climbmix-400b-shuffle
- Dataset version: - 
- License: MIT
- Number of documents per shard: 86,016 (86K)
- Raw Characters per shard: ~252,606,075.5 (252.6M)
- Number of tokens per shard: ~57,582,082 (57.6M)
- Duplicate removal:
- Document-length statistics:

## Tokenizer

- Tokenizer type: Byte-level BPE
- Vocabulary size: 16,384
- Training characters: 450M
- Regex pattern: 
- Special tokens: 
- Characters per token: 4.383
- Bytes per token: 4.390
- Tokenizer training time: 32.75 seconds
- Physical memory used: ~493.1 MB (This ram is used by the python script to train the tokenizer.)

## Model

- Parameter count: 98,441,728 (Used weight tying)
- Attention heads: 12
- Layers: 12
- Width: 768
- Context length: 1,024
- Vocabulary size: 16,384
- Dropout: 0.1
- Head dimension: 64


## Scaling and Compute Budget

<!-- Before weight tying:
- Training token budget: 2,220,492,800 tokens (≈2.22B)
- Training compute budget: 1.479 × 10¹⁸ FLOPs (≈1.48 exaFLOPs) -->

Weight tying reduced the training token budget by 11.33% and the training compute budget by 21.43%.

- Training tokens per parameter (TPP): 20 
- Training token budget: 1,968,834,560 (~1.97B)  
- Training compute budget: 1.162 x 10¹⁸ FLOPs (≈1.16 exaFLOPs)
- Training shards: 36

- Validation tokens: 57,582,082 (~57.6M)
- Validation shards: 1

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