# Contributing

Contributions are welcome :)

## Development setup

Please follow the installation instructions in `README.md` to set up a development environment.

## Repository organization

Reusable implementation belongs under `src/mesoGPT/`:

- `dataset/`: dataset downloading, reading, and dataloading
- `models/`: model architecture and model construction
- `tokenizer/`: tokenizer implementation
- `training/`: shared tokenizer and model training code
- `utils/`: generation, statistics, and analysis utilities

Experiment specific reports, figures, and experiment-specific analysis belong under
`experiments/exp_NNN/`. Experiment launch commands belong under `launchers/`,
and generated run artifacts belong under `outputs/`.

Do not create separate copies of the model or trainer for each experiment.
Add reusable behavior to the shared implementation and select it through
configuration or command-line arguments.

## Making changes

1. Create a focused branch.
2. Keep each change limited to one clear purpose.
3. Preserve existing behavior unless the change intentionally modifies it.
4. Update documentation when commands, paths, metadata, or behavior change.
5. Include validation appropriate to the change.

## Adding an experiment

Use the next available experiment number:

```text
experiments/exp_NNN/
```

An experiment should normally include:

```text
experiments/exp_NNN/
├── README.md
├── assets/
└── analysis scripts, if needed
```

Experiment-specific hyperparameters should live in its launcher or
configuration. Reusable model, dataset, tokenizer, profiling, and training
logic should remain under `src/mesoGPT/`.

Avoid duplicating an existing model or trainer just to change one feature.

## Checkpoints and generated files

Do not commit downloaded datasets, virtual environments or python caches.

The published model checkpoint is managed through Git LFS:

```text
outputs/**/model.pt
```

Before committing a checkpoint, verify that Git LFS recognizes it:

```bash
git check-attr filter -- path/to/model.pt
git lfs ls-files
```

The checkpoint should report `filter: lfs`.

Unnecessary artifacts should remain ignored unless there is a specific
reason to publish them.

Small result files such as metadata, logs, profiler summaries, and GPU metrics
may be committed when they are relevant to a documented experiment.

## Checkpoint compatibility

Changes to GPTConfig, model state-dictionary keys, or checkpoint metadata can
break inference with existing checkpoints.

When changing these areas:
- preserve compatibility
- document intentional format changes


## Documentation

Update the relevant README when changing:

- repository paths
- command-line arguments
- experiment launch commands
- model configuration
- checkpoint metadata
- installation requirements
- reported results

Commands in documentation should be runnable from the repository root.

## Pull requests

A pull request should include:

- a concise explanation of the change
- why the change is needed
- the validation performed
- any behavior or compatibility implications
- updated documentation where applicable

Before submitting, check:

- [ ] Reusable code is under `src/mesoGPT/`.
- [ ] Experiment-specific material is under `experiments/`.
- [ ] No dataset, cache, or trace files were added.
- [ ] Large checkpoints are managed by Git LFS.
- [ ] Existing checkpoint compatibility was considered.
- [ ] Relevant commands and documentation were updated.
- [ ] Validation results are included in the pull request.
- [ ] No existing weights are manipulated or deleted.

## Reporting problems

When reporting a bug, include:

- operating system
- Python version
- PyTorch and CUDA versions
- CPU or GPU model
- complete error message
- relevant model or experiment configuration