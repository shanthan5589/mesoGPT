# mesoGPT

<p align="center">
  <img src="assets/banner.jpg" alt="milky way" width="300">
</p>


mesoGPT is a successor of microGPT. This model is intended to train on more compute and data than its predecessor. 

Experiments:
- [exp_001](experiments/exp_001/README.md)
- [exp_002](experiments/exp_002/README.md)


```
mesoGPT/
├── .gitignore
├── .vscode/
│
├── mesoGPT/
│   ├── __init__.py
│   ├── common.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── dataloader.py
│   └── model.py
│
├── scripts/
│   ├── __init__.py
│   ├── tok_train.py
│   ├── base_train.py
│   ├── sample.py
│   └── count_params.py
```