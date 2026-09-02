from importlib import import_module


MODEL_REGISTRY = {
    "002": "experiments.exp_002.model",
    "003": "experiments.exp_003.model",
}


def build_model(exp_no, model_config):
    try:
        module_path = MODEL_REGISTRY[exp_no]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_REGISTRY))

        raise ValueError(
            f"Unknown model type: {exp_no!r}. "
            f"Available model types: {available}"
        ) from error

    module = import_module(module_path)

    GPT = module.GPT
    GPTConfig = getattr(module, "GPTConfig", None)

    config = GPTConfig(**model_config)

    return GPT(config=config)