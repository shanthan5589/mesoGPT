from importlib import import_module


ATTENTION_REGISTRY = {
    "manual": "002",
    "sdpa": "003",
}

MODEL_REGISTRY = {
    "002": "mesoGPT.models.model",
    "003": "mesoGPT.models.model",
}


def build_model(model_config):
    try:
        attention = model_config.get("attention")
        exp_no = ATTENTION_REGISTRY[attention]
        module_path = MODEL_REGISTRY[exp_no]
    except KeyError as error:
        available = ", ".join(ATTENTION_REGISTRY.keys())

        raise ValueError(
            f"Invalid attention type '{attention}'. "
            f"Available attention types: {available}"
        ) from error

    module = import_module(module_path)

    GPT = module.GPT
    GPTConfig = getattr(module, "GPTConfig", None)

    config = GPTConfig(**model_config)

    return GPT(config=config)