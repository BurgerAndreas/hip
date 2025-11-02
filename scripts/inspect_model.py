"""Script to initialize just the model and print its modules and weight shapes.

This script loads the model configuration and initializes the model without training setup.
"""

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hip.training_module import PotentialModule
from hip.path_config import CHECKPOINT_PATH_EQUIFORMER_HORM
from contextlib import redirect_stdout


def print_model_info(model):
    """Print detailed information about the model structure and weights."""
    print("=" * 80)
    print("MODEL STRUCTURE")
    print("=" * 80)

    # Print model architecture
    print(f"Model type: {type(model).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print()

    # Print module hierarchy
    print("MODULE HIERARCHY:")
    print("-" * 40)
    for name, module in model.named_modules():
        if name:  # Skip the root module
            indent = "  " * (name.count(".") + 1)
            print(f"{indent}{name}: {type(module).__name__}")

    print("\n" + "=" * 80)
    print("WEIGHT SHAPES")
    print("=" * 80)

    # Print weight shapes
    for name, param in model.named_parameters():
        print(f"{name}: {list(param.shape)} ({param.numel():,} parameters)")

    print("\n" + "=" * 80)
    print("MODULE DETAILS")
    print("=" * 80)

    # Print detailed module information
    for name, module in model.named_modules():
        if name:  # Skip the root module
            if name.startswith("blocks.0."):  # only print first transformer layer
                print(f"\n{name} ({type(module).__name__}):")
                if hasattr(module, "weight") and module.weight is not None:
                    print(f"  Weight shape: {list(module.weight.shape)}")
                if hasattr(module, "bias") and module.bias is not None:
                    print(f"  Bias shape: {list(module.bias.shape)}")

                # Print other attributes that might be interesting
                interesting_attrs = [
                    "in_features",
                    "out_features",
                    "num_heads",
                    "hidden_size",
                    "embed_dim",
                ]
                for attr in interesting_attrs:
                    if hasattr(module, attr):
                        print(f"  {attr}: {getattr(module, attr)}")


def setup_model(cfg: DictConfig):
    """Initialize the model with the given configuration."""
    print("Initializing model...")

    # Get configs
    model_config = cfg.model
    optimizer_config = dict(cfg.optimizer)
    training_config = dict(cfg.training)

    # Initialize the potential module
    pm = eval(cfg.potential_module_class)(
        model_config, optimizer_config, training_config
    )

    # Load checkpoint if specified
    if cfg.ckpt_model_path in [None, "None", "null"]:
        print(f"Not loading model checkpoint from {cfg.ckpt_model_path}")
    elif cfg.ckpt_model_path == "horm":
        print("Loading HORM checkpoint...")
        ckpt = torch.load(
            CHECKPOINT_PATH_EQUIFORMER_HORM, map_location="cpu", weights_only=True
        )
        # keys all start with `potential.`
        state_dict = {
            k.replace("potential.", ""): v for k, v in ckpt["state_dict"].items()
        }
        pm.potential.load_state_dict(state_dict, strict=False)
        print("HORM checkpoint loaded successfully")
    elif os.path.exists(cfg.ckpt_model_path):
        print(f"Loading checkpoint from {cfg.ckpt_model_path}...")
        pm = eval(cfg.potential_module_class).load_from_checkpoint(
            cfg.ckpt_model_path, strict=False
        )
        print("Checkpoint loaded successfully")
    else:
        print(f"Not loading model checkpoint from {cfg.ckpt_model_path}")

    print(f"{cfg.potential_module_class} initialized")
    return pm


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main function to initialize and inspect the model."""
    torch.set_float32_matmul_precision("high")

    # Initialize the model
    pm = setup_model(cfg)

    # Ensure output directory exists and dump all prints to a txt file
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inspect_model.txt")
    with open(out_path, "w") as f, redirect_stdout(f):
        print(pm.potential)
        print_model_info(pm.potential)


if __name__ == "__main__":
    """Try:
    python scripts/inspect_model.py
    python scripts/inspect_model.py ckpt_model_path=horm
    python scripts/inspect_model.py ckpt_model_path=/path/to/checkpoint.ckpt
    """
    main()
