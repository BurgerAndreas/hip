from pathlib import Path
from pprint import pprint

import torch

from hip.training_module import PotentialModule


def print_section(title: str, payload):
    print(f"\n{title}")
    pprint(payload)


def discover_device_count(ckpt_dict):
    candidates = []
    hyper_parameters = ckpt_dict.get("hyper_parameters", {})
    training_config = hyper_parameters.get("training_config")
    if isinstance(training_config, dict):
        candidates.append(training_config)
    pltrainer_config = hyper_parameters.get("pltrainer")
    if isinstance(pltrainer_config, dict):
        candidates.append(pltrainer_config)
    candidates.append(ckpt_dict)

    keys = ("devices", "gpus", "num_devices", "world_size")
    for section in candidates:
        for key in keys:
            if key in section:
                return section[key]
    return None


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    # ckpt_path = project_root / "ckpt" / "left.ckpt"
    # ckpt_path = project_root / "ckpt" / "eqv2_orig.ckpt"
    # ckpt_path = project_root / "ckpt" / "eqv2.ckpt"
    ckpt_path = project_root / "ckpt" / "left-df.ckpt"
    # ckpt_path = project_root / "ckpt" / "alpha.ckpt"
    print(f"Loading checkpoint from {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module = PotentialModule.load_from_checkpoint(str(ckpt_path), strict=False)

    print_section("Model configuration", module.model_config)
    print_section("Optimizer configuration", module.optimizer_config)
    print_section("Training configuration", module.training_config)
    print_section(
        "Checkpoint metadata",
        {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "device_count": discover_device_count(checkpoint),
        },
    )


if __name__ == "__main__":
    main()
