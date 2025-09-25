import os
import omegaconf
import numpy as np
import torch
from pathlib import Path
import hashlib

# e.g. inference kwargs
IGNORE_OVERRIDES = [
    "wandb",
    "do_wandb",
    "use_wandb",
    "wandb_run_id",
    "ckpt_resume_auto",
]

# some stuff is not relevant for the checkpoint
# allows to load checkpoint with the same name
IGNORE_OVERRIDES_CHECKPOINT = [
    "ckpt_resume_auto",
    "ckpt_model_path",
    "ckpt_trainer_path",
    "eval_hessian_method",
    "eval_max_samples",
    "eval_config_path",
    "eval_wandb_run_id",
    "num_workers",
]

REPLACE = {
    "+": "",
    "experiment=": "",
    "experiment": "",
    "training.lr_schedule_type=": "lr=",
    "training.eigen_loss": "el",
    "training.": "",
    "model.": "",
    "pltrainer.": "",
    "hessian_": "",
    "loss_type_vec=": "lossvec=",
    "loss_type=": "loss=",
    "trgt=hessian": "",
}

REPLACE_HUMAN = {
    "lr_schedule_config.step_size": "lr_step",
    "lr_schedule_type": "lr",
    "lr_schedule_config.": "lr_",
    "luca8w10only": "luca8 + luca/10",
    "luca8w10": "luca8 + luca/10 + m",
    "overfit100": "",
    "alldata": "",
    "preset=a": "wa bz=128",
    "preset=b": "Luca8",
    "preset=": "",
    "hesspred_": "",
    "num_layers_hessian": "l",
    "symmetric": "sym",
    "bz=128": "",
}


def name_from_config(args: omegaconf.DictConfig, is_checkpoint_name=False) -> str:
    """Generate a name for the model based on the config.
    Name is intended to be used as a file name for saving checkpoints and outputs.
    """
    try:
        # model name format:
        mname = ""
        # override format: 'pretrain_dataset=bridge,steps=10,use_wandb=False'
        override_names = ""
        # print(f'Overrides: {args.override_dirname}')
        if args.override_dirname:
            for arg in args.override_dirname.split(","):
                # make sure we ignore some overrides
                if np.any([ignore in arg for ignore in IGNORE_OVERRIDES]):
                    continue
                # ignore some more overrides for checkpoint names
                if is_checkpoint_name:
                    if np.any(
                        [ignore in arg for ignore in IGNORE_OVERRIDES_CHECKPOINT]
                    ):
                        continue
                override_names += " " + arg
    except Exception as error:
        print("\nname_from_config() failed:", error)
        print("args:", args)
        raise error
    for key, value in REPLACE.items():
        override_names = override_names.replace(key, value)
    if is_checkpoint_name or len(override_names) > 40:
        # Use a short, stable hash for checkpoint base name
        raw = override_names.strip()
        override_names = f"ck-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"
    else:
        # Make wandb name human readable
        for key, value in REPLACE_HUMAN.items():
            override_names = override_names.replace(key, value)

    # logger.info("name_from_config() mname: %s, override_names: %s", mname, override_names)
    _name = mname + override_names
    print(f"Name{' checkpoint' if is_checkpoint_name else ''}: {_name}")
    return _name


def set_gpu_name(args):
    """Set wandb.run.name."""
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_name = (
            gpu_name.replace("NVIDIA", "").replace("GeForce", "").replace(" ", "")
        )
        args.gpu_name = gpu_name
    except:
        pass
    return args


def find_latest_checkpoint(base_checkpoint_name: str, project: str) -> str:
    """
    Find the latest checkpoint file from directories matching the base name pattern.

    Args:
        base_checkpoint_name: Base name without slurm_job_id and timestamp
        project: Project name

    Returns:
        Path to the latest checkpoint file, or None if not found
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_base_dir = Path(f"{root_dir}/checkpoint/{project}")
    if not checkpoint_base_dir.exists():
        print(f"Checkpoint directory {checkpoint_base_dir} does not exist")
        return None

    # Find all directories that start with the base name
    pattern = f"{base_checkpoint_name}-*"
    matching_dirs = list(checkpoint_base_dir.glob(pattern))

    if not matching_dirs:
        print(f"No existing checkpoint directories found matching pattern: {pattern}")
        return None

    print(
        f"Found {len(matching_dirs)} matching checkpoint directories: {[d.name for d in matching_dirs]}"
    )

    # Find all checkpoint files in all matching directories
    all_checkpoints = []
    for dir_path in matching_dirs:
        ckpt_files = list(dir_path.glob("*.ckpt"))
        for ckpt_file in ckpt_files:
            all_checkpoints.append(ckpt_file)

    if not all_checkpoints:
        print("No checkpoint files found in matching directories")
        return None

    # Sort by modification time, newest first
    all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_checkpoint = all_checkpoints[0]

    print(f"Found {len(all_checkpoints)} checkpoint files")
    print(f"Latest checkpoint: {latest_checkpoint}")

    return str(latest_checkpoint)


if __name__ == "__main__":
    print(find_latest_checkpoint("horm", "horm"))
