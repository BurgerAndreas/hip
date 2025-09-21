"""Utilities for finding and managing checkpoints."""

import os
import glob
from typing import List, Optional, Dict
from pathlib import Path


def find_checkpoint_by_runid(
    runid: str, project_name: str = None, checkpoint_base_dir: str = "checkpoint"
) -> List[str]:
    """
    Find checkpoint directories containing a specific run ID.

    Args:
        runid: The run ID to search for
        project_name: Optional project name to narrow the search
        checkpoint_base_dir: Base directory where checkpoints are stored

    Returns:
        List of checkpoint directory paths containing the run ID
    """
    checkpoint_dirs = []

    if project_name:
        # Search within specific project
        search_pattern = os.path.join(checkpoint_base_dir, project_name, f"*{runid}*")
    else:
        # Search across all projects
        search_pattern = os.path.join(checkpoint_base_dir, "*", f"*{runid}*")

    matches = glob.glob(search_pattern)

    # Filter to only return directories
    checkpoint_dirs = [match for match in matches if os.path.isdir(match)]

    return checkpoint_dirs


def find_checkpoint_files_by_runid(
    runid: str, project_name: str = None, checkpoint_base_dir: str = "checkpoint"
) -> Dict[str, List[str]]:
    """
    Find all checkpoint files (.ckpt) within directories containing a specific run ID.

    Args:
        runid: The run ID to search for
        project_name: Optional project name to narrow the search
        checkpoint_base_dir: Base directory where checkpoints are stored

    Returns:
        Dictionary mapping checkpoint directory paths to lists of .ckpt files
    """
    checkpoint_dirs = find_checkpoint_by_runid(runid, project_name, checkpoint_base_dir)

    result = {}
    for checkpoint_dir in checkpoint_dirs:
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if ckpt_files:
            result[checkpoint_dir] = sorted(ckpt_files)

    return result


def get_best_checkpoint_by_runid(
    runid: str,
    project_name: str = None,
    checkpoint_base_dir: str = "checkpoint",
    metric: str = "val-totloss",
    best_is_min: bool = True,
) -> Optional[str]:
    """
    Find the best checkpoint file based on a metric for a specific run ID.

    Args:
        runid: The run ID to search for
        project_name: Optional project name to narrow the search
        checkpoint_base_dir: Base directory where checkpoints are stored
        metric: Metric name to evaluate (e.g., 'val-totloss', 'val-MAE_E')
        best_is_min: Whether lower values are better for the metric

    Returns:
        Path to the best checkpoint file, or None if not found
    """
    checkpoint_files = find_checkpoint_files_by_runid(
        runid, project_name, checkpoint_base_dir
    )

    if not checkpoint_files:
        return None

    best_file = None
    best_value = float("inf") if best_is_min else float("-inf")

    for checkpoint_dir, files in checkpoint_files.items():
        for file_path in files:
            filename = os.path.basename(file_path)

            # Parse the metric value from filename
            # Format: ff-{epoch:03d}-{val-totloss:.4f}-{val-MAE_E:.4f}-{val-MAE_F:.4f}.ckpt
            try:
                parts = filename.replace(".ckpt", "").split("-")
                if len(parts) >= 5:  # ff, epoch, val-totloss, val-MAE_E, val-MAE_F
                    if metric == "val-totloss" and len(parts) >= 3:
                        value = float(parts[2])
                    elif metric == "val-MAE_E" and len(parts) >= 4:
                        value = float(parts[3])
                    elif metric == "val-MAE_F" and len(parts) >= 5:
                        value = float(parts[4])
                    else:
                        continue

                    if (best_is_min and value < best_value) or (
                        not best_is_min and value > best_value
                    ):
                        best_value = value
                        best_file = file_path
            except (ValueError, IndexError):
                continue

    return best_file
