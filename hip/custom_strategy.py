"""Custom PyTorch Lightning strategy with flexible optimizer state loading."""

import warnings
from typing import Any, Mapping
import torch


def _optimizer_to_device(optimizer, device):
    """Move optimizer state to device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def flexible_load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
    """Load optimizer states with flexible parameter group matching.
    
    This function can be used to monkey-patch a strategy's load_optimizer_state_dict method.
    Matches parameters by tensor shape and position to handle model changes gracefully.
    
    Strategy:
    1. If parameter groups match exactly -> use standard loading
    2. If mismatch -> match by position and tensor shape within each group
    3. Only load state for parameters that exist in both checkpoint and current model
    
    Args:
        self: The strategy instance (will be bound when using types.MethodType)
        checkpoint: The checkpoint dictionary containing optimizer_states
    """
    optimizer_states = checkpoint["optimizer_states"]
    
    for optimizer, opt_state in zip(self.optimizers, optimizer_states):
        # Check if parameter groups match
        if len(optimizer.param_groups) != len(opt_state["param_groups"]):
            warnings.warn(
                f"\nOptimizer parameter group mismatch:\n"
                f"  Checkpoint has {len(opt_state['param_groups'])} parameter groups\n"
                f"  Current optimizer has {len(optimizer.param_groups)} parameter groups",
                UserWarning
            )
            try:
                # Get access to actual parameter tensors to match by shape
                # Build mapping of id -> parameter tensor for current optimizer
                id_to_param = {}
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        # p is the actual parameter tensor
                        id_to_param[id(p)] = p
                
                matched_params = 0
                unmatched_params = 0
                shape_mismatches = 0
                
                # Create mapping from old to new param ids
                id_map = {}
                for i in range(min(len(opt_state["param_groups"]), len(optimizer.param_groups))):
                    old_params = opt_state["param_groups"][i]["params"]
                    new_param_tensors = optimizer.param_groups[i]["params"]
                    
                    # Match params by position within the group, verify by shape if possible
                    for j in range(min(len(old_params), len(new_param_tensors))):
                        old_id = old_params[j]
                        new_param = new_param_tensors[j]
                        new_id = id(new_param)
                        
                        # Check if shapes match (if we have state info for the old param)
                        old_state = opt_state.get("state", {})
                        if old_id in old_state and "exp_avg" in old_state[old_id]:
                            # Get shape from optimizer state (e.g., exp_avg for Adam)
                            old_shape = old_state[old_id]["exp_avg"].shape
                            new_shape = new_param.shape
                            if old_shape != new_shape:
                                warnings.warn(
                                    f"  Group {i}, param {j}: Shape mismatch "
                                    f"(checkpoint: {old_shape}, current: {new_shape}) - skipping",
                                    UserWarning
                                )
                                shape_mismatches += 1
                                continue
                        
                        id_map[old_id] = new_id
                        matched_params += 1
                    
                    # Count unmatched in this group
                    unmatched_params += abs(len(old_params) - len(new_param_tensors))
                
                # Count entire unmatched groups
                group_diff = abs(len(opt_state["param_groups"]) - len(optimizer.param_groups))
                if group_diff > 0:
                    # Count params in unmatched groups
                    if len(opt_state["param_groups"]) > len(optimizer.param_groups):
                        for i in range(len(optimizer.param_groups), len(opt_state["param_groups"])):
                            unmatched_params += len(opt_state["param_groups"][i]["params"])
                    else:
                        for i in range(len(opt_state["param_groups"]), len(optimizer.param_groups)):
                            unmatched_params += len(optimizer.param_groups[i]["params"])
                
                warnings.warn(
                    f"  Matched {matched_params} parameters\n"
                    f"  Unmatched: {unmatched_params} parameters\n"
                    f"  Shape mismatches: {shape_mismatches} parameters",
                    UserWarning
                )
                
                # Remap the optimizer state for matched parameters only
                old_state = opt_state.get("state", {})
                for old_id, new_id in id_map.items():
                    if old_id in old_state:
                        optimizer.state[new_id] = old_state[old_id]
                
                # Update hyperparameters for matched parameter groups
                for i in range(min(len(opt_state["param_groups"]), len(optimizer.param_groups))):
                    loaded_group = opt_state["param_groups"][i]
                    current_group = optimizer.param_groups[i]
                    
                    # Update hyperparameters but keep current params list
                    for key in loaded_group:
                        if key != 'params':
                            current_group[key] = loaded_group[key]
            except Exception as e:
                warnings.warn(
                    f"Error loading optimizer state: {e}"
                    "Not loading optimizer state",
                    UserWarning
                )
                continue
        else:
            # Standard loading when groups match
            optimizer.load_state_dict(opt_state)
        
        _optimizer_to_device(optimizer, self.root_device)

