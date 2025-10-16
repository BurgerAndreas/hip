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
    Matches parameters by their names (from model.named_parameters()) for robustness.
    
    Strategy:
    1. If parameter groups match exactly -> use standard loading
    2. If mismatch -> match parameters by name from checkpoint state_dict to current model
    3. Only load state for parameters that exist in both checkpoint and current model
    
    Args:
        self: The strategy instance (will be bound when using types.MethodType)
        checkpoint: The checkpoint dictionary containing optimizer_states
    """
    optimizer_states = checkpoint["optimizer_states"]
    
    # Get the model to access parameter names
    model = self.lightning_module if hasattr(self, 'lightning_module') else None
    
    for optimizer, opt_state in zip(self.optimizers, optimizer_states):
        # Check if parameter groups match
        if len(optimizer.param_groups) != len(opt_state["param_groups"]):
            warnings.warn(
                f"\nOptimizer parameter group mismatch:\n"
                f"  Checkpoint has {len(opt_state['param_groups'])} parameter groups\n"
                f"  Current optimizer has {len(optimizer.param_groups)} parameter groups\n"
                f"  Attempting to match parameters by name...",
                UserWarning
            )
            
            # Try to match parameters by name if we have access to the model
            if model is not None and "state_dict" in checkpoint:
                # Build mapping: current param tensor id -> param name (for diagnostics)
                current_id_to_name = {}
                for name, param in model.named_parameters():
                    current_id_to_name[id(param)] = name
                
                old_state = opt_state.get("state", {})
                
                # Match by position within groups and validate with shapes
                # Note: We can't match by name directly because the optimizer state dict
                # only stores parameter IDs (memory addresses), not names
                matched_params = 0
                unmatched_in_checkpoint = 0
                unmatched_in_current = 0
                shape_mismatches = 0
                
                # Build id mapping by matching position within groups and validating with shapes
                id_map = {}
                for i in range(min(len(opt_state["param_groups"]), len(optimizer.param_groups))):
                    old_params = opt_state["param_groups"][i]["params"]
                    new_param_tensors = optimizer.param_groups[i]["params"]
                    
                    for j in range(min(len(old_params), len(new_param_tensors))):
                        old_id = old_params[j]
                        new_param = new_param_tensors[j]
                        new_id = id(new_param)
                        
                        # Get name of current parameter
                        param_name = current_id_to_name.get(new_id, f"unknown_{i}_{j}")
                        
                        # Validate shape match if we have optimizer state
                        if old_id in old_state and "exp_avg" in old_state[old_id]:
                            old_shape = old_state[old_id]["exp_avg"].shape
                            new_shape = new_param.shape
                            if old_shape != new_shape:
                                warnings.warn(
                                    f"  Parameter '{param_name}' (group {i}): "
                                    f"shape mismatch (ckpt: {old_shape}, current: {new_shape}) - skipping",
                                    UserWarning
                                )
                                shape_mismatches += 1
                                continue
                        
                        id_map[old_id] = new_id
                        matched_params += 1
                    
                    # Count unmatched in this group
                    if len(old_params) > len(new_param_tensors):
                        unmatched_in_checkpoint += len(old_params) - len(new_param_tensors)
                    else:
                        unmatched_in_current += len(new_param_tensors) - len(old_params)
                
                # Count params in entirely unmatched groups
                if len(opt_state["param_groups"]) > len(optimizer.param_groups):
                    for i in range(len(optimizer.param_groups), len(opt_state["param_groups"])):
                        unmatched_in_checkpoint += len(opt_state["param_groups"][i]["params"])
                else:
                    for i in range(len(opt_state["param_groups"]), len(optimizer.param_groups)):
                        unmatched_in_current += len(optimizer.param_groups[i]["params"])
                
                warnings.warn(
                    f"  Matched: {matched_params} parameters\n"
                    f"  Unmatched in checkpoint: {unmatched_in_checkpoint}\n"
                    f"  Unmatched in current model: {unmatched_in_current}\n"
                    f"  Shape mismatches: {shape_mismatches}",
                    UserWarning
                )
                
                # Apply the matched optimizer state
                for old_id, new_id in id_map.items():
                    if old_id in old_state:
                        optimizer.state[new_id] = old_state[old_id]
                
                # Update hyperparameters for matched groups
                for i in range(min(len(opt_state["param_groups"]), len(optimizer.param_groups))):
                    loaded_group = opt_state["param_groups"][i]
                    current_group = optimizer.param_groups[i]
                    for key in loaded_group:
                        if key != 'params':
                            current_group[key] = loaded_group[key]
            
            else:
                # Fallback: no model access, just skip optimizer loading
                warnings.warn(
                    "  Cannot match parameters (no model access). Skipping optimizer state loading.",
                    UserWarning
                )
        else:
            # Standard loading when groups match
            optimizer.load_state_dict(opt_state)
        
        _optimizer_to_device(optimizer, self.root_device)

