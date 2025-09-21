import torch
import itertools
import warnings
from collections import OrderedDict
from typing import Any, List, Mapping, namedtuple

# from torch/nn/modules/module.py

_EXTRA_STATE_KEY_SUFFIX = "_extra_state"


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def _load_from_state_dict(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    r"""Copy parameters and buffers from :attr:`state_dict` into only this module, but not its descendants.

    This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.
    Additionally, :attr:`local_metadata` can also contain the key
    `assign_to_params_buffers` that indicates whether keys should be
    assigned their corresponding tensor in the state_dict.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in self._load_state_dict_pre_hooks.values():
        hook(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    persistent_buffers = {
        k: v
        for k, v in self._buffers.items()
        if k not in self._non_persistent_buffers_set
    }
    local_name_params = itertools.chain(
        self._parameters.items(), persistent_buffers.items()
    )
    local_state = {k: v for k, v in local_name_params if v is not None}
    assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
    use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]
            if not torch.overrides.is_tensor_like(input_param):
                error_msgs.append(
                    f'While copying the parameter named "{key}", '
                    "expected torch.Tensor or Tensor-like object from checkpoint but "
                    f"received {type(input_param)}"
                )
                continue

            # This is used to avoid copying uninitialized parameters into
            # non-lazy modules, since they dont have the hook to do the checks
            # in such case, it will error when accessing the .shape attribute.
            is_param_lazy = torch.nn.parameter.is_lazy(param)
            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if (
                not is_param_lazy
                and len(param.shape) == 0
                and len(input_param.shape) == 1
            ):
                input_param = input_param[0]

            if not is_param_lazy and input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape {input_param.shape} from checkpoint, "
                    f"the shape in current model is {param.shape}."
                )
                continue

            if (
                param.is_meta
                and not input_param.is_meta
                and not assign_to_params_buffers
            ):
                warnings.warn(
                    f"for {key}: copying from a non-meta parameter in the checkpoint to a meta "
                    "parameter in the current model, which is a no-op. (Did you mean to "
                    "pass `assign=True` to assign items in the state dictionary to their "
                    "corresponding key in the module instead of copying them in place?)"
                )

            try:
                with torch.no_grad():
                    if use_swap_tensors:
                        new_input_param = param.module_load(
                            input_param, assign=assign_to_params_buffers
                        )
                        if id(new_input_param) == id(input_param) or id(
                            new_input_param
                        ) == id(param):
                            raise RuntimeError(
                                "module_load returned one of self or other, please .detach() "
                                "the result if returning one of the inputs in module_load"
                            )
                        if isinstance(param, torch.nn.Parameter):
                            if not isinstance(new_input_param, torch.nn.Parameter):
                                new_input_param = torch.nn.Parameter(
                                    new_input_param,
                                    requires_grad=param.requires_grad,
                                )
                            else:
                                new_input_param.requires_grad_(param.requires_grad)
                        torch.utils.swap_tensors(param, new_input_param)
                        del new_input_param
                    elif assign_to_params_buffers:
                        # Shape checks are already done above
                        if isinstance(param, torch.nn.Parameter):
                            if not isinstance(input_param, torch.nn.Parameter):
                                input_param = torch.nn.Parameter(
                                    input_param, requires_grad=param.requires_grad
                                )
                            else:
                                input_param.requires_grad_(param.requires_grad)
                        setattr(self, name, input_param)
                    else:
                        param.copy_(input_param)
            except Exception as ex:
                action = "swapping" if use_swap_tensors else "copying"
                error_msgs.append(
                    f'While {action} the parameter named "{key}", '
                    f"whose dimensions in the model are {param.size()} and "
                    f"whose dimensions in the checkpoint are {input_param.size()}, "
                    f"an exception occurred : {ex.args}."
                )
        elif strict:
            missing_keys.append(key)

    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if (
        getattr(self.__class__, "set_extra_state", torch.nn.Module.set_extra_state)
        is not torch.nn.Module.set_extra_state
    ):
        if extra_state_key in state_dict:
            self.set_extra_state(state_dict[extra_state_key])
        elif strict:
            missing_keys.append(extra_state_key)
    elif strict and (extra_state_key in state_dict):
        unexpected_keys.append(extra_state_key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix) and key != extra_state_key:
                input_name = key[len(prefix) :].split(".", 1)
                # Must be Module if it have attributes
                if len(input_name) > 1:
                    if input_name[0] not in self._modules:
                        unexpected_keys.append(key)
                elif input_name[0] not in local_state:
                    unexpected_keys.append(key)


def load_state_dict(
    self,
    state_dict: Mapping[str, Any],
    assign: bool = False,
    strict_unexpected_keys: bool = False,
    strict_missing_keys: bool = False,
):
    r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

    If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    .. warning::
        If :attr:`assign` is ``True`` the optimizer must be created after
        the call to :attr:`load_state_dict` unless
        :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        assign (bool, optional): When ``False``, the properties of the tensors
            in the current module are preserved while when ``True``, the
            properties of the Tensors in the state dict are preserved. The only
            exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
            for which the value from the module is preserved.
            Default: ``False``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing any keys that are expected
                by this module but missing from the provided ``state_dict``.
            * **unexpected_keys** is a list of str containing the keys that are not
                expected by this module but present in the provided ``state_dict``.

    Note:
        If a parameter or buffer is registered as ``None`` and its corresponding key
        exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
        ``RuntimeError``.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, local_state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata["assign_to_params_buffers"] = assign
        module._load_from_state_dict(
            local_state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {
                    k: v
                    for k, v in local_state_dict.items()
                    if k.startswith(child_prefix)
                }
                load(child, child_state_dict, child_prefix)  # noqa: F821

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible_keys)
            assert out is None, (
                "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                "expected to return new values, if incompatible_keys need to be modified,"
                "it should be done inplace."
            )

    load(self, state_dict)
    del load

    print("\nload_state_dict()")
    print(f"Unexpected keys: {len(unexpected_keys)} / {len(state_dict.keys())}")
    if len(unexpected_keys) > 0:
        print(unexpected_keys[:10])
    print(f"Missing keys: {len(missing_keys)} / {len(state_dict.keys())}")
    if len(missing_keys) > 0:
        print(missing_keys[:10])

    if strict_unexpected_keys:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0,
                "Unexpected key(s) in state_dict: {}. ".format(
                    ", ".join(f'"{k}"' for k in unexpected_keys[:20])
                ),
            )

    if strict_missing_keys:
        if len(missing_keys) > 0:
            error_msgs.insert(
                0,
                "Missing key(s) in state_dict: {}. ".format(
                    ", ".join(f'"{k}"' for k in missing_keys[:20])
                ),
            )

    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                self.__class__.__name__, "\n\t".join(error_msgs)
            )
        )
    return _IncompatibleKeys(missing_keys, unexpected_keys)
