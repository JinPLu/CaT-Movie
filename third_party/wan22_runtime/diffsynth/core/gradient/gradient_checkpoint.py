import torch
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from importlib import import_module
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ModuleNotFoundError:
    _HAS_DEEPSPEED = False


_in_self_attention = ContextVar("wan_checkpoint_self_attention", default=False)


@contextmanager
def self_attention_checkpoint_scope():
    token = _in_self_attention.set(True)
    try:
        yield
    finally:
        _in_self_attention.reset(token)


def self_attention_checkpoint_contexts(operator):
    def policy(context, op, *args, **kwargs):
        return (CheckpointPolicy.MUST_SAVE if op is operator and _in_self_attention.get()
                else CheckpointPolicy.PREFER_RECOMPUTE)
    return create_selective_checkpoint_contexts(policy)


def _self_attention_forward_operator(device):
    """The forward operator ``flash_attention`` actually dispatches on ``device``; its output is what gets saved.

    Only backends with a measured saving path are accepted: FA2 (A100, 11-07/11-08 acceptance) and cuDNN SDPA
    (compute capability >= 10). Any other backend raises instead of silently saving nothing.
    """
    backend = import_module("diffsynth.models.wan_video_dit").attention_backend(device)
    if backend == "cudnn":
        return torch.ops.aten._scaled_dot_product_cudnn_attention.default
    if backend == "flash_attn_2":
        return torch.ops.flash_attn._flash_attn_forward.default
    raise RuntimeError(f"saving SelfAttention has no accepted forward operator for the {backend!r} backend")


class _SelfAttentionCheckpointBinding:
    def __init__(self, dit, blocks, context_fn, cache_layers, checkpointed_layers):
        self.dit, self.blocks, self.context_fn = dit, blocks, context_fn
        self.cache_layers = cache_layers
        first_cached = checkpointed_layers - cache_layers
        self.cached_blocks = blocks[first_cached:checkpointed_layers]
        self.no_recompute_blocks = blocks[checkpointed_layers:]
        self.expected_contexts = ((None,) * first_cached + (context_fn,) * cache_layers
                                  + (None,) * len(self.no_recompute_blocks))
        self.expected_no_recompute = (False,) * checkpointed_layers + (True,) * len(self.no_recompute_blocks)
        self.active = True

    def validate(self):
        if (not self.active or tuple(self.dit.blocks) != self.blocks
                or any(block.activation_checkpoint_context_fn is not expected
                       for block, expected in zip(self.blocks, self.expected_contexts))
                or any(getattr(block, "_skip_activation_checkpointing", False) is not expected
                       for block, expected in zip(self.blocks, self.expected_no_recompute))):
            raise RuntimeError("SelfAttention checkpoint binding belongs to another owner")

    def restore(self):
        self.validate()
        for block in self.cached_blocks:
            block.activation_checkpoint_context_fn = None
        for block in self.no_recompute_blocks:
            del block._skip_activation_checkpointing
        self.active = False


def install_self_attention_checkpointing(dit, *, cache_layers=None, no_recompute_suffix_layers=None):
    """Save SA outputs in the last checkpointed blocks and keep all activations in an explicit suffix.

    None for ``cache_layers`` selects all checkpointed blocks. With no suffix, the existing
    suffix-aligned SA saving is unchanged.
    """
    blocks = tuple(dit.blocks)
    if not blocks or len({id(block) for block in blocks}) != len(blocks):
        raise ValueError("saving SelfAttention requires a nonempty sequence of distinct DiT blocks")
    if no_recompute_suffix_layers is not None and (
        type(no_recompute_suffix_layers) is not int or not 1 <= no_recompute_suffix_layers < len(blocks)
    ):
        raise ValueError(f"no_recompute_suffix_layers must be None or a positive integer below {len(blocks)}")
    checkpointed_layers = len(blocks) - (no_recompute_suffix_layers or 0)
    if cache_layers is not None and (type(cache_layers) is not int or not 1 <= cache_layers <= checkpointed_layers):
        raise ValueError(f"SelfAttention cache_layers must be None or an integer within 1..{checkpointed_layers}")
    cache_layers = checkpointed_layers if cache_layers is None else cache_layers
    if any(block.activation_checkpoint_context_fn is not None
           or hasattr(block, "_skip_activation_checkpointing") for block in blocks):
        raise RuntimeError("SelfAttention checkpoint contexts are already owned")
    if _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured():
        raise RuntimeError("saving SelfAttention requires ordinary non-reentrant checkpointing")
    devices = {parameter.device for parameter in dit.parameters()}
    if len(devices) != 1:
        raise RuntimeError(f"saving SelfAttention needs the DiT on one device to pick its attention operator; got {devices}")
    context_fn = partial(self_attention_checkpoint_contexts, _self_attention_forward_operator(devices.pop()))
    binding = _SelfAttentionCheckpointBinding(dit, blocks, context_fn, cache_layers, checkpointed_layers)
    for block in binding.cached_blocks:
        block.activation_checkpoint_context_fn = context_fn
    for block in binding.no_recompute_blocks:
        block._skip_activation_checkpointing = True
    return binding


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def create_custom_forward_use_reentrant(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward


def judge_args_requires_grad(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return True
    return False


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    context_fn=None,
    **kwargs,
):
    deepspeed_configured = _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured()
    skip_checkpointing = getattr(model, "_skip_activation_checkpointing", False)
    if context_fn is not None:
        if (not use_gradient_checkpointing or use_gradient_checkpointing_offload
                or deepspeed_configured or skip_checkpointing):
            raise RuntimeError("selective checkpoint contexts require ordinary non-reentrant checkpointing")
    if not torch.is_grad_enabled() or skip_checkpointing:
        return model(*args, **kwargs)
    if use_gradient_checkpointing_offload and deepspeed_configured:
        raise RuntimeError("checkpoint input offload requires ordinary non-reentrant checkpointing")
    if use_gradient_checkpointing and deepspeed_configured:
        all_args = args + tuple(kwargs.values())
        if not judge_args_requires_grad(*all_args):
            # get the first grad_enabled tensor from un_checkpointed forward
            model_output = model(*args, **kwargs)
        else:
            model_output = deepspeed.checkpointing.checkpoint(
                create_custom_forward_use_reentrant(model),
                *all_args,
            )
        return model_output
    if use_gradient_checkpointing_offload:
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("checkpoint input offload requires a tensor as the first positional input")
        input_x, shared_args = args[0], args[1:]

        def forward_with_shared_inputs(x):
            # 每次调用绑定自己的block及共享条件。非重入checkpoint记录这些条件的
            # autograd边；只把x直接交给input_saver，避免逐层复制同一份t_mod。
            return model(x, *shared_args, **kwargs)

        with torch.autograd.graph.save_on_cpu():
            model_output = torch.utils.checkpoint.checkpoint(
                forward_with_shared_inputs,
                input_x,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
            **({} if context_fn is None else {"context_fn": context_fn}),
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
