"""
GB10 / Nemotron compatibility patches. Import this BEFORE any of:
  transformers, peft, trl, mamba_ssm, accelerate

Patches applied:
  1. causal_conv1d.causal_conv1d_fn = None  (force torch_forward path)
  2. mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn → pure-PyTorch
  3. safetensors.safe_open metadata wrapper — handles metadata=None which
     transformers 4.46.3+ chokes on for older safetensors files.
"""

# 1. causal_conv1d ── force torch_forward by nulling out the fast-path entrypoints.
try:
    import causal_conv1d as _cc1d
    _cc1d.causal_conv1d_fn = None
    _cc1d.causal_conv1d_update = None
except ImportError:
    pass


# 2. rmsnorm_fn ── replace Triton kernel with a pure-PyTorch equivalent.
import torch as _torch


def _rmsnorm_fn_pytorch(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    if z is not None and not norm_before_gate:
        x = x * _torch.nn.functional.silu(z)
    orig_shape = x.shape
    if group_size is not None:
        x = x.view(*orig_shape[:-1], -1, group_size)
    normed = x * _torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if group_size is not None:
        normed = normed.view(orig_shape)
    out = normed * weight
    if bias is not None:
        out = out + bias
    if z is not None and norm_before_gate:
        out = out * _torch.nn.functional.silu(z)
    return out


try:
    import mamba_ssm.ops.triton.layernorm_gated as _lg
    _lg.rmsnorm_fn = _rmsnorm_fn_pytorch
    import sys as _sys
    for _mod_name, _mod in list(_sys.modules.items()):
        if "nemotron" in _mod_name and hasattr(_mod, "rmsnorm_fn"):
            _mod.rmsnorm_fn = _rmsnorm_fn_pytorch
except ImportError:
    pass


# 3. safetensors metadata patch ── transformers 4.46.3+ does
#    `metadata.get("format")` after `f.metadata()`, but safetensors files
#    written by older tools may have no metadata (returns None). Wrap
#    safe_open so .metadata() always returns a dict.
try:
    import safetensors as _st

    _orig_safe_open = _st.safe_open

    class _SafetensorsReaderWrapper:
        def __init__(self, reader):
            self._r = reader

        def metadata(self):
            m = self._r.metadata()
            if m is None:
                return {"format": "pt"}
            if not isinstance(m, dict):
                return {"format": "pt"}
            if m.get("format") is None:
                return {**m, "format": "pt"}
            return m

        def keys(self):
            return self._r.keys()

        def get_tensor(self, key):
            return self._r.get_tensor(key)

        def get_slice(self, key):
            return self._r.get_slice(key)

        def __getattr__(self, name):
            return getattr(self._r, name)

        def __iter__(self):
            return iter(self._r)

    class _SafetensorsContextWrapper:
        def __init__(self, *args, **kwargs):
            self._cm = _orig_safe_open(*args, **kwargs)

        def __enter__(self):
            return _SafetensorsReaderWrapper(self._cm.__enter__())

        def __exit__(self, *args):
            return self._cm.__exit__(*args)

    _st.safe_open = _SafetensorsContextWrapper
    # Also patch the torch submodule's reference if it's already loaded
    try:
        from safetensors import torch as _st_torch
        if hasattr(_st_torch, "safe_open"):
            _st_torch.safe_open = _SafetensorsContextWrapper
    except ImportError:
        pass
except ImportError:
    pass