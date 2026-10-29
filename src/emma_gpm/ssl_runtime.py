"""Explicit CUDA execution controls adapted from Demetra VideoMAE utilities.

Demetra provenance: moduli/videomae/utils.py configure_cuda_runtime,
maybe_compile_model; optim_factory.py fusedadamw. Native PyTorch AdamW avoids
an Apex dependency. These controls preserve model dimensions and objectives.
"""
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sysconfig

import torch


@dataclass(frozen=True)
class CudaRuntime:
    tf32: bool = False
    cudnn_benchmark: bool = False
    fused_adamw: bool = False
    static_positions: bool = False
    compile_mode: str = "off"
    sdpa_kernel: str = "auto"

    def __post_init__(self):
        if any(type(v) is not bool for v in (self.tf32, self.cudnn_benchmark, self.fused_adamw, self.static_positions)):
            raise ValueError("Runtime switches must be JSON booleans")
        if self.compile_mode not in {"off", "default", "reduce-overhead", "max-autotune-no-cudagraphs"}:
            raise ValueError("Unsupported compilation mode")
        if self.sdpa_kernel not in {"auto", "flash"}:
            raise ValueError("Expected auto or flash SDPA")

    def to_dict(self):
        return asdict(self)


def configure_runtime(settings, device):
    if torch.device(device).type != "cuda":
        if settings != CudaRuntime():
            raise ValueError("CUDA performance controls require a CUDA device")
        return
    torch.set_float32_matmul_precision("high" if settings.tf32 else "highest")
    # Existing PyTorch default enables TF32 convolutions. Preserve it for the
    # baseline; the explicit switch above controls remaining FP32 matmuls.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = settings.cudnn_benchmark
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(settings.sdpa_kernel == "auto")
    torch.backends.cuda.enable_math_sdp(settings.sdpa_kernel == "auto")
    torch.backends.cuda.enable_cudnn_sdp(settings.sdpa_kernel == "auto")


def make_optimizer(model, settings, *, lr, weight_decay):
    kwargs = {"fused": True} if settings.fused_adamw else {}
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=weight_decay, **kwargs)


def apply_optimizer_runtime(optimizer, settings):
    """Reapply kernel selection after loading groups from an older checkpoint.

    load_state_dict restores the old group's fused flag and step locations.
    Moment tensors, update counts, learning rates and decay remain unchanged.
    """
    for group in optimizer.param_groups:
        group["fused"] = True if settings.fused_adamw else None
        group["foreach"] = None
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "step" in state:
                state["step"] = state["step"].to(parameter.device if settings.fused_adamw else "cpu")


def compiler_headers():
    """Use optional user-local Ubuntu Python headers on headless GPU workers."""
    if (Path(sysconfig.get_path("include"))/"Python.h").exists():
        return
    root = Path.home()/".local/lib/python-dev/usr/include"
    include = root/Path(sysconfig.get_path("include")).name
    if not (include/"Python.h").exists():
        raise RuntimeError("Inductor requires Python development headers (Python.h). "
                           "Install python3-dev or extract libpython3-dev under ~/.local/lib/python-dev.")
    paths = [str(include), str(root)]
    if os.environ.get("CPATH"):
        paths.append(os.environ["CPATH"])
    os.environ["CPATH"] = os.pathsep.join(paths)


def compile_model_regions(model, settings):
    """Compile dense transformer stacks; keep mask validation outside the graph.

    Module.compile preserves state_dict names and the canonical model object.
    The shared encoder is specialized for both visible and full token lengths.
    Mask RNG and data validation remain in their original execution order.
    """
    if settings.static_positions:
        device = next(model.parameters()).device
        owners = [model.encoder.embeddings]
        if model.mae is not None:
            owners.append(model.mae)
        for owner in owners:
            value = owner.position_embeddings
            if isinstance(value, torch.nn.Parameter):
                raise ValueError("Only fixed positional encodings can become runtime buffers")
            if "position_embeddings" not in owner._buffers:
                delattr(owner, "position_embeddings")
                owner.register_buffer("position_embeddings", value.to(device), persistent=False)
    if settings.compile_mode == "off":
        return []
    compiler_headers()
    regions = [("encoder.encoder", model.encoder.encoder)]
    if model.head is not None:
        regions.append(("head", model.head))
    if model.mae is not None:
        regions.append(("mae.decoder", model.mae.decoder))
    for _, module in regions:
        module.compile(backend="inductor", mode=settings.compile_mode, fullgraph=True, dynamic=False)
    return [name for name, _ in regions]
