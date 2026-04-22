from pathlib import Path
import zipfile as _zipfile

import torch
import torch.nn as nn


def load_model_weights(
    model: nn.Module,
    path: str,
    device: torch.device,
    *,
    skip_if_torchscript: bool = False,
) -> None:
    """Load a state-dict or TorchScript archive into *model* in-place.

    Supports two checkpoint formats:
    - Plain state-dict: ``{layer_name: tensor, ...}``
    - Metadata envelope: ``{"arch": "<name>", "state_dict": {layer_name: tensor, ...}}``

    Parameters
    ----------
    model              : the ``nn.Module`` to load weights into
    path               : ``.pt`` file path (state-dict pickle or TorchScript ZIP)
    device             : target device for ``map_location``
    skip_if_torchscript: when ``True``, silently skip loading and print a notice
                         instead of attempting to extract weights from the archive.
                         Useful when warm-starting from a TorchScript checkpoint
                         is not meaningful (e.g. per-problem PINNs).

    Raises
    ------
    FileNotFoundError  : if *path* does not exist
    RuntimeError       : if loading fails for any other reason
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Check the file path or pass pretrained_path=None to train from scratch."
        )

    # Detect TorchScript by inspecting ZIP contents
    is_ts = False
    if _zipfile.is_zipfile(p):
        with _zipfile.ZipFile(p, "r") as _z:
            names = _z.namelist()
            is_ts = any("constants.pkl" in n or n.startswith("code/") for n in names)

    try:
        if is_ts:
            if skip_if_torchscript:
                print(
                    f"pretrained_path {path!r} is a TorchScript archive "
                    "— skipping warm-start (train from scratch)."
                )
                return
            jit = torch.jit.load(str(p), map_location=device)
            model.load_state_dict(jit.state_dict())
        else:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            # Unwrap metadata envelope if present
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt)
    except Exception as err:
        raise RuntimeError(
            f"Failed to load model weights from {path}: {err}"
        ) from err


def read_checkpoint_arch(path: str, device: torch.device) -> "str | None":
    """Return the ``arch`` string from a metadata-envelope checkpoint, or ``None``.

    Returns ``None`` when the file does not exist, is a TorchScript archive,
    cannot be loaded, or was saved without an ``arch`` key.
    """
    p = Path(path)
    if not p.exists():
        return None
    if _zipfile.is_zipfile(p):
        with _zipfile.ZipFile(p, "r") as _z:
            names = _z.namelist()
            if any("constants.pkl" in n or n.startswith("code/") for n in names):
                return None  # TorchScript archive — no arch key
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict):
            return ckpt.get("arch")  # None if key absent
    except Exception:
        pass
    return None


def read_checkpoint_n_points(path: str, device: torch.device) -> "int | None":
    """Return the fixed training resolution stored in checkpoint metadata, or ``None``."""
    p = Path(path)
    if not p.exists():
        return None
    if _zipfile.is_zipfile(p):
        with _zipfile.ZipFile(p, "r") as _z:
            names = _z.namelist()
            if any("constants.pkl" in n or n.startswith("code/") for n in names):
                return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "n_points" in ckpt:
            raw = ckpt["n_points"]
            if isinstance(raw, torch.Tensor):
                return int(raw.item())
            return int(raw)
    except Exception:
        pass
    return None
