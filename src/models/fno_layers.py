import torch
import torch.nn as nn


class SoftGating(nn.Module):
    """Multiplies each channel by a learned scalar: out = w * x.

    w has shape (1, channels, 1, 1) so it broadcasts over batch and spatial dims.
    """

    def __init__(self, channels, n_dim=2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, *(1,) * n_dim))

    def forward(self, x):
        return self.weight * x


def _build_skip(channels, skip_type):
    """Return an nn.Module for the requested skip type, or None.

    Args:
        channels:  number of input == output channels.
        skip_type: ``"soft-gating"``, ``"linear"``, ``"identity"``, or ``None``.
    """
    if skip_type is None:
        return None
    s = skip_type.lower()
    if s == "soft-gating":
        return SoftGating(channels)
    if s == "linear":
        return nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    if s == "identity":
        return nn.Identity()
    raise ValueError(
        f"Unknown skip_type {skip_type!r}. "
        "Expected 'soft-gating', 'linear', 'identity', or None."
    )


def _build_norm(channels, norm_type):
    """Return a normalisation module for ``channels`` channel-first 2-D tensors.

    Args:
        channels:  number of channels (``C`` in ``(B, C, H, W)``).
        norm_type: ``"instance_norm"``, ``"group_norm"``, or ``None``.
                   ``"group_norm"`` uses ``min(32, channels)`` groups.
    """
    if norm_type is None:
        return None
    s = norm_type.lower()
    if s == "instance_norm":
        return nn.InstanceNorm2d(channels, affine=True)
    if s == "group_norm":
        num_groups = min(32, channels)
        return nn.GroupNorm(num_groups, channels)
    raise ValueError(
        f"Unknown norm {norm_type!r}. "
        "Expected 'instance_norm', 'group_norm', or None."
    )


class ChannelMLP(nn.Module):
    """Two-layer channel-wise MLP using Conv1d(kernel=1), spatial-resolution-agnostic.

    Args:
        in_channels:    number of input (and output) channels.
        expansion:      hidden_channels = int(in_channels * expansion). Default 0.5.
        non_linearity:  activation applied between the two layers. Default F.gelu.
    """

    def __init__(self, in_channels, expansion=0.5, non_linearity=nn.functional.gelu):
        super().__init__()
        hidden = max(1, int(in_channels * expansion))
        self.fc1 = nn.Conv1d(in_channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x):
        # x: (batch, channels, *spatial)  — flatten spatial for Conv1d, restore after
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)   # (B, C, H*W)
        x = self.non_linearity(self.fc1(x))
        x = self.fc2(x)
        return x.reshape(shape)                  # (B, C, H, W)


class SpectralConv(nn.Module):
    """N-dimensional spectral convolution using rfftn.

    Args:
        in_channels:  number of input channels.
        out_channels: number of output channels.
        n_modes:      sequence of Fourier modes to keep per spatial dimension,
                      e.g. (12, 12) for 2-D.
    """

    def __init__(self, in_channels, out_channels, n_modes, enforce_hermitian_symmetry=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = list(n_modes)
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry

        scale = 1.0 / max(in_channels * out_channels, 1)
        weight_shape = (*n_modes, in_channels, out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(weight_shape))
        self.weight_imag = nn.Parameter(scale * torch.randn(weight_shape))

        # Build einsum string once at construction time: "bc{sp_str},{sp_str}co->bo{sp_str}"
        ndim = len(n_modes)
        sp_str = "".join(chr(ord("x") + i) for i in range(ndim))
        self._einsum_str = f"bc{sp_str},{sp_str}co->bo{sp_str}"

    def forward(self, x):
        # x: (batch, channels, *spatial)
        spatial = x.shape[2:]
        ndim = len(spatial)
        fft_dims = tuple(range(-ndim, 0))

        x_ft = torch.fft.rfftn(x, dim=fft_dims)

        # Clamp modes; the last dim is halved by the real FFT.
        modes = [min(self.n_modes[i], spatial[i]) for i in range(ndim - 1)]
        modes.append(min(self.n_modes[-1], spatial[-1] // 2 + 1))

        out_shape = (x.shape[0], self.out_channels) + tuple(
            spatial[i] if i < ndim - 1 else spatial[-1] // 2 + 1
            for i in range(ndim)
        )
        out_ft = torch.zeros(out_shape, dtype=torch.cfloat, device=x.device)

        slices = (slice(None), slice(None)) + tuple(slice(m) for m in modes)
        weight = torch.complex(
            self.weight_real[tuple(slice(m) for m in modes)],
            self.weight_imag[tuple(slice(m) for m in modes)],
        )
        out_ft[slices] = torch.einsum(self._einsum_str, x_ft[slices], weight)

        if self.enforce_hermitian_symmetry:
            # Split into (n-1)-dim ifftn + 1-dim irfft to fix GPU cuFFT aliasing.
            # The 0th and (even) Nyquist bins must be strictly real before irfft.
            ifft_dims = fft_dims[:-1]
            if ifft_dims:
                out_ft = torch.fft.ifftn(out_ft, s=tuple(spatial[:-1]), dim=ifft_dims)
            out_ft[..., 0].imag.zero_()
            if spatial[-1] % 2 == 0:
                out_ft[..., -1].imag.zero_()
            return torch.fft.irfft(out_ft, n=spatial[-1], dim=-1)
        else:
            return torch.fft.irfftn(out_ft, s=tuple(spatial), dim=fft_dims)


class FNOBlock(nn.Module):
    def __init__(
        self,
        channels,
        n_modes,
        enforce_hermitian_symmetry=True,
        channel_mlp_expansion=0.5,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        norm=None,
    ):
        super().__init__()
        self.spectral = SpectralConv(
            channels,
            channels,
            n_modes,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
        )
        self.fno_skip = _build_skip(channels, fno_skip)
        self.activation = nn.GELU()
        self.norm = _build_norm(channels, norm)
        self.channel_mlp = ChannelMLP(channels, expansion=channel_mlp_expansion)
        self.channel_mlp_skip = _build_skip(channels, channel_mlp_skip)

    def forward(self, x):
        # FNO branch: spectral(x) + optional skip(x) → activation → optional norm
        x_fno_skip = self.fno_skip(x) if self.fno_skip is not None else 0
        x = self.activation(self.spectral(x) + x_fno_skip)
        if self.norm is not None:
            x = self.norm(x)
        # ChannelMLP branch: mlp(x) + optional skip(x)
        x_mlp_skip = self.channel_mlp_skip(x) if self.channel_mlp_skip is not None else 0
        x = self.channel_mlp(x) + x_mlp_skip
        return x


class DomainPadding(nn.Module):
    """Symmetrically zero-pads the spatial domain before FNO blocks and crops
    it back after, reducing spectral wrap-around aliasing at boundaries.

    The padding amount is computed as ``round(fraction * resolution)`` and is
    cached per resolution so it is only computed once per unique input size.

    Input/output shape: (batch, channels, H, W)  — channel-first.

    Args:
        domain_padding: fraction of each spatial dimension to pad (e.g. 0.1 =
            10% padding on each side of each dimension).
    """

    def __init__(self, domain_padding: float):
        super().__init__()
        self.domain_padding = float(domain_padding)
        self._pad_cache: dict[tuple, list[int]] = {}
        self._unpad_cache: dict[tuple, tuple] = {}

    def _compute(self, resolution: tuple) -> tuple[list[int], tuple]:
        """Compute and cache padding/unpadding specs for a given spatial size."""
        if resolution not in self._pad_cache:
            # Number of pixels to pad on each side per dimension
            pads = [round(self.domain_padding * r) for r in resolution]
            # F.pad(x, ...) takes sizes in reverse-dim order, two values per dim
            pad_arg = [v for p in reversed(pads) for v in (p, p)]
            # Unpad slices: skip `p` at start, skip `p` at end
            unpad = (Ellipsis,) + tuple(
                slice(p if p else None, -p if p else None) for p in pads
            )
            self._pad_cache[resolution] = pad_arg
            self._unpad_cache[resolution] = unpad
        return self._pad_cache[resolution], self._unpad_cache[resolution]

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        pad_arg, _ = self._compute(tuple(x.shape[2:]))
        return torch.nn.functional.pad(x, pad_arg, mode="constant", value=0.0)

    def unpad(self, x: torch.Tensor, original_resolution: tuple) -> torch.Tensor:
        _, unpad = self._compute(original_resolution)
        return x[unpad]


class GridEmbedding2D(nn.Module):
    """Appends (x, y) coordinate channels to a channel-last 2-D grid tensor.

    Input shape:  (batch, H, W, C)
    Output shape: (batch, H, W, C+2)

    Coordinates are in [0, 1] and are recomputed only when the spatial
    resolution changes (cached otherwise).
    """

    def __init__(self):
        super().__init__()
        self._cached_grid: torch.Tensor | None = None
        self._cached_res: tuple | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        if self._cached_grid is None or self._cached_res != (H, W):
            x_1d = torch.linspace(0, 1, H, dtype=x.dtype, device=x.device)
            y_1d = torch.linspace(0, 1, W, dtype=x.dtype, device=x.device)
            xx, yy = torch.meshgrid(x_1d, y_1d, indexing="ij")  # (H, W)
            self._cached_grid = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
            self._cached_res = (H, W)
        grid = self._cached_grid.to(device=x.device, dtype=x.dtype)
        return torch.cat([x, grid.unsqueeze(0).expand(B, -1, -1, -1)], dim=-1)
