import torch.nn as nn

from models.conditional_inputs import CONDITIONAL_INPUT_CHANNELS
from models.fno_layers import DomainPadding, FNOBlock, GridEmbedding2D


class FNO2DModel(nn.Module):
    """2-D FNO operating on feature channels in channel-last layout.

    When ``positional_embedding='grid'``, the model appends ``x``/``y``
    coordinates internally, so callers should pass feature channels only
    and strip legacy coordinate channels before invocation.
    """

    def __init__(
        self,
        in_channels=CONDITIONAL_INPUT_CHANNELS,
        out_channels=1,
        width=32,
        n_modes=(12, 12),
        n_layers=4,
        enforce_hermitian_symmetry=True,
        channel_mlp_expansion=0.5,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        positional_embedding=None,
        domain_padding=None,
        norm=None,
    ):
        super().__init__()
        if positional_embedding == "grid":
            self.positional_embedding = GridEmbedding2D()
            lift_in = in_channels + 2
        elif positional_embedding is None:
            self.positional_embedding = None
            lift_in = in_channels
        else:
            raise ValueError(
                f"Unknown positional_embedding {positional_embedding!r}. "
                "Expected 'grid' or None."
            )
        if domain_padding is not None and float(domain_padding) > 0:
            self.domain_padding = DomainPadding(float(domain_padding))
        else:
            self.domain_padding = None
        self.lift = nn.Linear(lift_in, width)
        self.blocks = nn.ModuleList(
            [
                FNOBlock(
                    width,
                    n_modes,
                    enforce_hermitian_symmetry=enforce_hermitian_symmetry,
                    channel_mlp_expansion=channel_mlp_expansion,
                    fno_skip=fno_skip,
                    channel_mlp_skip=channel_mlp_skip,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        if self.domain_padding is not None:
            original_res = tuple(x.shape[2:])
            x = self.domain_padding.pad(x)

        for block in self.blocks:
            x = block(x)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x, original_res)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.activation(self.proj1(x))
        return self.proj2(x)
