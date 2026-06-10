import copy
import math

import torch
import torch.nn as nn
from einops import rearrange

from nester.agent.agent.helpers import check_index_range, check_unit_interval


# Mask IDs
MP_ID_START = 1
PAD_ID = 0

_IDX_DTYPE = torch.int32


class PotTransformerEncoderLayer(nn.Module):
    """Drop-in `nn.TransformerEncoderLayer` with batch-first MHA and padding mask support."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: nn.Module,
        layer_norm_eps: float,
        norm_first: bool,
        batch_first: bool,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        out = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class PotTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class Pot_RL(nn.Module):
    def __init__(
        self,
        transformer: dict,
        mlp_heads: dict,
        data_patch_size: int = None,
        max_candidates: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        self.embed_dim = transformer.get("embed_dim", 128)
        self.num_heads = transformer.get("num_heads", 8)
        self.num_layers = transformer.get("num_layers", 4)
        self.num_fourier_bands = transformer.get("num_fourier_bands", 4)
        self.num_registers = transformer.get("num_registers", 4)
        self.layer_norm_eps = float(transformer.get("layer_norm_eps", 1e-5))
        self.activation = transformer.get("activation", "gelu")
        self.norm_first = transformer.get("norm_first", False)

        self.mlp_nb_layers = mlp_heads.get("num_layers", 4)
        self.mlp_hidden_dim = mlp_heads.get("hidden_dim", 128)

        self.patch_size = data_patch_size
        self.max_polygons = max_candidates
        if self.patch_size is None:
            raise ValueError("`data_patch_size` must be provided (non-None).")

        # Static padding target used to keep attention shapes constant.
        # Env raises if resampled contour points exceed `max_vertices` (no silent crop).
        self.max_vertices = int(transformer.get("max_vertices", 1000))
        self.max_patches_per_poly = (self.max_vertices + self.patch_size - 1) // self.patch_size
        self.max_patches_per_batch = self.max_polygons * self.max_patches_per_poly
        print(f"max_patches_per_batch: {self.max_patches_per_batch}")

        self.device = torch.device(device)
        self.patch_embed = nn.Linear(self.patch_size * 2, self.embed_dim)

        self._build_transformer()
        self._build_embeddings()
        self._build_mlp_heads()

        self.to(self.device)
        self._register_idx_buffers()

    def _register_idx_buffers(self) -> None:
        """Pre-register small int32 index tensors used in routing and scatter ops."""
        dev = self.device
        n = int(self.max_patches_per_batch)
        self.register_buffer(
            "_idx_zero",
            torch.zeros((), dtype=_IDX_DTYPE, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "_idx_max_poly",
            torch.tensor(self.max_polygons - 1, dtype=_IDX_DTYPE, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "_idx_max_patch_pos",
            torch.tensor(n, dtype=_IDX_DTYPE, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "_mp_lo",
            torch.tensor(MP_ID_START, dtype=_IDX_DTYPE, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "_patch_pos_table",
            torch.arange(n, dtype=_IDX_DTYPE, device=dev).view(1, -1),
            persistent=False,
        )
        self.register_buffer(
            "_seg_ids_table",
            torch.arange(n, dtype=_IDX_DTYPE, device=dev).view(1, 1, -1),
            persistent=False,
        )

    def _build_transformer(self):
        if self.activation == "gelu":
            activation_fn = nn.GELU()
        elif self.activation == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        encoder_layer = PotTransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.0,
            activation=activation_fn,
            layer_norm_eps=self.layer_norm_eps,
            norm_first=self.norm_first,
            batch_first=True,
        )
        self.transformer_encoder = PotTransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps),
        )

    def _build_mlp_heads(self):

        input_dim = self.embed_dim

        # +2 for `normalized_step` and `prev_best_occupancy` after the transformer readout.
        input_dim = input_dim + 2

        if self.mlp_nb_layers == 1:
            self.mlp_heads = nn.Sequential(nn.Linear(input_dim, self.embed_dim))
            return

        layers = []
        layers.extend(
            [
                nn.Linear(input_dim, self.mlp_hidden_dim),
                nn.LayerNorm(self.mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.0),
            ]
        )

        for _ in range(self.mlp_nb_layers - 2):
            layers.extend(
                [
                    nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
                    nn.LayerNorm(self.mlp_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.0),
                ]
            )
        layers.append(nn.Linear(self.mlp_hidden_dim, self.embed_dim))
        self.mlp_heads = nn.Sequential(*layers)

    def _build_embeddings(self):
        # Global registers
        self.register_tokens = nn.Parameter(
            torch.randn(1, self.num_registers, self.embed_dim) * 0.02
        )

        # Geometric Embeddings
        fourier_dim = 2 * self.num_fourier_bands
        self.contour_projection = nn.Linear(fourier_dim, self.embed_dim)

        self.polygon_index_embedding = nn.Embedding(self.max_polygons, self.embed_dim)
        
        self.polygon_cls_token = nn.Parameter(
                torch.randn(1, 1, self.embed_dim) * 0.02
            )

    def _compute_segment_lengths(self, patch_mask):
        patch_mask = patch_mask.to(_IDX_DTYPE)
        bs, max_patches = patch_mask.shape
        boundary = torch.zeros(bs, max_patches, device=self.device, dtype=torch.bool)
        boundary[:, 1:] = patch_mask[:, 1:] != patch_mask[:, :-1]
        boundary[:, 0] = True
        segment_idx = boundary.to(_IDX_DTYPE).cumsum(dim=1) - 1
        max_segs = max_patches
        valid_mask = patch_mask != self._idx_zero
        seg_idx_valid = segment_idx.clone()
        seg_idx_valid[~valid_mask] = 0
        seg_counts = torch.zeros(bs, max_segs, device=self.device, dtype=_IDX_DTYPE)
        seg_counts.scatter_add_(1, seg_idx_valid.long(), valid_mask.to(_IDX_DTYPE))
        segment_lengths = seg_counts.gather(1, segment_idx)
        return segment_lengths

    def _valid_poly_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        poly_indices = patch_mask - self._mp_lo
        is_mp = patch_mask >= self._mp_lo
        check_index_range(
            poly_indices,
            0,
            self.max_polygons - 1,
            "polygon patch index",
            mask=is_mp,
        )
        poly_idx_lookup = torch.where(is_mp, poly_indices, self._idx_zero)
        counts = torch.zeros(
            patch_mask.shape[0], self.max_polygons, dtype=_IDX_DTYPE, device=self.device
        )
        counts.scatter_add_(1, poly_idx_lookup.long(), is_mp.to(_IDX_DTYPE))
        return counts > 0

    def _get_cpe_vectorized(
        self,
        segment_lengths,
        patch_positions,
        *,
        dtype: torch.dtype,
        active_mask: torch.Tensor,
    ):
        if active_mask.any():
            active_lens = segment_lengths[active_mask]
            if (active_lens < 1).any():
                raise ValueError(
                    "segment_lengths must be >= 1 for active patches, "
                    f"min={int(active_lens.min().item())}"
                )
        # Run trig in float32; bfloat16 sin/cos can promote to unsupported dtypes.
        seg_p = torch.where(
            active_mask,
            segment_lengths,
            torch.ones_like(segment_lengths),
        ).to(dtype=torch.float32)
        pos_f = patch_positions.to(dtype=torch.float32)
        angles = (2.0 * math.pi * pos_f) / seg_p
        freqs = (2.0 ** torch.arange(self.num_fourier_bands, device=self.device, dtype=torch.float32)).view(
            1, 1, -1
        )
        ang_freq = angles.unsqueeze(-1) * freqs
        feats32 = torch.cat([torch.sin(ang_freq), torch.cos(ang_freq)], dim=-1)
        feats = feats32.to(dtype=dtype)
        return self.contour_projection(feats)

    def _embed_patches(self, patches, patch_mask):
        patch_mask = patch_mask.to(_IDX_DTYPE)
        bs, max_patches, _, _ = patches.shape
        patches_flat = rearrange(patches, "b n p c -> (b n) (p c)")

        # Base Geometric Embedding
        geo_embed = self.patch_embed(patches_flat)
        geo_embed = rearrange(geo_embed, "(b n) d -> b n d", b=bs, n=max_patches)

        # CPE
        boundary = torch.zeros(bs, max_patches, device=self.device, dtype=torch.bool)
        boundary[:, 1:] = patch_mask[:, 1:] != patch_mask[:, :-1]
        boundary[:, 0] = True
        cumsum = self._patch_pos_table[:, :max_patches].expand(bs, -1)
        is_mp = patch_mask >= self._mp_lo
        poly_indices = patch_mask - self._mp_lo
        check_index_range(
            poly_indices,
            0,
            self.max_polygons - 1,
            "polygon patch index",
            mask=is_mp,
        )
        seg_id = boundary.to(_IDX_DTYPE).cumsum(dim=1) - 1  # [B, N], starts at 0 (boundary[:, 0] = True)
        check_index_range(seg_id, 0, max_patches - 1, "segment id", mask=is_mp)
        num_segs = max_patches
        seg_id_lookup = torch.where(is_mp, seg_id, self._idx_zero)
        seg_ids = self._seg_ids_table[:, :, :num_segs]
        seg_membership = seg_id_lookup.unsqueeze(-1) == seg_ids
        pos = cumsum.unsqueeze(-1).expand(bs, max_patches, num_segs)
        pad_val = self._idx_max_patch_pos.expand(bs, max_patches, num_segs)
        first_pos_per_seg = torch.where(seg_membership, pos, pad_val).min(dim=1).values
        seg_start = first_pos_per_seg.gather(1, seg_id_lookup)
        patch_positions = cumsum - seg_start
        segment_lengths = self._compute_segment_lengths(patch_mask)
        cpe = self._get_cpe_vectorized(
            segment_lengths,
            patch_positions,
            dtype=patches.dtype,
            active_mask=is_mp,
        )

        is_pad = patch_mask == self._idx_zero

        embeddings = torch.zeros(bs, max_patches, self.embed_dim, device=self.device, dtype=patches.dtype)
        
        # Add Geometry + CPE to valid patches
        embeddings = embeddings + is_mp.unsqueeze(-1) * (geo_embed + cpe)

        poly_idx_lookup = torch.where(is_mp, poly_indices, self._idx_zero)
        poly_idx_embed = self.polygon_index_embedding(poly_idx_lookup)
        embeddings = embeddings + is_mp.unsqueeze(-1) * poly_idx_embed

        return embeddings, is_pad

    def forward(self, batch, normalized_step=None, prev_best_occupancy=None):
        patches, patch_mask = batch
        patches = patches.to(self.device, non_blocking=True)
        patch_mask = patch_mask.to(self.device, non_blocking=True).to(_IDX_DTYPE)
        bs = patches.shape[0]
        # normalized_step, prev_best_occupancy: (bs,) in [0, 1].

        cur_patches = patches.shape[1]
        if cur_patches > self.max_patches_per_batch:
            raise ValueError(
                f"Input has {cur_patches} patches, but max_patches_per_batch={self.max_patches_per_batch}. "
                "Increase `transformer.max_vertices` or adjust env limits."
            )
        valid_poly_mask = self._valid_poly_mask(patch_mask)

        # ----------------------------------------------------------------------
        # 2. Embed Patches
        # ----------------------------------------------------------------------
        patch_tokens, patch_pad_mask = self._embed_patches(patches, patch_mask)

        # ----------------------------------------------------------------------
        # 2. Prepare Sequence (Registers + Patches + Optional CLS)
        # ----------------------------------------------------------------------
        # Registers
        registers = self.register_tokens.expand(bs, -1, -1)
        reg_mask = torch.zeros(
            bs, self.num_registers, dtype=torch.bool, device=self.device
        )

        tokens_list = [registers, patch_tokens]
        mask_list = [reg_mask, patch_pad_mask]

        cls_tokens = self.polygon_cls_token.expand(bs, self.max_polygons, -1)
        poly_indices = self._patch_pos_table[0, : self.max_polygons]
        idx_embed = self.polygon_index_embedding(poly_indices)
        cls_tokens = cls_tokens + idx_embed.unsqueeze(0).expand(bs, -1, -1)
            
        cls_mask = ~valid_poly_mask
        
        tokens_list.append(cls_tokens)
        mask_list.append(cls_mask)

        x = torch.cat(tokens_list, dim=1)
        padding_mask = torch.cat(mask_list, dim=1)

        # ----------------------------------------------------------------------
        # 4. Transformer
        # ----------------------------------------------------------------------
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # ----------------------------------------------------------------------
        # 5. Readout & Aggregation
        # ----------------------------------------------------------------------
        
        cls_start = self.num_registers + patches.shape[1]
        polygon_representation = x[:, cls_start:, :]

        # ----------------------------------------------------------------------
        # 6. MLP Projection
        # ----------------------------------------------------------------------
        polygon_representation_flat = rearrange(polygon_representation, "b n d -> (b n) d")

        # Append normalized timestep feature to every polygon token.
        if normalized_step is None:
            normalized_step = torch.zeros(bs, device=self.device, dtype=patches.dtype)
        else:
            normalized_step = normalized_step.to(self.device, non_blocking=True).to(dtype=patches.dtype)
            check_unit_interval(normalized_step, "normalized_step")
        if prev_best_occupancy is None:
            prev_best_occupancy = torch.zeros(bs, device=self.device, dtype=patches.dtype)
        else:
            prev_best_occupancy = prev_best_occupancy.to(self.device, non_blocking=True).to(dtype=patches.dtype)
            check_unit_interval(prev_best_occupancy, "prev_best_occupancy")
        step_feat = (
            normalized_step.view(bs, 1, 1)
            .expand(bs, polygon_representation.shape[1], 1)
            .to(dtype=polygon_representation.dtype)
        )
        best_feat = (
            prev_best_occupancy.view(bs, 1, 1)
            .expand(bs, polygon_representation.shape[1], 1)
            .to(dtype=polygon_representation.dtype)
        )
        step_flat = rearrange(step_feat, "b n s -> (b n) s")
        best_flat = rearrange(best_feat, "b n s -> (b n) s")
        mlp_in = torch.cat([polygon_representation_flat, step_flat, best_flat], dim=-1)

        polygon_features = self.mlp_heads(mlp_in)
        polygon_features = rearrange(polygon_features, "(b n) d -> b n d", b=bs)

        return polygon_features, valid_poly_mask