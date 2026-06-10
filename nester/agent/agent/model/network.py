import torch
import torch.nn as nn
from einops import rearrange

from nester.agent.agent.helpers import check_index_range, check_unit_interval

_IDX_DTYPE = torch.int32


def _create_mlp(input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.GELU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ValueNetwork(nn.Module):
    def __init__(self, encoder_dim: int, max_polygons: int, value_config: dict, device: str = "cuda"):
        super().__init__()
        value_hidden_dims = value_config.get("hidden_dims", [256, 128])
        value_input_dim = max_polygons * encoder_dim + 2
        self.value_network = _create_mlp(value_input_dim, 1, value_hidden_dims).to(device)

    def forward(
        self,
        polygon_cls_outputs: torch.Tensor,
        valid_poly_mask: torch.Tensor,
        normalized_step: torch.Tensor,
        prev_best_occupancy: torch.Tensor,
    ) -> torch.Tensor:
        masked_cls = polygon_cls_outputs * valid_poly_mask.unsqueeze(-1)
        flat_cls = rearrange(masked_cls, "b n d -> b (n d)")
        dt = polygon_cls_outputs.dtype
        check_unit_interval(normalized_step, "normalized_step")
        check_unit_interval(prev_best_occupancy, "prev_best_occupancy")
        step_feat = normalized_step.float().to(dtype=dt).unsqueeze(-1)
        best_feat = prev_best_occupancy.float().to(dtype=dt).unsqueeze(-1)
        flat_cls = torch.cat([flat_cls, step_feat, best_feat], dim=-1)
        return self.value_network(flat_cls).squeeze(-1)


class SelectionRotationPolicy(nn.Module):
    def __init__(self, encoder_dim: int, policy_config: dict, device: str = "cuda"):
        super().__init__()
        self.num_rot_coarse_bins = int(policy_config.get("rotation_coarse_bins", 72))
        self.num_rot_fine_bins = int(policy_config.get("rotation_fine_bins", 10))
        self.num_rot_bins = self.num_rot_coarse_bins * self.num_rot_fine_bins
        pointer_hidden_dims = policy_config.get("pointer_hidden_dims", [64])
        rot_hidden_dims = policy_config.get("pose_hidden_dims", [128, 64])
        self.pointer_head = _create_mlp(encoder_dim, 1, pointer_hidden_dims).to(device)
        self.rotation_coarse_head = _create_mlp(
            encoder_dim, self.num_rot_coarse_bins, rot_hidden_dims
        ).to(device)
        self.rotation_fine_head = _create_mlp(
            encoder_dim,
            self.num_rot_coarse_bins * self.num_rot_fine_bins,
            rot_hidden_dims,
        ).to(device)

    def forward(self, polygon_features: torch.Tensor, valid_poly_mask: torch.Tensor):
        pointer_logits = self.pointer_head(polygon_features).squeeze(-1)
        neg_inf = torch.tensor(
            float("-inf"), device=polygon_features.device, dtype=pointer_logits.dtype
        )
        pointer_logits = pointer_logits.masked_fill(~valid_poly_mask, neg_inf)
        coarse_logits = self.rotation_coarse_head(polygon_features)
        fine_logits = self.rotation_fine_head(polygon_features).reshape(
            *polygon_features.shape[:2],
            self.num_rot_coarse_bins,
            self.num_rot_fine_bins,
        )
        rotation_logits = (
            coarse_logits.log_softmax(dim=-1).unsqueeze(-1)
            + fine_logits.log_softmax(dim=-1)
        ).flatten(-2)
        return pointer_logits, rotation_logits


class Stage2Block(nn.Module):
    def __init__(self, d_model, nhead, ff_mult, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        ff_dim = d_model * ff_mult
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tokens, key_padding_mask):
        x = tokens
        x = self.norm1(x + self.self_attn(x, x, x, key_padding_mask=key_padding_mask)[0])
        x = self.norm2(x + self.ff(x))
        return x


class Stage2PlacementNetwork(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        nfp_patch_size: int,
        policy_config: dict,
        device: str = "cuda",
    ):
        super().__init__()
        self.subregion_bins = int(policy_config.get("subregion_bins", 5))
        self.encoder_dim = encoder_dim
        self.nfp_patch_size = nfp_patch_size
        patch_flat = nfp_patch_size * 2
        # Max 1-based component id: 1=exterior, 2–3=largest holes (see select_nfp_component_rings).
        self.max_nfp_components = int(policy_config.get("max_nfp_components", 3))
        self.num_layers = int(policy_config.get("stage2_num_blocks", 4))
        self.nfp_patch_embed = nn.Linear(patch_flat, encoder_dim).to(device)
        self.nfp_poly_embedding = nn.Embedding(self.max_nfp_components + 1, encoder_dim).to(device)
        self.delta_theta_embed = _create_mlp(2, encoder_dim, [encoder_dim]).to(device)
        self.candidate_pos_embedding = nn.Embedding(
            int(policy_config.get("max_candidates", 10)), encoder_dim
        ).to(device)
        num_heads = int(policy_config.get("cross_attention_heads", 4))
        ff_mult = int(policy_config.get("stage2_ff_mult", 4))
        self.blocks = nn.ModuleList([
            Stage2Block(encoder_dim, num_heads, ff_mult, 0.0)
            for _ in range(self.num_layers)
        ]).to(device)
        hidden_dims = policy_config.get("placement_hidden_dims", [128, 64])
        self.placement_head = _create_mlp(encoder_dim, self.subregion_bins, hidden_dims).to(device)

    def forward(
        self,
        nfp_patches: torch.Tensor,
        nfp_mask: torch.Tensor,
        nfp_poly_ids: torch.Tensor,
        nfp_vertex_mask: torch.Tensor,
        candidate_cls: torch.Tensor,
        selected_ids: torch.Tensor,
        delta_theta: torch.Tensor,
        scene_tokens: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t_nfp, p, _ = nfp_patches.shape
        if p != self.nfp_patch_size:
            raise ValueError(f"nfp_patches last dim expected P={self.nfp_patch_size}, got {p}")
        if scene_tokens.shape[:2] != scene_mask.shape:
            raise ValueError(
                f"scene_tokens shape prefix {tuple(scene_tokens.shape[:2])} does not match "
                f"scene_mask shape {tuple(scene_mask.shape)}"
            )
        if scene_tokens.shape[0] != b or scene_tokens.shape[2] != self.encoder_dim:
            raise ValueError(
                f"scene_tokens expected (B, N, {self.encoder_dim}) with B={b}, "
                f"got {tuple(scene_tokens.shape)}"
            )
        if nfp_vertex_mask is None:
            raise ValueError("nfp_vertex_mask must be provided")
        if nfp_vertex_mask.shape != nfp_patches.shape[:3]:
            raise ValueError(
                f"nfp_vertex_mask shape {tuple(nfp_vertex_mask.shape)} does not match "
                f"nfp_patches shape prefix {tuple(nfp_patches.shape[:3])}"
            )
        nfp_vertex_mask = nfp_vertex_mask.to(device=nfp_patches.device, dtype=torch.bool)
        check_index_range(nfp_poly_ids, 0, self.max_nfp_components, "nfp_poly_ids")
        poly_ids = nfp_poly_ids.to(_IDX_DTYPE)
        flat = nfp_patches.masked_fill(~nfp_vertex_mask.unsqueeze(-1), 0.0).reshape(b, t_nfp, -1)
        token_embed = self.nfp_patch_embed(flat)
        nfp_tok = token_embed + self.nfp_poly_embedding(poly_ids)
        check_index_range(
            selected_ids,
            0,
            self.candidate_pos_embedding.num_embeddings - 1,
            "selected_ids",
        )
        selected_ids = selected_ids.to(_IDX_DTYPE)
        candidate_tok = candidate_cls + self.candidate_pos_embedding(selected_ids)
        delta_theta = delta_theta.to(device=nfp_patches.device).float()
        delta_tok = self.delta_theta_embed(
            torch.stack([torch.sin(delta_theta), torch.cos(delta_theta)], dim=-1)
        )
        scene_mask = scene_mask.to(device=nfp_mask.device, dtype=torch.bool)
        all_tokens = torch.cat(
            [candidate_tok.unsqueeze(1), delta_tok.unsqueeze(1), scene_tokens, nfp_tok],
            dim=1,
        )
        all_kpm = torch.cat(
            [
                torch.zeros((b, 2), dtype=torch.bool, device=nfp_mask.device),
                ~scene_mask,
                ~nfp_mask,
            ],
            dim=1,
        )
        for block in self.blocks:
            all_tokens = block(all_tokens, all_kpm)
        nfp_out = all_tokens[:, 2 + scene_tokens.shape[1] :, :]
        local_logits = self.placement_head(nfp_out)
        s = self.subregion_bins
        if s <= 1:
            vertex_idx = torch.zeros(1, dtype=torch.long, device=nfp_patches.device)
        else:
            vertex_idx = (
                torch.arange(s, dtype=torch.long, device=nfp_patches.device) * (p - 1)
            ) // (s - 1)
        subregion_valid = nfp_vertex_mask.index_select(2, vertex_idx)
        valid_logits = nfp_mask.unsqueeze(-1) & subregion_valid
        local_logits = local_logits.masked_fill(
            ~valid_logits,
            torch.tensor(float("-inf"), device=local_logits.device, dtype=local_logits.dtype),
        )
        logits = local_logits.reshape(b, t_nfp * s)
        return logits
