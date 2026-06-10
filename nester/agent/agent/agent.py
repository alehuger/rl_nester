from typing import Dict

import torch
import torch.nn as nn
from torch.distributions import Categorical

from nester.agent.agent.helpers import check_finite_logits, check_index_range, wrap_angle
from nester.agent.agent.model.PoT import Pot_RL
from nester.agent.agent.model.network import (
    SelectionRotationPolicy,
    Stage2PlacementNetwork,
    ValueNetwork,
)


class PotRLAgent(nn.Module):
    def __init__(
        self,
        encoder_config: dict,
        policy_config: dict,
        value_config: dict,
        data_patch_size: int,
        device: str = "cuda",
        max_patches: int = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.max_patches = max_patches

        # 1. Encoder
        self.encoder = Pot_RL(**encoder_config, device=device)
        encoder_dim = self.encoder.embed_dim
        self.max_polygons = self.encoder.max_polygons

        # 2. Value and Policy Networks
        self.value_network = ValueNetwork(
            encoder_dim, self.max_polygons, value_config, device=device
        )
        self.stage1_policy = SelectionRotationPolicy(encoder_dim, policy_config, device=device)
        stage2_policy_config = dict(policy_config)
        stage2_policy_config["max_candidates"] = self.max_polygons
        self.stage2_policy = Stage2PlacementNetwork(
            encoder_dim, self.encoder.patch_size, stage2_policy_config, device=device
        )
        self.num_rot_bins = self.stage1_policy.num_rot_bins
        self.register_buffer("rot_centers", self._build_uniform_centers(self.num_rot_bins))

        self.data_patch_size = data_patch_size

    # ### MAIN AGENT METHODS ###

    def _prepare_obs_tensors(self, obs: Dict[int, dict]) -> dict:
        batch_of_dicts = list(obs.values())

        normalized_steps = torch.tensor(
            [float(d.get("normalized_step", 0.0)) for d in batch_of_dicts],
            dtype=torch.float32,
        ).to(self.device, non_blocking=True)
        prev_best_occupancy = torch.tensor(
            [float(d.get("prev_best_occupancy", 0.0)) for d in batch_of_dicts],
            dtype=torch.float32,
        ).to(self.device, non_blocking=True)

        # 1. Collate data from all environments in the batch
        flat_vertices, poly_lengths, batch_ids, original_ids = self._collate_flattened_observations(
            batch_of_dicts
        )

        # Move data to the agent's device
        flat_vertices = flat_vertices.to(self.device, non_blocking=True)
        poly_lengths = poly_lengths.to(self.device, non_blocking=True)
        batch_ids = batch_ids.to(self.device, non_blocking=True)
        original_ids = original_ids.to(self.device, non_blocking=True)

        # 2. Create padded patch tensors and mask
        patches, patch_mask = self._vectorized_patch_padding(
            flat_vertices,
            poly_lengths,
            batch_ids,
            original_ids,
            self.encoder.patch_size,
            self.max_patches,
        )
        polygon_current_theta = self._collate_polygon_theta(batch_of_dicts)
        return {
            "patches": patches,
            "patch_mask": patch_mask,
            "normalized_steps": normalized_steps,
            "prev_best_occupancy": prev_best_occupancy,
            "polygon_current_theta": polygon_current_theta,
            "flat_vertices": flat_vertices,
            "poly_lengths": poly_lengths,
            "batch_ids": batch_ids,
            "original_ids": original_ids,
        }

    def _process_obs(
        self, obs: Dict[int, dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prepared = self._prepare_obs_tensors(obs)
        return (
            prepared["patches"],
            prepared["patch_mask"],
            prepared["normalized_steps"],
            prepared["prev_best_occupancy"],
        )

    def get_value(self, obs: Dict[int, dict]) -> torch.Tensor:
        with torch.no_grad():
            encoded = self._encode_stage1_base(obs)
            value = encoded["value"]
        return value

    def get_action_and_value(
        self,
        obs: Dict[int, dict],
        action: dict = None,
        deterministic: bool = False,
        nfp_patches: torch.Tensor = None,
        nfp_mask: torch.Tensor = None,
        nfp_poly_ids: torch.Tensor = None,
        nfp_vertex_mask: torch.Tensor = None,
    ):
        encoded = self._encode_stage1_base(obs)
        return self._rollout_from_encoded(
            encoded=encoded,
            action=action,
            deterministic=deterministic,
            nfp_patches=nfp_patches,
            nfp_mask=nfp_mask,
            nfp_poly_ids=nfp_poly_ids,
            nfp_vertex_mask=nfp_vertex_mask,
        )

    def get_action_and_value_from_encoded(
        self,
        encoded: dict,
        action: dict = None,
        deterministic: bool = False,
        nfp_patches: torch.Tensor = None,
        nfp_mask: torch.Tensor = None,
        nfp_poly_ids: torch.Tensor = None,
        nfp_vertex_mask: torch.Tensor = None,
    ):
        return self._rollout_from_encoded(
            encoded=encoded,
            action=action,
            deterministic=deterministic,
            nfp_patches=nfp_patches,
            nfp_mask=nfp_mask,
            nfp_poly_ids=nfp_poly_ids,
            nfp_vertex_mask=nfp_vertex_mask,
        )

    def _rollout_from_encoded(
        self,
        encoded: dict,
        action: dict = None,
        deterministic: bool = False,
        nfp_patches: torch.Tensor = None,
        nfp_mask: torch.Tensor = None,
        nfp_poly_ids: torch.Tensor = None,
        nfp_vertex_mask: torch.Tensor = None,
    ):
        stage1_action = action["stage1"] if action is not None and "stage1" in action else None
        stage2_action = action["stage2"] if action is not None and "stage2" in action else None
        stage1_out = self._stage1_action(
            encoded=encoded,
            action_1=stage1_action,
            deterministic=deterministic,
        )

        if nfp_patches is None or nfp_mask is None:
            return {
                "stage1": stage1_out,
                "value": encoded["value"],
                "context": encoded,
            }

        stage2_out = self._stage2_action(
            encoded=encoded,
            stage1_action=stage1_out["action"],
            nfp_patches=nfp_patches,
            nfp_mask=nfp_mask,
            nfp_poly_ids=nfp_poly_ids,
            nfp_vertex_mask=nfp_vertex_mask,
            action_2=stage2_action,
            deterministic=deterministic,
        )

        joint_logprob = stage1_out["logprob"] + stage2_out["logprob"]
        entropy_total = stage1_out["entropy"] + stage2_out["entropy"]
        return {
            "action": {
                **stage1_out["action"],
                **stage2_out["action"],
            },
            "joint_logprob": joint_logprob,
            "joint_entropy": entropy_total,
            "value": encoded["value"],
            "stage1": stage1_out,
            "stage2": stage2_out,
            "logprob_components": {
                "stage1_item": stage1_out["logprob_components"]["item"],
                "stage1_rotation": stage1_out["logprob_components"]["rotation"],
                "stage2_place": stage2_out["logprob_components"]["place"],
            },
            "entropy_components": {
                "stage1_item": stage1_out["entropy_components"]["item"],
                "stage1_rotation": stage1_out["entropy_components"]["rotation"],
                "stage2_place": stage2_out["entropy_components"]["place"],
                "total": entropy_total,
            },
        }

    def evaluate_actions(
        self,
        obs: Dict[int, dict],
        action_1: dict,
        action_2: dict,
        nfp_patches: torch.Tensor,
        nfp_mask: torch.Tensor,
        nfp_poly_ids: torch.Tensor,
        nfp_vertex_mask: torch.Tensor = None,
    ):
        encoded = self._encode_stage1_base(obs)
        stage1_out = self._stage1_action(encoded=encoded, action_1=action_1, deterministic=False)
        stage2_out = self._stage2_action(
            encoded=encoded,
            stage1_action=stage1_out["action"],
            nfp_patches=nfp_patches,
            nfp_mask=nfp_mask,
            nfp_poly_ids=nfp_poly_ids,
            nfp_vertex_mask=nfp_vertex_mask,
            action_2=action_2,
            deterministic=False,
        )
        joint_logprob = stage1_out["logprob"] + stage2_out["logprob"]
        joint_entropy = stage1_out["entropy"] + stage2_out["entropy"]
        return {
            "joint_logprob": joint_logprob,
            "joint_entropy": joint_entropy,
            "value": encoded["value"],
            "stage1": stage1_out,
            "stage2": stage2_out,
        }

    @staticmethod
    def _collate_polygon_theta(batch_of_dicts: list[dict]) -> torch.Tensor:
        if not batch_of_dicts:
            return torch.zeros((0, 0), dtype=torch.float32)
        max_poly = max(
            int(d["poly_ids"].max()) + 1 if len(d.get("poly_ids", [])) > 0 else 0
            for d in batch_of_dicts
        )
        if max_poly <= 0:
            max_poly = 1
        out = torch.zeros((len(batch_of_dicts), max_poly), dtype=torch.float32)
        for b, obs in enumerate(batch_of_dicts):
            for pid, th in zip(obs.get("poly_ids", []), obs.get("poly_theta", [])):
                out[b, int(pid)] = float(th)
        return out

    def _pad_polygon_theta(self, polygon_current_theta: torch.Tensor) -> torch.Tensor:
        n = int(self.max_polygons)
        if polygon_current_theta.shape[1] == n:
            return polygon_current_theta
        if polygon_current_theta.shape[1] > n:
            return polygon_current_theta[:, :n]
        pad = torch.zeros(
            (polygon_current_theta.shape[0], n - polygon_current_theta.shape[1]),
            dtype=polygon_current_theta.dtype,
        )
        return torch.cat([polygon_current_theta, pad], dim=1)

    def _absolute_theta(
        self,
        polygon_current_theta: torch.Tensor,
        polygon_id: torch.Tensor,
        delta_theta: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(polygon_id.shape[0], device=polygon_id.device)
        current = polygon_current_theta[batch_idx, polygon_id].to(
            device=delta_theta.device, dtype=delta_theta.dtype
        )
        return wrap_angle(current + delta_theta)

    def _build_uniform_centers(self, bins: int) -> torch.Tensor:
        return torch.linspace(-torch.pi, torch.pi, steps=bins + 1)[:-1] + (torch.pi / bins)

    def _decode_delta_rotation(self, theta_idx: torch.Tensor) -> torch.Tensor:
        """Map bin index to delta_theta in [-pi, pi] (relative rotation)."""
        theta_idx = theta_idx.long()
        centers = self.rot_centers.to(theta_idx.device)
        return centers[theta_idx]

    def _encode_stage1_base(self, obs: Dict[int, dict]) -> dict:
        prepared = self._prepare_obs_tensors(obs)
        patches = prepared["patches"]
        patch_mask = prepared["patch_mask"]
        normalized_steps = prepared["normalized_steps"]
        prev_best = prepared["prev_best_occupancy"]
        polygon_cls_outputs, valid_poly_mask = self.encoder(
            (patches, patch_mask),
            normalized_step=normalized_steps,
            prev_best_occupancy=prev_best,
        )
        h1 = polygon_cls_outputs
        value = self.value_network(h1, valid_poly_mask, normalized_steps, prev_best)
        item_logits, rotation_logits = self.stage1_policy(h1, valid_poly_mask)
        return {
            "h1": h1,
            "valid_poly_mask": valid_poly_mask,
            "value": value,
            "item_logits": item_logits,
            "rotation_logits": rotation_logits,
            "patches": patches,
            "patch_mask": patch_mask,
            "normalized_steps": normalized_steps,
            "prev_best_occupancy": prev_best,
            "polygon_current_theta": self._pad_polygon_theta(prepared["polygon_current_theta"]).to(
                self.device, non_blocking=True
            ),
            "flat_vertices": prepared["flat_vertices"],
            "poly_lengths": prepared["poly_lengths"],
            "batch_ids": prepared["batch_ids"],
            "original_ids": prepared["original_ids"],
        }

    def _stage1_action(self, encoded: dict, action_1: dict = None, deterministic: bool = False):
        valid_poly_mask = encoded["valid_poly_mask"].bool()
        stage1_allowed_mask = valid_poly_mask
        if (~stage1_allowed_mask.any(dim=1)).any():
            raise ValueError("No valid polygon available for selection")
        masked_item_logits = encoded["item_logits"].masked_fill(
            ~stage1_allowed_mask, torch.finfo(encoded["item_logits"].dtype).min
        )
        item_dist = Categorical(logits=check_finite_logits(masked_item_logits.float(), "item"))
        if action_1 is None:
            polygon_id = torch.argmax(item_dist.probs, dim=-1) if deterministic else item_dist.sample()
        else:
            polygon_id = action_1["polygon_id"].to(self.device).long()
            batch_idx_check = torch.arange(polygon_id.shape[0], device=self.device)
            if not stage1_allowed_mask[batch_idx_check, polygon_id].all():
                raise ValueError("Provided selection id is invalid")
        b = polygon_id.shape[0]
        batch_idx = torch.arange(b, device=self.device)
        selected_rot_logits = encoded["rotation_logits"][batch_idx, polygon_id].float()
        rot_dist = Categorical(logits=check_finite_logits(selected_rot_logits, "rotation"))
        if action_1 is None:
            theta_idx = torch.argmax(rot_dist.probs, dim=-1) if deterministic else rot_dist.sample()
        else:
            theta_idx = action_1["theta_idx"].to(self.device).long()
        delta_theta = self._decode_delta_rotation(theta_idx)
        logprob_item = item_dist.log_prob(polygon_id)
        logprob_rot = rot_dist.log_prob(theta_idx)
        polygon_current_theta = encoded["polygon_current_theta"].float()
        theta_abs = self._absolute_theta(polygon_current_theta, polygon_id, delta_theta)
        rot_entropy_selected = rot_dist.entropy()
        item_raw = encoded["item_logits"][encoded["valid_poly_mask"]].float()
        item_logit_max = item_raw.abs().max().item() if item_raw.numel() > 0 else 0.0
        rot_logit_max = selected_rot_logits.abs().max().item()
        return {
            "action": {
                "polygon_id": polygon_id,
                "theta_idx": theta_idx,
                "theta": delta_theta,
                "theta_abs": theta_abs,
            },
            "logprob": logprob_item + logprob_rot,
            "entropy": item_dist.entropy() + rot_entropy_selected,
            "logprob_components": {"item": logprob_item, "rotation": logprob_rot},
            "entropy_components": {
                "item": item_dist.entropy(),
                "rotation": rot_entropy_selected,
            },
            "logit_abs_max": {"item": item_logit_max, "rotation": rot_logit_max},
        }

    def _translation_from_patch_subregion(
        self,
        nfp_patches: torch.Tensor,
        patch_idx: torch.Tensor,
        subregion_idx: torch.Tensor,
    ) -> torch.Tensor:
        n_bins = self.stage2_policy.subregion_bins
        b, _, p, _ = nfp_patches.shape
        device = nfp_patches.device
        bi = torch.arange(b, device=device)
        check_index_range(patch_idx, 0, nfp_patches.shape[1] - 1, "nfp_patch_idx")
        patch = nfp_patches[bi, patch_idx.long()]
        check_index_range(subregion_idx, 0, n_bins - 1, "nfp_subregion_idx")
        k = subregion_idx.long()
        if p <= 1:
            return patch[:, 0]
        if n_bins <= 1:
            vi = torch.zeros(b, dtype=torch.long, device=device)
        else:
            vi = (k * (p - 1)) // (n_bins - 1)
        check_index_range(vi, 0, p - 1, "nfp vertex index")
        return patch[bi, vi]

    def _stage2_action(
        self,
        encoded: dict,
        stage1_action: dict,
        nfp_patches: torch.Tensor,
        nfp_mask: torch.Tensor,
        nfp_poly_ids: torch.Tensor,
        nfp_vertex_mask: torch.Tensor = None,
        action_2: dict = None,
        deterministic: bool = False,
    ):
        work_device = encoded["h1"].device
        nfp_patches = nfp_patches.to(work_device, non_blocking=True)
        nfp_mask = nfp_mask.to(work_device, non_blocking=True).bool()
        if nfp_poly_ids is None:
            raise ValueError("nfp_poly_ids must be provided for Stage-2 action")
        nfp_poly_ids = nfp_poly_ids.to(work_device, non_blocking=True).long()
        if nfp_vertex_mask is not None:
            nfp_vertex_mask = nfp_vertex_mask.to(work_device, non_blocking=True).bool()
        b = encoded["h1"].shape[0]
        selected_ids = stage1_action["polygon_id"].to(work_device, non_blocking=True).long()
        delta_theta = stage1_action["theta"].to(work_device, non_blocking=True)

        h1 = encoded["h1"]
        batch_idx = torch.arange(b, device=work_device)
        candidate_cls = h1[batch_idx, selected_ids]

        logits = self.stage2_policy(
            nfp_patches=nfp_patches,
            nfp_mask=nfp_mask,
            nfp_poly_ids=nfp_poly_ids,
            nfp_vertex_mask=nfp_vertex_mask,
            candidate_cls=candidate_cls,
            selected_ids=selected_ids,
            delta_theta=delta_theta,
            scene_tokens=h1,
            scene_mask=encoded["valid_poly_mask"].to(work_device, non_blocking=True).bool(),
        )
        logits_f32 = logits.float()
        raw_finite = logits_f32[logits_f32.isfinite()]
        place_logit_max = raw_finite.abs().max().item() if raw_finite.numel() > 0 else 0.0
        place_dist = Categorical(logits=check_finite_logits(logits_f32, "stage2_place"))
        n_sub = self.stage2_policy.subregion_bins
        if action_2 is None:
            flat_idx = (
                torch.argmax(place_dist.probs, dim=-1) if deterministic else place_dist.sample()
            )
        else:
            flat_idx = (
                action_2["nfp_patch_idx"].to(work_device, non_blocking=True).long() * n_sub
            ) + action_2["nfp_subregion_idx"].to(work_device, non_blocking=True).long()
        nfp_patch_idx = flat_idx // n_sub
        nfp_subregion_idx = flat_idx % n_sub
        translation = self._translation_from_patch_subregion(
            nfp_patches, nfp_patch_idx, nfp_subregion_idx
        )
        return {
            "action": {
                "nfp_patch_idx": nfp_patch_idx,
                "nfp_subregion_idx": nfp_subregion_idx,
                "translation": translation,
            },
            "logits": logits,
            "logprob": place_dist.log_prob(flat_idx),
            "entropy": place_dist.entropy(),
            "logprob_components": {"place": place_dist.log_prob(flat_idx)},
            "entropy_components": {"place": place_dist.entropy()},
            "logit_abs_max": {"place": place_logit_max},
        }

    ### HELPER METHODS FOR INPUT DATA PROCESSING ###

    @staticmethod
    def _collate_flattened_observations(batch_of_dicts: list[dict]):
        """
        Takes a list of environment outputs and merges them into
        contiguous tensors for the vectorized GPU pipeline.

        Args:
            batch_of_dicts (list): List of dicts, where each dict contains:
                                - 'flat_vertices': np.array
                                - 'poly_lengths': np.array
                                - 'poly_ids': np.array (Real IDs from env)

        Returns:
            flat_vertices (Tensor): (Total_Verts, 2)
            poly_lengths (Tensor): (Total_Polys,)
            batch_ids (Tensor): (Total_Polys,) - Maps poly to batch index
            original_ids (Tensor): (Total_Polys,) - Maps poly to original ID
        """

        # Pre-allocate lists to hold the tensors from each env
        all_verts = []
        all_lens = []
        all_batch_ids = []
        all_original_ids = []

        for batch_idx, obs in enumerate(batch_of_dicts):
            # 1. Vertices: Just append the list of coordinates
            #    (The environments already did the work of flattening them)
            v = torch.from_numpy(obs["flat_vertices"])
            all_verts.append(v)

            # 2. Lengths: Append the size of each polygon
            lengths_tensor = torch.from_numpy(obs["poly_lengths"])
            all_lens.append(lengths_tensor)

            # 3. Original IDs: Keep track of which family member this is
            original_id = torch.from_numpy(obs["poly_ids"])
            all_original_ids.append(original_id)

            # 4. Batch IDs: Create a tensor saying "These polys belong to Batch X"
            #    If this env has 5 polygons, we create [X, X, X, X, X]
            num_polys = lengths_tensor.shape[0]
            b_ids = torch.full((num_polys,), batch_idx, dtype=torch.long)
            all_batch_ids.append(b_ids)

        # Concatenate into the single "Mega-Tensors" the GPU expects
        # This is very fast (memory copy)
        flat_vertices = torch.cat(all_verts, dim=0)
        poly_lengths = torch.cat(all_lens, dim=0)
        batch_ids = torch.cat(all_batch_ids, dim=0)
        original_ids = torch.cat(all_original_ids, dim=0)

        return flat_vertices, poly_lengths, batch_ids, original_ids

    @staticmethod
    def _vectorized_patch_padding(
        flat_vertices, poly_lengths, batch_ids, original_poly_ids, patch_size, max_patches=None
    ):
        device = flat_vertices.device

        # --- 1. Calculate Dimensions ---
        patches_per_poly = (poly_lengths + patch_size - 1) // patch_size
        total_patches = patches_per_poly.sum().item()

        # --- 2. Expand Metadata (Repeat for every patch) ---
        # Lengths: needed for modulo
        expanded_poly_lens = torch.repeat_interleave(poly_lengths, patches_per_poly)
        # Batch IDs: needed for final scatter
        expanded_batch_ids = torch.repeat_interleave(batch_ids, patches_per_poly)
        # Original IDs: needed for the mask
        expanded_original_ids = torch.repeat_interleave(original_poly_ids, patches_per_poly)

        # Calculate global start indices for every polygon in the flat array
        poly_start_indices = torch.cat(
            [torch.tensor([0], device=device), poly_lengths.cumsum(0)[:-1]]
        )
        expanded_poly_starts = torch.repeat_interleave(poly_start_indices, patches_per_poly)

        # --- 3. Generate Local Patch Indices ---
        # (0, 1, 2... for patches within a specific polygon)
        patch_cumsum = patches_per_poly.cumsum(0)
        patch_start_indices = torch.cat([torch.tensor([0], device=device), patch_cumsum[:-1]])
        expanded_patch_starts = torch.repeat_interleave(patch_start_indices, patches_per_poly)

        global_patch_range = torch.arange(total_patches, device=device)
        local_patch_idx = global_patch_range - expanded_patch_starts

        # --- 4. Generate Gather Indices (Cyclic Logic) ---
        start_vertex_per_patch = local_patch_idx * patch_size
        window_offsets = torch.arange(patch_size, device=device).unsqueeze(0)

        # Raw indices: start + 0, start + 1...
        raw_indices = start_vertex_per_patch.unsqueeze(1) + window_offsets

        # Modulo: Wrap around the polygon length
        cyclic_indices = raw_indices % expanded_poly_lens.unsqueeze(1)

        # Shift: Add the polygon's start position in the flat buffer
        final_gather_indices = cyclic_indices + expanded_poly_starts.unsqueeze(1)

        # --- 5. GATHER ---
        flattened_patches = flat_vertices[final_gather_indices]

        # --- 6. SCATTER Prep ---
        batch_size = batch_ids.max().item() + 1

        # Count patches per batch to define max size
        patches_per_batch = torch.bincount(expanded_batch_ids, minlength=batch_size)
        if max_patches is None:
            max_patches_per_batch = patches_per_batch.max().item()
        else:
            max_patches_per_batch = max_patches

        # Calculate Row Indices: Where does this patch go within its Batch?
        # (0, 1, 2... for patches within a specific batch element)
        batch_starts = torch.cat(
            [torch.tensor([0], device=device), patches_per_batch.cumsum(0)[:-1]]
        )

        # Map starts to every patch
        expanded_batch_starts = batch_starts[expanded_batch_ids]
        row_indices = global_patch_range - expanded_batch_starts

        # --- 7. SCATTER ---
        # PAD_ID is 0 in the encoder
        final_tensor = torch.zeros(
            (batch_size, max_patches_per_batch, patch_size, 2),
            dtype=torch.float32,
            device=device,
        )
        final_mask = torch.zeros(
            (batch_size, max_patches_per_batch), dtype=torch.long, device=device
        )

        # Truncate if necessary
        scatter_batch_ids = expanded_batch_ids
        scatter_row_indices = row_indices
        scatter_patches = flattened_patches
        scatter_original_ids = expanded_original_ids

        if max_patches is not None:
            overflow = row_indices >= max_patches_per_batch
            if bool(overflow.any().item()):
                raise ValueError(
                    f"Batch needs more than max_patches={max_patches} patch rows per env element. "
                    "Increase model max_patches or reduce contour length via resample_length / "
                    "max_vertices."
                )

        # Write Patch Data
        final_tensor[scatter_batch_ids, scatter_row_indices] = scatter_patches

        # Write ID Data. Encoder expects 1-based polygon IDs.
        final_mask[scatter_batch_ids, scatter_row_indices] = scatter_original_ids + 1


        return final_tensor, final_mask
