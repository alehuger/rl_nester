# `nester.env` — simulation

### [Architecture](../ARCHITECTURE.md) · [Agent](../agent/AGENT.md)

**Gym environments for 2D irregular nesting.** Owns the MDP, panel state, occupancy reward, and **no-fit polygon (NFP)** geometry via CGAL. Reads polygon databases from `.npz` paths or shared memory; never imports `agent`.

---

## Layout

```
env/
├── env/       base_env, training_env, inference_env, generator
├── panel/     panel.py, panel_polygon.py, nfp/ (cgal, utils)
├── utils/     vec_envs, sharing_memory, helpers
└── constants.py
```

---

## Environments

| Class | Generator | Use |
|-------|-----------|-----|
| `TrainingEnv` | `nfp_random` | PPO rollouts; random families from train DB |
| `InferenceEnv` | `bench` | Fixed family in SHM; bench metrics & exports |
| `BaseEnv` | configurable | Shared logic — subclass only, not a runtime mode |

```python
from nester.env import TrainingEnv, InferenceEnv, DynamicAsyncVectorEnv
from nester.env import GeneratorNFPRandom, GeneratorBench, Scenario
```

`DynamicAsyncVectorEnv` wraps many env workers and exposes batch `step`, `compute_nfp`, and curriculum hooks for PPO.

---

## One step

1. **Reset** — generator builds an initial layout (all pieces already on the panel).
2. **Stage 1 (agent)** — pick `polygon_id` and rotation **delta** `Δθ`.
3. **`compute_nfp_for_action`** — CGAL Minkowski sum → resampled NFP patch rings for that piece at `theta_abs`.
4. **Stage 2 (agent)** — pick `translation` on an NFP patch.
5. **`step`** — apply `Δθ` + translation, update panel, return reward.

`env.step` action dict:

| Key | Meaning |
|-----|---------|
| `polygon_id` | Placed piece to move |
| `theta` | Rotation **delta** (radians), not absolute angle |
| `translation` | 2D offset from stage-2 placement |

---

## Observation

Flat dict per env (vectorized as `Dict[env_index, obs]`):

| Field | Role |
|-------|------|
| `flat_vertices` | Resampled contour vertices (panel-centered) |
| `poly_lengths` | Vertex count per placed polygon |
| `poly_ids` | Panel slot ids |
| `poly_theta` | Absolute rotation per piece — used by agent for NFP, not fed to PoT |
| `normalized_step` | `step_count / episode_length` |
| `prev_best_occupancy` | Best occupancy so far in the rollout segment |

---

## Reward

With `max_reward=True` (default in train/eval configs): sparse **peak-seeking** reward — `max(0, occupancy − rollout_best_so_far)`. Only new occupancy highs yield positive reward.

With `max_reward=False`: dense improvement signal vs previous step occupancy.

---

## Generators

| Name | Class | Data |
|------|-------|------|
| `nfp_random` | `GeneratorNFPRandom` | Sample family from train `.npz`; NFP-aware random init |
| `bench` | `GeneratorBench` | Fixed polygon set from shared memory |
| `polygons` | `GeneratorPolygons` | Generic polygon sampling |

Polygon DB schema: `polygons` array + `lengths` per row (`.npz` under `data/db/processed/`).

---

## NFP stack

- **`panel/nfp/cgal.py`** — `CGALPY` Minkowski sum → forbidden region boundary.
- **`panel/nfp/utils.py`** — select component rings, patchify boundary for the agent.
- **`panel/panel.py`** — occupancy ratio, rendering, polygon storage.

Requires **`cgalpy`** (Python 3.12 wheel on Drive, or build from [cgal-python-bindings](https://bitbucket.org/taucgl/cgal-python-bindings)).
