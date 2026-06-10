# Nester — agent package

Architecture and data flow for `nester/agent`: policy runtime, PoT encoder, training (PPO), and bench evaluation.

Companion: [ARCHITECTURE.md](../ARCHITECTURE.md) · [ENV.md](../env/ENV.md)

---

## 1. Directory layout

```
nester/agent/
├── __init__.py                 # Public exports (PotRLAgent, PPOTrainer, …)
├── agent/
│   ├── agent.py                # PotRLAgent — two-stage policy + value
│   ├── utils.py                # make_env, perform_step, checkpoints
│   └── model/
│       ├── PoT.py              # Pot_RL encoder
│       └── network.py          # stage-1 / stage-2 / value heads
└── processes/
    ├── train/                  # PPOTrainer, logger
    └── eval/                   # bench.py, layout.py, plots.py
```

| Layer | Imports `env`? | Role |
|-------|----------------|------|
| `agent.agent.model` | No | Pure PyTorch |
| `agent.agent` | Yes | Policy + NFP hookup |
| `agent.processes` | Yes | Training and bench orchestration |

---

## 2. Who depends on whom

```
                         ┌─────────────────────┐
                         │  processes/train    │
                         │  processes/eval     │
                         └──────────┬──────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            v                       v                       v
     agent/utils.py           agent/agent.py          model/PoT.py
     make_env, ckpt                │                       │
            │                       └───────────┬───────────┘
            │                                   v
            │                            model/network.py
            │                                   │
            v                                   v
     ┌──────────────┐                    PotRLAgent
     │  nester.env  │◄─────────────────────────┘
     │ TrainingEnv  │
     │ InferenceEnv │
     └──────────────┘
```

Rule: `env` never imports `agent`. `agent` imports `env` for rollouts and NFP.

---

## 3. One improvement step (two-stage policy)

The panel is **fully placed at reset** (generator builds an initial layout). Each step **repositions one already-placed piece** — not sequential “place from empty” nesting.

```
  ┌─────────────┐
  │ observation │  flat_vertices, poly_ids, normalized_step, …
  └──────┬──────┘
         │
         v
  ┌──────────────────────────────────────┐
  │  STAGE 1 — PotRLAgent              │
  │  PoT encoder → pick polygon + Δθ   │
  └──────┬───────────────────────────────┘
         │  polygon_id, theta_abs (for NFP only)
         v
  ┌──────────────────────────────────────┐
  │  ENV — compute_nfp_for_action        │
  │  feasible placement patches (NFP)    │
  └──────┬───────────────────────────────┘
         │  nfp_patches, masks, poly_ids
         v
  ┌──────────────────────────────────────┐
  │  STAGE 2 — PotRLAgent              │
  │  logits on NFP patch grid            │
  └──────┬───────────────────────────────┘
         │  translation (placement)
         v
  ┌──────────────────────────────────────┐
  │  ENV — step                          │
  │  apply Δθ + translation → reward     │
  └──────────────────────────────────────┘
```

**Rotation flow:** stage 1 samples a rotation **delta** (bin index). The agent reads `poly_theta` from the obs dict only to form `theta_abs = wrap(poly_theta[id] + Δθ)` for NFP geometry. `env.step` receives the same **Δθ** plus stage-2 **translation**. Stage 2 does not choose rotation.

PPO and vector eval often cache the encoder after stage 1:

```
  encode(obs) ──► stage1 action ──► env NFP ──► stage2 (from_encoded)
```

Single-env helper: `perform_step()` in `utils.py` runs that chain in one call.

---

## 4. Inside `PotRLAgent`

```
  observation dict
        │
        v
  ┌───────────┐     ┌─────────────────────────┐
  │  Pot_RL   │────►│ SelectionRotationPolicy │  stage 1
  │  encoder  │     │  · polygon_id           │
  └─────┬─────┘     │  · rotation 72×10 bins  │
        │           └─────────────────────────┘
        │
        ├──────────────────► ValueNetwork  (critic)
        │
        │     (after env NFP)
        v
  ┌─────────────────────────┐
  │ Stage2PlacementNetwork  │  stage 2
  │  · selected polygon h₁  │
  │  · sin/cos Δθ token     │
  │  · scene polygon tokens │
  │  · NFP patch tokens     │
  │  · subregion bins       │
  └─────────────────────────┘
```

`Pot_RL` uses dynamic batch-local patch sequence length up to the configured
`max_vertices` bound; it does not pad every batch to the worst-case contour
budget. Empty polygon CLS slots are masked in transformer attention.

| API | Use |
|-----|-----|
| `get_agent("pot")` | Factory → `PotRLAgent` |
| `get_action_and_value(obs)` | Stage 1 only if no NFP tensors |
| `get_action_and_value_from_encoded(...)` | Stage 2 after NFP (PPO path) |
| `get_value(obs)` | Value head only |
| `perform_step(env, agent, obs)` | Full step on one env |

---

## 5. Training pipeline

```
  run/local/train/config.yaml
         │
         v
  BaseTrainer.setup_environment()
         │
         ├── SharedPolygonData  ──►  shared memory
         │
         └── DynamicAsyncVectorEnv
                    │
                    │  make_env() × num_envs
                    v
              TrainingEnv          (generator: nfp_random — fixed in class)
                    │
                    v
              PPOTrainer
                    │
                    │  each step: stage1 → batch NFP → stage2 → step
                    v
              checkpoint (.pt)
                    │
                    └── agent_state_dict + config
```

---

## 6. Evaluation paths

```
                    ┌─────────────────┐
                    │  checkpoint.pt  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              v                             v
     ┌────────────────┐           ┌────────────────┐
     │  PPO / offline │           │  Bench eval    │
     │  TrainingEnv   │           │  InferenceEnv  │
     │  (train only)  │           │  EvalPipeline  │
     └────────────────┘           └────────┬───────┘
                                          │
                                          v
                               get_bench_rollout_exports()
                               (best occupancy + best panel)
```

| Script / class | Env | Purpose |
|----------------|-----|---------|
| `PPOTrainer` | `TrainingEnv` × N | Learn policy |
| `EvalPipeline` | `InferenceEnv` × batch | Cloud/local bench on fixed SHM family |
| `load_agent_model` | `TrainingEnv` | Interactive / analysis |

---

## 7. Observation and action (agent ↔ env)

Vector env batch: `Dict[env_index, obs_dict]`.

**Observation (per env):**

```
  flat_vertices ────────┐
  poly_lengths          ├──► PoT patchify (cyclic last patch) ──► encoder
  poly_ids              │
  normalized_step       ├──► encoder context + value head
  prev_best_occupancy ──┘

  poly_theta ───────────────► agent only: theta_abs for NFP (not encoded)
```

| Field | Shape | Meaning |
|-------|-------|---------|
| `flat_vertices` | `(V, 2)` | Resampled contour vertices (current placement) |
| `poly_lengths` | `(P,)` | Vertices per placed polygon |
| `poly_ids` | `(P,)` | Panel slot id |
| `poly_theta` | `(P,)` | Absolute rotation per piece — **not** fed to PoT; used for NFP `theta_abs` |
| `normalized_step` | scalar | `step_count / episode_length` |
| `prev_best_occupancy` | scalar | Best occupancy in the current rollout segment |

**Patchify:** if the last patch has fewer than `patch_size` vertices, indices wrap along the contour to fill the patch (closes the polyline for the token grid; does not change env geometry). The encoder keeps the batch-local max patch count and raises if it exceeds the configured `max_vertices` budget.

**Action (`env.step`):**

| Field | Meaning |
|-------|---------|
| `polygon_id` | Placed piece to move |
| `theta` | Rotation **delta** (radians), chosen in stage 1 |
| `translation` | 2D offset from stage-2 NFP patch choice |

---

## 7b. Training objective (`max_reward`)

With `environment.max_reward: true` (default in train/eval YAMLs):

- Reward = `max(0, occupancy − rollout_best_so_far)` — sparse, **peak-seeking**.
- Intermediate layouts may worsen; only **new highs** earn reward.
- Episode return is the sum of those positive deltas, not “final occupancy minus initial”.
- The critic sees `prev_best_occupancy` in the value input so it can learn under this shaping.

This matches a large-step budget (e.g. 60 steps × many episodes): optimize the **absolute best** occupancy observed during the rollout, not a dense per-step improvement signal.

---

## 7c. Curriculum (intentional off-buffer warmup)

When `training.ppo.curriculum.enabled`:

1. After reset (and again when an episode ends mid-rollout), the **current policy** runs `K` warmup steps that are **not stored** in the PPO buffer.
2. `curriculum_start_training_phase` resets `step_count` and reward counters. **Panel geometry** stays at the **last warmup step** (latest layout). **`prev_best_occupancy`** and **`best_occupancy_rollout`** are set to the **peak occupancy** seen during warmup (the `max_reward` bar).
3. Effect: the agent self-plays to raise the reward threshold, then the on-policy segment continues from the latest layout — rewards only flow from beating the warmup peak, not from already being at it.

Warmup transitions are deliberately off-policy; only steps after `curriculum_start_training_phase` are trained. GAE uses `terminated` boundaries at episode end before warmup injects the next segment.

---

## 7d. Advantage filtering (intentional off-policy subsampling)

After `training.ppo.adv_filter_start_after_iteration` (default 50), PPO **drops** rollout samples with `|advantage| < eta × moving_max_|A|`.

- Reduces gradient noise from near-zero-advantage steps under sparse `max_reward`.
- This is **not** vanilla on-policy PPO; it is an accepted trade-off for peak-seeking training.
- Tune `eta`, `adv_filter_beta`; monitor `adv_filtered_pct` in logs.

---

## 8. Checkpoints

```
  checkpoint.pt
  ├── agent_state_dict
  └── config
        ├── environment   (data_path, family_size_range, episode_length, …)
        └── model
              ├── agent_type: pot
              └── model_kwargs
                    ├── encoder_config
                    ├── policy_config
                    ├── value_config
                    └── data_patch_size
```

Loaders: `load_agent_model` (strict), `load_agent_state_dict_merged` (strict — raises on shape mismatch; allows widening encoder/value input by one feature for older checkpoints), `apply_env_model_overrides`.

---

## 8b. Train vs bench eval

| Aspect | `PPOTrainer` / `TrainingEnv` | `EvalPipeline` / `InferenceEnv` |
|--------|------------------------------|----------------------------------|
| Init | `GeneratorNFPRandom` each episode | Fixed bench family from SHM |
| Warmup | Curriculum policy warmup (optional) | Warm-start / elite replay (optional) |
| Reward | `max_reward` sparse peaks | Same env rule; metrics on best occupancy |
| Goal | Learn policy | Measure best achievable occupancy |

Bench numbers are **not** a literal replay of the training MDP: they answer “how well does this checkpoint search on a fixed suite under eval sampling settings?” Compare using the same `max_reward` semantics and reported `best_occupancy_ratio`, not raw final-step occupancy alone.

---
