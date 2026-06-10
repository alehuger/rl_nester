# Nester — Architecture

### [README](../README.md) · [Environment](./env/ENV.md) · [Agent](./agent/AGENT.md)

**The `nester` package has two subpackages: `env` (MDP + geometry) and `agent` (policy + training).** `env` never imports `agent`. Launchers, dataset builders, and artifacts live outside the package — see [run/README.md](../run/README.md).

---

## 🧭 Inside `nester`

```
                    ┌─────────────┐
                    │  nester.env │  Gym MDP, NFP, panel
                    └──────┬──────┘
                           │  observations, step, compute_nfp
                           ▼
                    ┌─────────────┐
                    │ nester.agent│  PoT, PPO, bench
                    └─────────────┘
```

| Subpackage | Owns | Must not import |
|------------|------|-----------------|
| `env` | Generators, panel, rewards, CGAL NFP | `agent` |
| `agent` | Networks, `PotRLAgent`, train/eval loops | — (imports `env`) |

---

## 🔄 Control loop (env ↔ agent)

Layout **improvement**: pieces are placed at reset; each step repositions one piece.

```
obs ──► agent stage 1 ──► polygon_id, Δθ
              │
              ▼
         env.compute_nfp
              │
              ▼
         agent stage 2 ──► translation on NFP patches
              │
              ▼
         env.step ──► obs, reward
```

Details: [env/ENV.md](./env/ENV.md) (observation, action, reward) · [agent/AGENT.md](./agent/AGENT.md) (PoT, PPO, checkpoints).

---

## 📦 Layout

```
nester/
├── env/                 simulation & geometry
│   ├── env/             BaseEnv, TrainingEnv, InferenceEnv, generators
│   ├── panel/           Panel, NFP (cgal, utils)
│   └── utils/           vec envs, shared-memory loaders
└── agent/               learning & decisions
    ├── agent/           PotRLAgent, utils, model/
    └── processes/       train/ (PPO), eval/ (bench)
```

---

## 📐 Principles

1. **Geometry in `env.panel`** — feasibility and NFP stay out of torch code.
2. **Pure models** — `agent.agent.model` has no Gym import.
3. **Thin orchestration** — `processes/` coordinates rollouts; placement rules stay in `env`.
4. **Two-stage actions** — rotation choice, then NFP-conditioned placement (matches `env.step`).
5. **One public surface per subpackage** — re-exports in `nester/env/__init__.py` and `nester/agent/__init__.py`.

---

## 🔌 Imports

```python
from nester.env import TrainingEnv, InferenceEnv, DynamicAsyncVectorEnv
from nester.agent import PotRLAgent, PPOTrainer, EvalPipeline
```

```text
nester.env.*
nester.env.env.*
nester.env.panel.*
nester.agent.*
nester.agent.agent.*
nester.agent.agent.model.*
nester.agent.processes.train.*
nester.agent.processes.eval.*
```
