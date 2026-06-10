# Run — entrypoints

### [README](../README.md)

Launchers outside the `nester` package: configs, local scripts, data factory, GCP Batch.

```
run/data_factory ──► data/db/processed/*.npz
run/local        ──► train / eval
run/cloud        ──► Docker + GCS + Batch
```

## Layout

```
run/
├── data_factory/preprocessing.py
├── local/train/   config.yaml  run.py
├── local/eval/    config.yaml  run.py
└── cloud/         Dockerfile, gcs.py, batch.py, train/, eval/
```

## Local

```bash
python run/local/train/run.py [config.yaml]
python run/local/eval/run.py [--config config.yaml]
```

Checkpoints: `data/runs/<exp_name>/<timestamp>/models/`.

## Data factory (optional)

```bash
python run/data_factory/preprocessing.py
```

## Cloud (GCP)

Fill in `cloud:` in `run/cloud/train/config.yaml` or `run/cloud/eval/config.yaml`, then:

```bash
./run/cloud/train/build.sh [config.yaml]
./run/cloud/eval/build.sh  [config.yaml]
```

Artifacts sync to `gs://<bucket>/runs/...`.
