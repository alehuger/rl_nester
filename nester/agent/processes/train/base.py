import os
import random
import time
import yaml
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from datetime import datetime
from abc import ABC, abstractmethod

from nester import PATH_TO_PROJECT
from nester.agent.agent.agent import PotRLAgent
from nester.agent.processes.train.logger import NoOpLogger, RLLogger
from nester.env.utils.vec_envs import DynamicAsyncVectorEnv
from nester.env.utils.sharing_memory import SharedPolygonData
from nester.agent.agent.utils import (
    make_env,
    apply_env_model_overrides,
    load_agent_state_dict_merged,
    try_load_optimizer_state_dict,
)


class BaseTrainer(ABC):
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.load_config()

    def setup(self):
        self.setup_distributed()
        self.setup_environment()
        self.setup_training()
        self.setup_agent()
        self.setup_logger()

    def setup_distributed(self):
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group("gloo")
            self._synced_datetime = self._sync_datetime_via_fs()
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self._synced_datetime = None
        self.is_main = self.rank == 0

    def _sync_datetime_via_fs(self):
        marker = Path("/tmp/.train_datetime")
        if self.rank == 0:
            dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            tmp = Path("/tmp/.train_datetime.tmp")
            tmp.write_text(dt)
            tmp.rename(marker)
            return dt
        for _ in range(120):
            if marker.exists():
                return marker.read_text().strip()
            time.sleep(0.5)
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path) as f:
            global_cfg = yaml.safe_load(f)
        self.global_cfg = global_cfg

        self.env_cfg = global_cfg.get("environment", {})
        self.model_cfg = global_cfg.get("model", {})

        training_cfg = global_cfg.get("training", {})
        self.exp_name = training_cfg.get("exp_name", "default_experiment")
        self.scale_cfg = training_cfg.get("scale", {})
        self.optimizer_cfg = training_cfg.get("optimizer", {})
        self.settings_cfg = training_cfg.get("settings", {})
        self.data_patch_size = self.model_cfg["model_kwargs"]["data_patch_size"]

    # SETUPS
    def setup_training(self):
        datetime_str = self._synced_datetime or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_folder = Path(PATH_TO_PROJECT) / "data/runs/"
        self.exp_folder = self.run_folder / self.exp_name / datetime_str

        if self.is_main:
            print(f"Experiment folder: {self.exp_folder}")
            exp_path = Path(self.exp_folder)
            exp_path.mkdir(parents=True, exist_ok=True)
            (exp_path / "models").mkdir(exist_ok=True)

            cfg_dst = exp_path / Path(self.config_path).name
            try:
                shutil.copy(self.config_path, cfg_dst)
            except Exception:
                with open(cfg_dst, "w") as f:
                    yaml.safe_dump(self.global_cfg, f, sort_keys=False)

            if "cloud" in self.global_cfg:
                cloud_cfg = self.global_cfg.get("cloud", {})
                bucket_name = cloud_cfg.get("bucket")
                if bucket_name:
                    self.sync_to_bucket(bucket_name)

        seed = self.settings_cfg.get("seed", 1) + self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.settings_cfg.get("torch_deterministic", True)

        if self.world_size > 1:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.settings_cfg.get("cuda", True) else "cpu"
            )

    def setup_environment(self):
        data_path_config = self.env_cfg.get("data_path", "")
        data_path = data_path_config if data_path_config.startswith("/") else PATH_TO_PROJECT + data_path_config
        num_envs = self.scale_cfg.get("num_envs", 4)
        seed = self.settings_cfg.get("seed", 1) + self.rank * num_envs
        family_size_range = self.env_cfg.get("family_size_range", [4, 6])

        max_samples = self.env_cfg.get("limit_samples", 100)
        shared_memory_dict = None
        self.shared_data = SharedPolygonData(data_path, max_polygons=max_samples)
        shared_memory_dict = self.shared_data.create_shared_memories()

        episode_length = self.env_cfg.get("episode_length", 100)
        resample_length = self.env_cfg.get("resample_length", 0.1)
        resample_length_nfp = self.env_cfg.get("resample_length_nfp", resample_length)
        limit_samples = self.env_cfg.get("limit_samples", 100)

        self.envs = DynamicAsyncVectorEnv(
            [
                make_env(
                    seed + i,
                    data_path,
                    family_size_range,
                    episode_length,
                    resample_length,
                    resample_length_nfp,
                    limit_samples,
                    shared_memory_dict,
                    env_index=i,
                    nfp_patch_size=int(self.env_cfg.get("nfp_patch_size", self.data_patch_size)),
                    max_reward=bool(self.env_cfg.get("max_reward", False)),
                )
                for i in range(num_envs)
            ]
        )

    def setup_agent(self):
        agent_type = self.model_cfg.get("agent_type", "pot")
        if agent_type != "pot":
            raise ValueError(f"Unsupported agent_type: {agent_type!r}; only 'pot' is supported")
        model_kwargs = apply_env_model_overrides(
            self.env_cfg,
            self.model_cfg.get("model_kwargs", {}),
        )
        common_seed = self.settings_cfg.get("seed", 1)
        random.seed(common_seed)
        np.random.seed(common_seed)
        torch.manual_seed(common_seed)

        self.agent = PotRLAgent(**model_kwargs, device=str(self.device))

        rank_seed = common_seed + self.rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)

        mounted_folder = self.model_cfg.get("mounted_folder", "")
        pretrained_model_path = self.model_cfg.get("pretrained_model_path", None)

        learning_rate = float(self.optimizer_cfg.get("learning_rate", "3e-4"))
        optimizer_eps = float(self.optimizer_cfg.get("eps", "1e-5"))

        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate, eps=optimizer_eps)

        self.resume_iteration = 0
        self.global_step_init = 0
        resume_full = bool(self.settings_cfg.get("resume_from_checkpoint", False))

        if pretrained_model_path:
            if self.is_main:
                print(f"Loading pretrained model from {pretrained_model_path}")
            base = (
                Path(mounted_folder)
                if str(mounted_folder).startswith("/")
                else Path(PATH_TO_PROJECT) / str(mounted_folder)
            )
            run_dir = base / str(pretrained_model_path).lstrip("/")
            models_dir = run_dir / "models"
            reload_name = self.model_cfg.get("pretrained_checkpoint", None)
            if reload_name in (None, "", "best"):
                ckpt = models_dir / "best.pt"
                if not ckpt.exists():
                    ckpts = sorted(models_dir.glob("checkpoint_*.pt"))
                    ckpt = ckpts[-1] if ckpts else ckpt
            else:
                stem = str(reload_name).strip()
                if not stem.endswith(".pt"):
                    stem = f"{stem}.pt"
                ckpt = models_dir / stem
            if ckpt.exists():
                if self.is_main:
                    print(f"Loading pretrained model from {ckpt}")
                checkpoint = torch.load(ckpt, map_location=self.device, weights_only=False)
                load_agent_state_dict_merged(self.agent, checkpoint["agent_state_dict"])
                if resume_full:
                    try_load_optimizer_state_dict(
                        self.optimizer, checkpoint.get("optimizer_state_dict")
                    )
                    self.global_step_init = int(checkpoint.get("global_step", 0))
                    self.resume_iteration = int(checkpoint.get("update", 0))
                    if self.is_main:
                        print(
                            f"Resume: global_step={self.global_step_init}, "
                            f"iteration={self.resume_iteration}, "
                            f"optimizer_lr={self.optimizer.param_groups[0]['lr']}"
                        )
                elif self.is_main:
                    print("resume_from_checkpoint=false: weights only, fresh optimizer schedule")
            else:
                raise ValueError(f"Pretrained model not found at {ckpt}")

    def setup_logger(self):
        if self.is_main:
            self.logger = RLLogger(self.exp_folder)
        else:
            self.logger = NoOpLogger()

    # BUCKET SYNCING
    def sync_to_bucket(self, bucket_name: str):
        if not self.is_main:
            return
        from google.cloud import storage

        if not bucket_name:
            return

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        datetime_str = self.exp_folder.name
        remote_prefix = f"runs/{self.exp_name}/{datetime_str}"

        print(f"Initializing bucket structure at gs://{bucket_name}/{remote_prefix}...")

        # Upload config to make folder visible
        config_path = self.exp_folder / Path(self.config_path).name
        if config_path.exists():
            blob = bucket.blob(f"{remote_prefix}/config.yaml")
            blob.upload_from_filename(str(config_path))
            print("  ✓ Uploaded config.yaml")
        else:
            print("  ⚠ Warning: config.yaml not found, folder may not be visible")

        print(f"  ✓ Bucket structure initialized at gs://{bucket_name}/{remote_prefix}")
        print("  Note: Folders in GCS appear when files are uploaded to them")
        print("✅ Sync to bucket completed successfully!")

    def upload_to_bucket(self, bucket_name: str, update: int):
        if not self.is_main:
            return
        from google.cloud import storage
        from google.cloud.exceptions import GoogleCloudError
        import time

        if not bucket_name:
            return

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        datetime_str = self.exp_folder.name
        remote_prefix = f"runs/{self.exp_name}/{datetime_str}"

        upload_tb_logs = self.global_cfg.get("cloud", {}).get("upload_tb_logs", True)
        skip_recent_seconds = self.global_cfg.get("cloud", {}).get("skip_recent_files_seconds", 60)

        print(f"Uploading experiment data to gs://{bucket_name}/{remote_prefix} (update {update})...")

        # Upload config
        config_path = self.exp_folder / Path(self.config_path).name
        if config_path.exists():
            try:
                blob = bucket.blob(f"{remote_prefix}/config.yaml")
                blob.upload_from_filename(str(config_path))
                print("  ✓ Uploaded config.yaml")
            except Exception as e:
                print(f"  ✗ Failed to upload config.yaml: {e}")

        # Upload all experiment files (models, logs, etc.)
        uploaded_count = 0
        skipped_count = 0
        failed_count = 0
        current_time = time.time()

        for root, _, files in os.walk(self.exp_folder):
            for f in files:
                local_path = os.path.join(root, f)
                rel_path = os.path.relpath(local_path, start=str(self.exp_folder))
                remote_path = f"{remote_prefix}/{rel_path}"

                # Skip TensorBoard event files if disabled or if recently modified
                if f.endswith(".tfevents."):
                    if not upload_tb_logs:
                        skipped_count += 1
                        continue
                    try:
                        file_mtime = os.path.getmtime(local_path)
                        if current_time - file_mtime < skip_recent_seconds:
                            skipped_count += 1
                            continue
                    except Exception:
                        pass

                try:
                    blob = bucket.blob(remote_path)
                    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)

                    # Upload file (GCS client handles timeouts internally)
                    blob.upload_from_filename(local_path)
                    uploaded_count += 1
                    if file_size_mb > 50:
                        print(f"    Uploaded large file: {rel_path} ({file_size_mb:.1f}MB)")
                except GoogleCloudError as e:
                    failed_count += 1
                    print(f"  ✗ Failed to upload {rel_path}: {e}")
                except Exception as e:
                    failed_count += 1
                    print(f"  ✗ Error uploading {rel_path}: {e}")

        print(
            f"  ✓ Uploaded {uploaded_count} files"
            + (f", skipped {skipped_count}" if skipped_count > 0 else "")
            + (f", failed {failed_count}" if failed_count > 0 else "")
        )
        print(f"Upload completed for update {update}")

    def save_model(self, update: int, global_step: int, best: bool = False):
        if not self.is_main:
            return
        filefolder = self.exp_folder / "models"
        filepath = filefolder / f"checkpoint_{global_step}.pt" if not best else filefolder / "best.pt"
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        agent_state_dict = self.agent.state_dict()

        # Verify transformer layers are included
        transformer_keys = [k for k in agent_state_dict.keys() if "transformer_blocks" in k]
        if transformer_keys:
            num_layers = len(
                set(k.split(".")[2] for k in transformer_keys if len(k.split(".")) > 2 and k.split(".")[2].isdigit())
            )
            print(f"Saving model with {num_layers} transformer layers ({len(transformer_keys)} transformer parameters)")

        checkpoint = {
            "agent_state_dict": agent_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.global_cfg,
            "update": update,
            "global_step": global_step,
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def _close_shared_data(self):
        shared = getattr(self, "shared_data", None)
        if shared is None:
            return
        self.shared_data = None
        try:
            shared.close()
        except Exception:
            pass

    def __del__(self):
        self._close_shared_data()

    @abstractmethod
    def train(self):
        pass
