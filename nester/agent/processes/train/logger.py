from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class RLLogger:
    def __init__(self, exp_folder: str = None):
        self.writer = SummaryWriter(exp_folder)
        self.exp_folder = exp_folder
        self.metrics_dir = Path(exp_folder) / "metrics" if exp_folder else None
        if self.metrics_dir is not None:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self._best_occ_csv = self.metrics_dir / "best_occupancy.csv"

    def log_end_update(
        self,
        v_loss,
        pg_loss,
        entropy_loss,
        old_approx_kl,
        approx_kl,
        clipfrac,
        global_step,
        entropy_stage1_item=None,
        entropy_stage1_rotation=None,
        entropy_stage2_place=None,
        clipfrac_stage1_item=None,
        clipfrac_stage1_rotation=None,
        clipfrac_stage2_place=None,
        approx_kl_stage1_item=None,
        approx_kl_stage1_rotation=None,
        approx_kl_stage2_place=None,
        adv_filtered_pct=None,
        adv_filter_retained_ratio=None,
        logit_abs_max_item=None,
        logit_abs_max_rotation=None,
        logit_abs_max_place=None,
    ):
        self.writer.add_scalar("losses/value_loss", v_loss, global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss, global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        self.writer.add_scalar("losses/clipfrac", clipfrac, global_step)

        if entropy_stage1_item is not None:
            self.writer.add_scalar("entropy/stage1_item", entropy_stage1_item, global_step)
        if entropy_stage1_rotation is not None:
            self.writer.add_scalar("entropy/stage1_rotation", entropy_stage1_rotation, global_step)
        if clipfrac_stage1_item is not None:
            self.writer.add_scalar("clipfrac/stage1_item", clipfrac_stage1_item, global_step)
        if clipfrac_stage1_rotation is not None:
            self.writer.add_scalar("clipfrac/stage1_rotation", clipfrac_stage1_rotation, global_step)
        if clipfrac_stage2_place is not None:
            self.writer.add_scalar("clipfrac/stage2_place", clipfrac_stage2_place, global_step)
        if entropy_stage2_place is not None:
            self.writer.add_scalar("entropy/stage2_place", entropy_stage2_place, global_step)
        if approx_kl_stage1_item is not None:
            self.writer.add_scalar("approx_kl/stage1_item", approx_kl_stage1_item, global_step)
        if approx_kl_stage1_rotation is not None:
            self.writer.add_scalar("approx_kl/stage1_rotation", approx_kl_stage1_rotation, global_step)
        if approx_kl_stage2_place is not None:
            self.writer.add_scalar("approx_kl/stage2_place", approx_kl_stage2_place, global_step)
        if adv_filtered_pct is not None:
            self.writer.add_scalar("ppo/adv_filtered_pct", adv_filtered_pct, global_step)
        if adv_filter_retained_ratio is not None:
            self.writer.add_scalar("ppo/adv_filter_retained_ratio", adv_filter_retained_ratio, global_step)
        if logit_abs_max_item is not None:
            self.writer.add_scalar("logit_abs_max/item", logit_abs_max_item, global_step)
        if logit_abs_max_rotation is not None:
            self.writer.add_scalar("logit_abs_max/rotation", logit_abs_max_rotation, global_step)
        if logit_abs_max_place is not None:
            self.writer.add_scalar("logit_abs_max/place", logit_abs_max_place, global_step)

    def log_curriculum_metrics(
        self,
        global_step,
        *,
        warmup_steps: float,
        init_occupancy: float,
        best_warmup_occupancy: float,
    ):
        self.writer.add_scalar("curriculum/warmup_steps", warmup_steps, global_step)
        self.writer.add_scalar("curriculum/init_occupancy", init_occupancy, global_step)
        self.writer.add_scalar(
            "curriculum/best_warmup_occupancy", best_warmup_occupancy, global_step
        )

    def log_training_metrics(
        self,
        explained_var,
        lrnow,
        sps,
        global_step,
        gpu_stats=None,
        cpu_stats=None,
        timing_stats=None,
    ):
        self._loss_charts(explained_var, lrnow, sps, global_step)
        self._log_gpu_stats(global_step, gpu_stats)
        self._log_cpu_stats(global_step, cpu_stats)
        self._log_timing_stats(global_step, timing_stats)

    def log_rollout_metrics(
        self,
        increase_occupancy_ratio,
        final_occupancy_ratio,
        global_step,
        episodic_return=None,
        validation_score=None,
        best_occupancy_ratio=None,
        nonzero_reward_ratio=None,
        final_reward_cumulative=None,
    ):
        self.writer.add_scalar(
            "rewards/mean_increase_occupancy_ratio",
            increase_occupancy_ratio.mean().item(),
            global_step,
        )
        self.writer.add_scalar(
            "rewards/final_occupancy_ratio",
            final_occupancy_ratio.mean().item(),
            global_step,
        )
        p5_final_occupancy_ratio = np.percentile(final_occupancy_ratio, 95)
        q1_final_occupancy_ratio = np.percentile(final_occupancy_ratio, 25)
        q3_final_occupancy_ratio = np.percentile(final_occupancy_ratio, 75)
        self.writer.add_scalar("rewards/final_occupancy_ratio_p5", p5_final_occupancy_ratio, global_step)
        self.writer.add_scalar("rewards/final_occupancy_ratio_q1", q1_final_occupancy_ratio, global_step)
        self.writer.add_scalar("rewards/final_occupancy_ratio_q3", q3_final_occupancy_ratio, global_step)

        p5_episodic_return = np.percentile(episodic_return, 95)
        q3_episodic_return = np.percentile(episodic_return, 75)
        q1_episodic_return = np.percentile(episodic_return, 25)
        self.writer.add_scalar("rewards/episodic_return", episodic_return.mean().item(), global_step)
        self.writer.add_scalar("rewards/episodic_return_p5", p5_episodic_return, global_step)
        self.writer.add_scalar("rewards/episodic_return_q3", q3_episodic_return, global_step)
        self.writer.add_scalar("rewards/episodic_return_q1", q1_episodic_return, global_step)

        self.writer.add_scalar(
            "rewards/validation_score", validation_score.mean().item(), global_step
        )
        if final_reward_cumulative is not None:
            self.writer.add_scalar(
                "rewards/final_reward_cumulative",
                final_reward_cumulative.mean().item(),
                global_step,
            )
        if validation_score is not None:
            self._log_best_occupancy_metrics(
                np.asarray(
                    best_occupancy_ratio if best_occupancy_ratio is not None else validation_score,
                    dtype=np.float64,
                ).reshape(-1),
                global_step,
            )
        if nonzero_reward_ratio is not None:
            arr = np.asarray(nonzero_reward_ratio, dtype=np.float64).reshape(-1)
            arr = arr[~np.isnan(arr)]
            if arr.size > 0:
                self.writer.add_scalar("metrics/nonzero_reward_pct", float(np.mean(arr)) * 100, global_step)

    def _log_best_occupancy_metrics(self, values: np.ndarray, global_step: int):
        self._log_distribution_metrics("metrics/best_occupancy", values, global_step)

        values = values[~np.isnan(values)]
        if values.size == 0:
            return
        mean = float(np.mean(values))
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        p95 = float(np.percentile(values, 95))
        p99 = float(np.percentile(values, 99))
        if self.metrics_dir is None:
            return
        path = self._best_occ_csv
        new_file = not path.exists()
        line = f"{global_step},{mean},{q1},{q3},{p95},{p99},{values.size}\n"
        with open(path, "a") as f:
            if new_file:
                f.write("global_step,mean,q1,q3,p95,p99,n\n")
            f.write(line)

    def _log_distribution_metrics(self, prefix: str, values: np.ndarray, global_step: int):
        values = values[~np.isnan(values)]
        if values.size == 0:
            return
        self.writer.add_scalar(f"{prefix}/mean", float(np.mean(values)), global_step)
        self.writer.add_scalar(f"{prefix}/q1", float(np.percentile(values, 25)), global_step)
        self.writer.add_scalar(f"{prefix}/q3", float(np.percentile(values, 75)), global_step)
        self.writer.add_scalar(f"{prefix}/p95", float(np.percentile(values, 95)), global_step)
        self.writer.add_scalar(f"{prefix}/p99", float(np.percentile(values, 99)), global_step)

    def _loss_charts(self, explained_var, lrnow, sps, global_step):
        self.writer.add_scalar("losses/explained_var", explained_var, global_step)
        self.writer.add_scalar("charts/learning_rate", lrnow, global_step)
        self.writer.add_scalar("charts/SPS", sps, global_step)

    def _log_gpu_stats(self, global_step: int, gpu_stats=None):
        if gpu_stats is None or not gpu_stats:
            return

        avg_gpu_util = gpu_stats.get("avg_gpu_util", 0)
        max_gpu_util = gpu_stats.get("max_gpu_util", 0)
        avg_mem_used = gpu_stats.get("avg_memory_used_mb", 0)
        max_mem_used = gpu_stats.get("max_memory_used_mb", 0)

        print(
            f"GPU Stats at step {global_step}: "
            f"Avg Util: {avg_gpu_util:.2f}%, Max Util: {max_gpu_util:.2f}%, "
            f"Avg VRAM: {avg_mem_used:.2f}MB, Max VRAM: {max_mem_used:.2f}MB"
        )

        self.writer.add_scalar("hardware/gpu_util_avg", avg_gpu_util, global_step)
        self.writer.add_scalar("hardware/gpu_util_max", max_gpu_util, global_step)
        self.writer.add_scalar("hardware/memory_used_mb_avg", avg_mem_used, global_step)
        self.writer.add_scalar("hardware/memory_used_mb_max", max_mem_used, global_step)
        self.writer.add_scalar(
            "hardware/temperature_avg",
            gpu_stats.get("avg_temperature", 0),
            global_step,
        )
        self.writer.add_scalar(
            "hardware/temperature_max",
            gpu_stats.get("max_temperature", 0),
            global_step,
        )
        self.writer.add_scalar("hardware/power_avg", gpu_stats.get("avg_power_w", 0), global_step)
        self.writer.add_scalar("hardware/power_max", gpu_stats.get("max_power_w", 0), global_step)

    def _log_cpu_stats(self, global_step: int, cpu_stats=None):
        if cpu_stats is None or not cpu_stats:
            return
        self.writer.add_scalar(
            "hardware/cpu_util_avg",
            cpu_stats.get("avg_cpu_util", 0),
            global_step,
        )
        self.writer.add_scalar(
            "hardware/cpu_util_max",
            cpu_stats.get("max_cpu_util", 0),
            global_step,
        )
        self.writer.add_scalar(
            "hardware/cpu_memory_used_mb_avg",
            cpu_stats.get("avg_memory_used_mb", 0),
            global_step,
        )
        self.writer.add_scalar(
            "hardware/cpu_memory_used_mb_max",
            cpu_stats.get("max_memory_used_mb", 0),
            global_step,
        )

    def _log_timing_stats(self, global_step: int, timing_stats=None):
        if timing_stats is None or not timing_stats:
            return
        for op_name, stats in timing_stats.items():
            if isinstance(stats, dict):
                self.writer.add_scalar(
                    f"timing/{op_name}_per_step_ms",
                    stats.get("mean_per_step_ms", 0),
                    global_step,
                )


class NoOpLogger:
    """Silent logger for non-main distributed ranks."""

    def __getattr__(self, name):
        return lambda *a, **kw: None
