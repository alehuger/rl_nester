"""GCP Batch helpers shared by train and bench eval deploy scripts."""

from __future__ import annotations

import json
import time

import yaml
from google.cloud import batch_v1

from nester.agent.processes.eval.layout import (
    bench_checkpoint_steps_slug,
    bench_eval_budget_strat_tag_from_config,
    bench_eval_name_segment,
)

GCP_BATCH_JOB_ID_MAX_LEN = 63


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_docker_image(config: dict) -> str:
    cloud = config["cloud"]
    return (
        f"{cloud['region']}-docker.pkg.dev/{cloud['project_id']}/"
        f"{cloud['repository_name']}/{cloud['docker_tag']}"
    )


def submit_batch_container_job(config, docker_image, job_id, extra_env=None):
    cloud = config["cloud"]
    client = batch_v1.BatchServiceClient()

    env_vars = {"CONFIG_JSON": json.dumps(config)}
    if extra_env:
        env_vars.update(extra_env)

    accelerator = batch_v1.AllocationPolicy.Accelerator(
        type_=cloud["gpu_type"], count=cloud["gpu_count"]
    )

    task = batch_v1.TaskSpec(
        runnables=[
            batch_v1.Runnable(
                container=batch_v1.Runnable.Container(
                    image_uri=docker_image,
                    options="--shm-size=16g --ipc=host",
                ),
                environment=batch_v1.Environment(variables=env_vars),
            ),
        ],
        max_retry_count=0,
    )

    provisioning_map = {
        "SPOT": batch_v1.AllocationPolicy.ProvisioningModel.SPOT,
        "STANDARD": batch_v1.AllocationPolicy.ProvisioningModel.STANDARD,
    }
    provisioning_model = provisioning_map.get(
        cloud.get("provisioning_model", "SPOT"),
        batch_v1.AllocationPolicy.ProvisioningModel.SPOT,
    )

    instances = [
        batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
            install_gpu_drivers=True,
            policy=batch_v1.AllocationPolicy.InstancePolicy(
                machine_type=cloud["machine_type"],
                provisioning_model=provisioning_model,
                accelerators=[accelerator],
                boot_disk=batch_v1.AllocationPolicy.Disk(size_gb=100),
            ),
        )
    ]

    allocation_policy = batch_v1.AllocationPolicy(instances=instances)
    if "zone" in cloud:
        allocation_policy.location = batch_v1.AllocationPolicy.LocationPolicy(
            allowed_locations=[f"zones/{cloud['zone']}"]
        )

    job = batch_v1.Job(
        task_groups=[batch_v1.TaskGroup(task_spec=task)],
        allocation_policy=allocation_policy,
        logs_policy=batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        ),
    )

    request = batch_v1.CreateJobRequest(
        parent=f"projects/{cloud['project_id']}/locations/{cloud['region']}",
        job=job,
        job_id=job_id,
    )
    response = client.create_job(request=request)
    print("✓ Job created:", response.name)
    return response


def _is_run_timestamp_segment(seg: str) -> bool:
    if "_" not in seg:
        return False
    a, b = seg.split("_", 1)
    return len(a) == 8 and a.isdigit() and b.isdigit() and len(b) >= 4


def _model_folder_job_slug(model_folder, max_len: int) -> str:
    parts = [p for p in str(model_folder).strip("/").split("/") if p]
    if parts and parts[0].lower() == "nester":
        parts = parts[1:]
    if parts and _is_run_timestamp_segment(parts[-1]):
        parts = parts[:-1]
    segs = [p.lower().replace("_", "-") for p in parts]
    base = "-".join(segs) if segs else "model"
    while "--" in base:
        base = base.replace("--", "-")
    base = base.strip("-") or "model"
    if len(base) <= max_len:
        return base
    tail = base[-max_len:].strip("-")
    return tail if tail else base[-max_len:]


def submit_training_job(config, docker_image):
    safe_name = config["training"]["exp_name"].lower().replace("_", "-").replace("/", "-")
    job_id = f"rl-{safe_name}-{int(time.time())}"
    return submit_batch_container_job(config, docker_image, job_id)


def submit_bench_eval_job(config, docker_image):
    model = config.get("model", {})
    settings = config.get("settings", {})
    data = config.get("data", {})
    ts = int(time.time())
    ck_slug = bench_checkpoint_steps_slug(model.get("checkpoint_file"))
    budget = bench_eval_budget_strat_tag_from_config(settings, data)
    eval_name = bench_eval_name_segment(settings.get("eval_name"))
    prefix = "bench-eval-"
    name_part = f"-{eval_name}" if eval_name else ""
    suffix = f"-ckpt-{ck_slug}-{budget}{name_part}-{ts}"
    room = GCP_BATCH_JOB_ID_MAX_LEN - len(prefix) - len(suffix)
    room = max(room, 4)
    base = _model_folder_job_slug(model.get("model_folder"), room)
    job_id = f"{prefix}{base}{suffix}".lower()
    if len(job_id) > GCP_BATCH_JOB_ID_MAX_LEN:
        job_id = job_id[:GCP_BATCH_JOB_ID_MAX_LEN].rstrip("-")
        while job_id and not job_id[-1].isalnum():
            job_id = job_id[:-1]
        if not job_id:
            job_id = f"bench-eval-{ts}"
        elif not job_id[-1].isalnum():
            job_id = f"{job_id}a"
    return submit_batch_container_job(
        config,
        docker_image,
        job_id,
        extra_env={"NESTER_ENTRYPOINT_MODE": "bench_eval"},
    )
