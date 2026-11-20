from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess

import torch


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata.

    TODO: collect at least:
      - Git commit SHA and whether the working tree is dirty;
      - identity of the prepared training + validation manifests;
      - Python, PyTorch, torchvision, DVC and MLflow versions;
      - OS/platform and CUDA runtime (when available);
      - compute device; random seed; image size; training parameters;
      - MLflow Run ID.

    Hints: subprocess, platform, importlib.metadata, torch,
    training_data_identity().
    """
    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    try:
        git_commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        git_status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        git_commit_result = None
        git_status_result = None

    git_commit = (
        git_commit_result.stdout.strip()
        if git_commit_result is not None and git_commit_result.returncode == 0
        else "unavailable"
    )
    git_dirty = (
        bool(git_status_result.stdout.strip())
        if git_status_result is not None and git_status_result.returncode == 0
        else "unavailable"
    )

    return {
        "data": {
            "training_data_identity": training_data_identity(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "dvc": package_version("dvc"),
            "mlflow": package_version("mlflow"),
            "os": platform.platform(),
            "cuda": torch.version.cuda or "unavailable",
        },
        "configuration": {
            "image_size": int(params["data"]["image_size"]),
            "train": params["train"],
            "augmentation": params["augmentation"],
        },
        "execution": {
            "device": device,
            "random_state": int(params["train"]["random_state"]),
            "mlflow_run_id": mlflow_run_id,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
