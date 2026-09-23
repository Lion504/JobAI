"""Explicit active profiles; historical experiment files remain unchanged."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_MODEL_CONFIG = "configs/model_qwen3_4b_shared.yaml"
DEFAULT_COMPARISON_CONFIG = "configs/comparison_final.yaml"


def load_profile(repo, model_config=None):
    repo = Path(repo)
    path = repo / (model_config or os.environ.get("JOBAI_MODEL_CONFIG") or DEFAULT_MODEL_CONFIG)
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    eval_path = repo / model.get("evaluation_config", "configs/eval.yaml")
    evaluation = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    return path, model, eval_path, evaluation
