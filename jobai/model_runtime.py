"""Lazy GPU helpers: importing this module never loads a model or downloads data."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from jobai.forecasting import digest_json, read_json, sha256, write_json


def check_fast_kernels(model_id, required=True):
    if not model_id.startswith("Qwen/Qwen3.5-"):
        return {}
    checks = {"causal_conv1d": "causal_conv1d_fn", "fla.ops.gated_delta_rule": "chunk_gated_delta_rule"}
    status = {}
    for module, function in checks.items():
        try:
            status[module] = callable(getattr(importlib.import_module(module), function))
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            status[module] = False
            print(f"Kernel unavailable: {module}: {exc}")
    if required and not all(status.values()):
        raise RuntimeError("Qwen3.5 fast kernels are unavailable. Run notebook 06's optional kernel-install cell, "
                           "restart the runtime, and rerun setup. Do not start a 75k run on the slow fallback. "
                           "Changing require_fast_kernels is an explicit opt-in to the slower path.")
    return status


def base_directory(repo, model_id, download=False):
    path = Path(repo) / "models/base" / model_id.replace("/", "--")
    ready = path / ".download_complete"
    if not ready.exists():
        assert download, f"Missing cached base model: {path}. Download it with 06 or restore its cache; 07 never silently downloads."
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().model_info(model_id).sha
        path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_id, revision=revision, local_dir=path)
        write_json(path / "jobai_base_snapshot.json", {"model_id": model_id, "revision": revision})
    assert (path / "config.json").is_file(), f"Incomplete base cache: {path}"
    index = path / "model.safetensors.index.json"
    weights = set(read_json(index)["weight_map"].values()) if index.exists() else {"model.safetensors"}
    assert all((path / name).is_file() and (path / name).stat().st_size > 0 for name in weights), f"Missing base weight shards: {path}"
    if not ready.exists():
        ready.write_text(model_id + "\n", encoding="utf-8")
    return path


def tokenizer_identity(tokenizer):
    return digest_json({"vocab": tokenizer.get_vocab(), "chat_template": tokenizer.chat_template,
                        "special_tokens": tokenizer.special_tokens_map,
                        "class": type(tokenizer).__name__})


def adapter_identity(path):
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix == ".safetensors" or p.name in ("adapter_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")))
    assert any(p.suffix == ".safetensors" for p in files), f"Missing adapter weights: {path}"
    return digest_json({p.name: sha256(p) for p in files})


def load_tokenizer(base_dir, model_id, adapter_dir=None):
    from transformers import AutoProcessor, AutoTokenizer
    source = adapter_dir or base_dir
    # Historical adapters contain their own tokenizers. Do not substitute the
    # new model's tokenizer when comparing different base model families.
    if adapter_dir is not None:
        return AutoTokenizer.from_pretrained(source, local_files_only=True), None
    if model_id.startswith("Qwen/Qwen3.5-"):
        processor = AutoProcessor.from_pretrained(source, local_files_only=True)
        return processor.tokenizer, processor
    return AutoTokenizer.from_pretrained(source, local_files_only=True), None


def load_quantized_base(base_dir, architecture, bf16):
    import torch
    import transformers
    from transformers import BitsAndBytesConfig
    model_class = (transformers.AutoModelForMultimodalLM if architecture == "multimodal_text_only"
                   else transformers.AutoModelForCausalLM)
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = model_class.from_pretrained(
        base_dir, local_files_only=True, device_map={"": 0}, dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
    )
    assert all(p.device.type == "cuda" for p in model.parameters()), "CPU/disk offloading detected; stop before slow training."
    return model


def attach_language_lora(model, config, checkpointing):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    suffixes = set(config["target_modules"])
    # Exact full module names prevent similarly named vision projections from
    # receiving trainable adapters. Frozen base weights remain frozen.
    targets = [name for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in suffixes
               and not any(marker in name.lower() for marker in ("visual", "vision", "mtp"))]
    assert targets, "No matching language LoRA modules. Inspect the model architecture."
    model = get_peft_model(model, LoraConfig(
        r=int(config["r"]), lora_alpha=int(config["alpha"]), lora_dropout=float(config["dropout"]),
        target_modules=targets, bias=config["bias"], task_type=config["task_type"]))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable and all("lora_" in name and not any(v in name.lower() for v in ("vision", "visual")) for name in trainable)
    model.config.use_cache = False
    return model, targets


def resolve_repo_path(repo, value):
    """Translate saved Colab paths when the repository has been copied locally."""
    path = Path(value)
    if path.is_absolute():
        prefix = "/content/drive/MyDrive/JobAI/"
        if str(path).startswith(prefix):
            return Path(repo) / str(path)[len(prefix):]
        return path
    return Path(repo) / path


def discover_comparison_runs(repo, comparison_cfg, model_cfg, card):
    repo = Path(repo)
    current_id = comparison_cfg.get("current_run_id")
    source = (repo / "data/manifests/finetune_runs" / f"{current_id}.json" if current_id
              else repo / "data/manifests/finetune_run_manifest.json")
    assert source.is_file(), "Run the updated notebook 06 once; the shared-run manifest is missing."
    current = read_json(source)
    assert current["base_model"] == model_cfg["candidate_to_run"], "Latest run uses a different model; select current_run_id explicitly."
    assert current["training_horizons"] == [1, 2, 4], "The current run must be a shared adapter."
    assert current["prompt_schema"] == "five_table_shared_v1", "The latest pointer is an older experiment."
    for split in ("train", "validation"):
        assert current[f"{split}_split_sha256"] == card["files"][split]["sha256"], f"Current adapter uses a different {split} dataset."
    assert current["feature_window_quarters"] == card["feature_window_quarters"]
    assert not current.get("test_data_used", False)
    current = {**current, "source_manifest": str(source.relative_to(repo)), "role": "current_shared",
               "model_label": "qwen35_4b_shared_" + str(current["train_examples_used"]),
               "train_count": current["train_examples_used"], "history_quarters": current["feature_window_quarters"],
               "max_seq_length": current["model_config"]["training"]["max_seq_length"],
               "provenance": "training_manifest"}
    runs = [current]
    for spec in comparison_cfg["previous_runs"]:
        run_id = spec["run_id"]
        candidates = [repo / "data/manifests/finetune_runs" / f"{run_id}.json",
                      repo / "reports/model_evaluations" / run_id / "evaluation_manifest.json"]
        source = next((p for p in candidates if p.is_file()), None)
        assert source, f"Missing historical evidence for {run_id}; restore it or explicitly edit configs/comparison.yaml."
        evidence = read_json(source)
        assert evidence["base_model"] == spec["base_model"]
        assert evidence.get("train_examples_used", evidence.get("train_examples")) == spec["train_examples"], "Historical training count disagrees with comparison roster."
        assert not evidence.get("test_data_used", evidence.get("test_data_used_for_training", False))
        assert sorted(evidence.get("training_horizons", [1, 2, 4])) == [1, 2, 4]
        history = evidence.get("feature_window_quarters", spec["history_quarters"])
        schema = evidence.get("prompt_schema", spec["prompt_schema"])
        runs.append({**evidence, "base_model": spec["base_model"], "model_label": spec["label"],
                     "role": "previous_shared", "train_count": spec["train_examples"],
                     "history_quarters": int(history), "prompt_schema": schema,
                     "max_seq_length": spec["max_seq_length"], "source_manifest": str(source.relative_to(repo)),
                     "provenance": "training_manifest" if source == candidates[0] and
                         {"feature_window_quarters", "prompt_schema"}.issubset(evidence)
                         else "explicit_historical_compatibility_declaration"})
    for run in runs:
        run["adapter_dir"] = resolve_repo_path(repo, run["adapter_path"])
        assert (run["adapter_dir"] / "adapter_config.json").is_file(), f"Missing adapter: {run['adapter_dir']}"
        run["adapter_sha256"] = adapter_identity(run["adapter_dir"])
    assert len({run["model_label"] for run in runs}) == len(runs)
    return runs
