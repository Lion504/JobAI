"""Resumable, model-specific forecast generation for notebook 07."""
from __future__ import annotations

import gc
import importlib.metadata
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd

from jobai.forecasting import (KEYS, digest_json, input_view, parse_prediction,
                               prompt_messages, read_json, sha256, write_json)
from jobai.model_runtime import (base_directory, check_fast_kernels, load_quantized_base,
                                 load_tokenizer, tokenizer_identity)


class PredictionCache:
    """Atomic per-batch files: completed batches survive a disconnected Colab."""
    def __init__(self, root, spec, example_ids):
        self.spec = spec
        self.key = digest_json(spec)
        self.path = Path(root) / self.key[:24]
        self.example_ids = set(example_ids)
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "spec.json"
        if manifest.exists():
            assert read_json(manifest) == spec, "Prediction cache identity mismatch."
        else:
            write_json(manifest, spec)

    def load(self):
        rows = {}
        for path in sorted(self.path.glob("part_*.json")):
            for row in read_json(path):
                identity = row["example_id"]
                assert identity in self.example_ids, "Unexpected example in prediction cache."
                assert identity not in rows, "Duplicate prediction-cache records."
                assert row["run_id"] == self.spec["run_id"], "Wrong adapter in cache."
                rows[identity] = row
        return rows

    def save(self, rows):
        assert len({r["example_id"] for r in rows}) == len(rows)
        assert all(r["example_id"] in self.example_ids for r in rows)
        write_json(self.path / f"part_{uuid.uuid4().hex}.json", rows)


def generate_comparison(repo, runs, sample, comparison_cfg, model_cfg, test_sha):
    """One GPU base at a time; no downloads; skip all cached generations."""
    import torch
    from peft import PeftModel
    repo = Path(repo)
    records = sample.to_dict("records")
    by_model = defaultdict(list)
    for run in runs:
        by_model[run["base_model"]].append(run)
    candidates = {c["id"]: c for c in model_cfg["candidates"]}
    all_rows, receipts = [], []
    helper_hashes = {name: sha256(repo / "jobai" / name) for name in
                     ("forecasting.py", "model_runtime.py", "evaluation.py")}
    for model_id, group in by_model.items():
        base_path = base_directory(repo, model_id, download=False)
        base_snapshot = base_path / "jobai_base_snapshot.json"
        base_revision = read_json(base_snapshot).get("revision") if base_snapshot.exists() else None
        candidate = candidates.get(model_id)
        assert candidate, f"Add historical base model {model_id} to model.yaml candidates."
        model = None
        try:
            for run in group:
                tokenizer, _ = load_tokenizer(base_path, model_id, run["adapter_dir"])
                tokenizer.padding_side = "left"
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                spec = {"protocol": "shared_comparison_v1", "run_id": run["run_id"],
                        "base_model": model_id, "base_revision": base_revision,
                        "base_config_sha256": sha256(base_path / "config.json"),
                        "adapter_sha256": run["adapter_sha256"], "tokenizer_sha256": tokenizer_identity(tokenizer),
                        "prompt_schema": run["prompt_schema"], "history_quarters": run["history_quarters"],
                        "max_seq_length": run["max_seq_length"], "test_sha256": test_sha,
                        "sample_ids_sha256": digest_json(sample.example_id.tolist()),
                        "generation": {"max_new_tokens": comparison_cfg["max_new_tokens"],
                                       "do_sample": False, "enable_thinking": False},
                        "helper_hashes": helper_hashes,
                        "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft")}}
                cache = PredictionCache(repo / "data/cache/model_evaluations" / run["run_id"], spec, sample.example_id)
                saved = cache.load()
                missing = [row for row in records if row["example_id"] not in saved]
                print(f"{run['model_label']}: reusing {len(saved)}/{len(records)} predictions")
                started = time.monotonic()
                if missing:
                    assert torch.cuda.is_available(), "Uncached forecasts require a Colab GPU. Cached reports do not."
                    if model is None:
                        check_fast_kernels(model_id, candidate.get("require_fast_kernels", False))
                        base = load_quantized_base(base_path, candidate["architecture"], torch.cuda.is_bf16_supported())
                        model = PeftModel.from_pretrained(base, run["adapter_dir"], adapter_name=run["model_label"], local_files_only=True)
                        del base
                    elif run["model_label"] not in model.peft_config:
                        model.load_adapter(run["adapter_dir"], adapter_name=run["model_label"], local_files_only=True)
                    model.set_adapter(run["model_label"])
                    model.eval()
                    batch_size = int(comparison_cfg["generation_batch_size"])
                    for start in range(0, len(missing), batch_size):
                        batch = missing[start:start + batch_size]
                        texts = [tokenizer.apply_chat_template(prompt_messages(row, run["prompt_schema"], run["history_quarters"]),
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
                        inputs = tokenizer(texts, padding=True, add_special_tokens=False, truncation=False,
                                           return_tensors="pt").to(model.device)
                        assert inputs["input_ids"].shape[1] <= run["max_seq_length"], f"Prompt too long for {run['run_id']}; no truncation allowed."
                        with torch.inference_mode():
                            outputs = model.generate(**inputs, max_new_tokens=comparison_cfg["max_new_tokens"],
                                                     do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
                        responses = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                        batch_rows = []
                        for row, response in zip(batch, responses):
                            change = parse_prediction(response)
                            _, scale, _, _ = input_view(row, run["history_quarters"])
                            pred = max(0.0, float(row["last_value"]) + change * scale) if change is not None else None
                            batch_rows.append({**{k: row[k] for k in ["example_id", *KEYS, "series_family", "forecast_scope", "measure_code"]},
                                "model": run["model_label"], "run_id": run["run_id"], "train_examples": run["train_count"],
                                "comparison_role": run["role"], "prompt_schema": run["prompt_schema"],
                                "y_true": float(row["target_value"]), "y_pred": pred, "last_value": float(row["last_value"]),
                                "prediction_scale": scale, "parsed": change is not None, "response": response})
                        cache.save(batch_rows)
                        saved.update({row["example_id"]: row for row in batch_rows})
                        print(f"{run['model_label']}: {len(saved)}/{len(records)}", end="\r")
                        del inputs, outputs
                    print()
                assert set(saved) == set(sample.example_id), "Incomplete predictions."
                all_rows.extend(saved[row["example_id"]] for row in records)
                receipts.append({"run_id": run["run_id"], "cache_key": cache.key,
                                 "cache_path": str(cache.path.relative_to(repo)),
                                 "new_predictions": len(missing), "generation_seconds_this_session": time.monotonic() - started})
                del tokenizer
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(all_rows), receipts
