"""Data contracts, deterministic sampling and prompts. No model/GPU imports."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROMPT_VERSION = "five_table_shared_v1"
KEYS = ["table_id", "series_id", "horizon_q", "origin_quarter", "target_quarter"]
METRICS = ["MAE", "RMSE", "MASE", "sMAPE_pct"]
ALIASES = [("quarter_of_year", "quarter"), ("qoq_change_scaled", "qoq"),
           ("yoy_change_scaled", "yoy"), ("recent_mean_4_scaled", "mean4"),
           ("window_mean_scaled", "mean_window"), ("recent_slope_4_scaled", "slope4"),
           ("window_slope_scaled", "slope_window"), ("recent_std_4_scaled", "std4"),
           ("window_std_scaled", "std_window"), ("zero_fraction_window", "zero_fraction")]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    if immutable:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)


def verify_pipeline(repo, cfg, splits=("train", "validation")):
    """Fail before loading a model if 03/04/05 disagree. Never reads test in 06."""
    repo = Path(repo)
    card = read_json(repo / "data/manifests/panel_dataset_card.json")
    baseline = read_json(repo / "reports/baseline_run_manifest.json")
    assert card.get("gate", {}).get("finetuning_ready"), "Dataset gate failed; run 03–05."
    expected = {"feature_window_quarters": cfg["feature_window_quarters"],
                "horizons_q": cfg["horizons"], "splits": cfg["splits"]}
    for key, value in expected.items():
        assert card.get(key) == value, f"Notebook 05 {key} is stale; rebuild 04/05 with configs/eval.yaml."
        assert baseline.get(key) == value, f"Notebook 04 {key} is stale; rebuild 04 then 05."
    assert card.get("feature_set") == cfg["engineered_feature_set"], "Panel feature set is stale."
    assert baseline.get("engineered_feature_set") == cfg["engineered_feature_set"], "Baseline features are stale."
    assert set(card.get("selected_series_by_table", {})) == set(cfg["panel_selection"]["target_tables"]), "Panel target tables are stale; run 03–05."
    selected = sha256(repo / "data/processed/selected_series.csv")
    assert card.get("selection_catalog_sha256") == selected == baseline.get("selection_catalog_sha256"), "03/04/05 selection checksums differ."
    assert card.get("baseline_manifest_sha256") == sha256(repo / "reports/baseline_run_manifest.json"), "05 refers to a different baseline run."
    assert {"last_value", "seasonal_naive", "ridge", "ridge_enhanced"}.issubset(baseline["models"])
    for relative, evidence in baseline["outputs"].items():
        assert sha256(repo / relative) == evidence["sha256"], f"Baseline output changed: {relative}"
    for split in splits:
        evidence = card["files"][split]
        assert sha256(repo / evidence["path"]) == evidence["sha256"], f"Panel {split} checksum mismatch."
    # New exports record source identities; older otherwise-matching exports
    # are supported, but their absence is surfaced to the caller.
    sources = baseline.get("normalized_inputs", {})
    for relative, expected_sha in sources.items():
        assert sha256(repo / relative) == expected_sha, f"Normalized source changed: {relative}; rerun 03–05."
    return card, baseline


def _rank(frame, seed):
    result = frame.copy()
    result["_rank"] = result.example_id.map(lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())
    return result


def balanced_sample(frame, limit, seed, horizons=(1, 2, 4), priority_tables=("11l1", "11n1")):
    """Equal horizon quotas, then deterministic round-robin strata; no replacement."""
    frame = frame[frame.horizon_q.isin(horizons)].copy()
    assert frame.example_id.is_unique, "Duplicate training/validation example IDs."
    assert set(frame.horizon_q) == set(horizons), "A requested horizon has no examples."
    if limit is None:
        return frame.sort_values("example_id").reset_index(drop=True)
    assert isinstance(limit, int) and limit >= len(horizons)
    parts = []
    for index, horizon in enumerate(sorted(horizons)):
        quota = limit // len(horizons) + (index < limit % len(horizons))
        group = _rank(frame[frame.horizon_q == horizon], seed + horizon)
        assert len(group) >= quota, f"H{horizon} has {len(group)} records, needs {quota}; lower the TOTAL budget."
        priority = group[group.table_id.isin(priority_tables)].sort_values("_rank")
        assert len(priority) <= quota, f"H{horizon}: ATP retention exceeds budget; increase budget explicitly."
        rest = group[~group.example_id.isin(priority.example_id)].copy()
        strata = ["table_id", "forecast_scope", "target_size_band", "volatility_band"]
        rest["_stratum"] = rest[strata].astype(str).agg("|".join, axis=1)
        rest["_round"] = rest.groupby("_stratum")["_rank"].rank(method="first")
        rest["_stratum_rank"] = rest._stratum.map(lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())
        rest = rest.sort_values(["_round", "_stratum_rank", "_rank"]).head(quota - len(priority))
        parts.append(pd.concat([priority, rest], ignore_index=True))
    result = pd.concat(parts, ignore_index=True)
    result = result.drop(columns=[c for c in result if c.startswith("_")])
    assert len(result) == limit and result.example_id.is_unique
    return result.sort_values("example_id").reset_index(drop=True)


def matched_test_sample(frame, count, seed):
    parts = []
    for horizon, group in frame.groupby("horizon_q", sort=True):
        assert len(group) >= count, f"H{horizon}: fewer than {count} eligible comparison cases."
        group = _rank(group, seed + int(horizon))
        group["_stratum"] = group[["table_id", "forecast_scope"]].astype(str).agg("|".join, axis=1)
        group["_round"] = group.groupby("_stratum")["_rank"].rank(method="first")
        group["_stratum_rank"] = group._stratum.map(lambda x: hashlib.sha256(f"{seed + int(horizon)}|{x}".encode()).hexdigest())
        parts.append(group.sort_values(["_round", "_stratum_rank", "_rank"]).head(count))
    result = pd.concat(parts, ignore_index=True)
    return result.drop(columns=[c for c in result if c.startswith("_")]).sort_values(
        ["horizon_q", "table_id", "series_id", "origin_quarter"]).reset_index(drop=True)


def input_view(row, history_quarters):
    values = np.asarray(json.loads(row["input_values_json"])[-history_quarters:], dtype=float)
    assert len(values) == history_quarters and np.isfinite(values).all()
    scale = max(abs(float(values[-1])), float(np.std(values)), 1.0)
    slope = lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0])
    recent = values[-4:]
    features = dict(quarter_of_year=int(row["origin_quarter"][-1]),
                    qoq_change_scaled=float((values[-1] - values[-2]) / scale),
                    yoy_change_scaled=float((values[-1] - values[-5]) / scale),
                    recent_mean_4_scaled=float((recent.mean() - values[-1]) / scale),
                    window_mean_scaled=float((values.mean() - values[-1]) / scale),
                    recent_slope_4_scaled=slope(recent) / scale, window_slope_scaled=slope(values) / scale,
                    recent_std_4_scaled=float(recent.std() / scale), window_std_scaled=float(values.std() / scale),
                    zero_fraction_window=float(np.mean(values == 0)))
    origin = int(row["origin_quarter"][:4]) * 4 + int(row["origin_quarter"][-1]) - 1
    start = origin - history_quarters + 1
    return values.tolist(), scale, features, f"{start // 4}Q{start % 4 + 1}"


def prompt_messages(row, schema=PROMPT_VERSION, history_quarters=12):
    history, scale, features, start = input_view(row, history_quarters)
    number = lambda x: format(float(x), ".6g")
    features_text = "; ".join(f"{alias}={number(features[key])}" for key, alias in ALIASES)
    dims = json.dumps(json.loads(row["dimensions_json"]), ensure_ascii=False, sort_keys=True)
    if schema == PROMPT_VERSION:
        system = "You forecast Finnish job-vacancy time series from Statistics Finland and KEHA. "
        lines = [f"Source family: {row['series_family']}", f"Table: {row['table_id']}",
                 f"Forecast scope: {row['forecast_scope']}", f"Measure: {row['measure_code']}",
                 f"Dimensions: {dims}", f"{len(history)} quarterly values, oldest to newest: {[number(v) for v in history]}",
                 f"Origin-safe features: {features_text}"]
        context = json.loads(row.get("hierarchical_context_json", "{}"))
        if context:
            lines.append(f"Reference context: {json.dumps(context, ensure_ascii=False, sort_keys=True)}")
        lines += [f"Forecast origin: {row['origin_quarter']}", f"Forecast horizon: {int(row['horizon_q'])} quarter(s)"]
    else:
        assert schema in ("legacy_v1", "enhanced_shared_v1"), f"Unknown prompt schema: {schema}"
        system = "You forecast Finnish registered job vacancies. "
        description = "Eight quarterly vacancy values" if schema == "legacy_v1" and len(history) == 8 else f"{len(history)} quarterly vacancy values"
        lines = [f"Series family: {row['series_family']}", f"Table: {row['table_id']}",
                 f"Dimensions: {dims}", f"History start: {start}",
                 f"{description}, oldest to newest: {[number(v) for v in history]}"]
        if schema == "enhanced_shared_v1":
            lines.append(f"Origin-safe features: {features_text}")
        lines += [f"Forecast origin: {row['origin_quarter']}", f"Forecast horizon: {int(row['horizon_q'])} quarter(s)",
                  f"Target quarter: {row['target_quarter']}"]
    lines += [f"Scale: {number(scale)}", "Predict target_scaled_change = (target vacancy value - latest history value) / scale."]
    return [{"role": "system", "content": system + "Return exactly one JSON object with one numeric field named target_scaled_change."},
            {"role": "user", "content": "\n".join(lines)}]


def encode_record(tokenizer, messages, completion, max_length):
    """Explicit completion-only labels; same non-thinking prefix as inference."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
    target = tokenizer(completion, add_special_tokens=False)["input_ids"]
    assert tokenizer.eos_token_id is not None, "Tokenizer has no EOS token."
    target = target + [tokenizer.eos_token_id]
    ids = prompt + target
    assert len(ids) <= max_length, f"Record needs {len(ids)} tokens; limit {max_length}. Increase limit or shorten prompt; no records were truncated."
    assert len(target) > 1, "Empty training completion."
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt) + target}


def parse_prediction(text):
    text = text.split("</think>")[-1].strip()
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        value = json.loads(match.group())["target_scaled_change"]
        if isinstance(value, bool):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, KeyError, TypeError):
        return None


def add_errors(frame):
    result = frame.copy()
    result["abs_error"] = (result.y_true - result.y_pred).abs()
    result["squared_error"] = (result.y_true - result.y_pred) ** 2
    result["scaled_abs_error"] = result.abs_error / result.mase_scale
    denom = result.y_true.abs() + result.y_pred.abs()
    result["smape_component_pct"] = 200 * result.abs_error / denom.replace(0, np.nan)
    result.loc[denom == 0, "smape_component_pct"] = 0.0
    return result


def metric_tables(frame):
    """Macro-average per-series errors, matching notebook 04's definition."""
    keys = ["model", "table_id", "forecast_scope", "series_id", "horizon_q"]
    by_series = frame.groupby(keys, dropna=False).agg(
        n_forecasts=("y_true", "size"), MAE=("abs_error", "mean"),
        RMSE=("squared_error", lambda x: float(np.sqrt(x.mean()))),
        MASE=("scaled_abs_error", "mean"), sMAPE_pct=("smape_component_pct", "mean")).reset_index()
    def aggregate(keys):
        return by_series.groupby(keys, dropna=False).agg(
            **{m: (m, "mean") for m in METRICS}, n_series=("series_id", "nunique"),
            n_forecasts=("n_forecasts", "sum")).reset_index()
    return by_series, aggregate(["model", "horizon_q"]), aggregate(["model", "table_id", "forecast_scope", "horizon_q"])
