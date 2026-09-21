# Adapter retention — shared Qwen3.5-4B experiment

Inventory date: 2026-09-21. **No files have been deleted.** Keep everything until the new shared run and comparison complete successfully. This is a retention plan, not permission to erase files. Paths below are relative to these exact roots:

- Local project: `/Users/Alan/Documents/Study/MLProject-JobAI`
- Colab project: `/content/drive/MyDrive/JobAI`

## Keep for the default comparison

| Reference | Adapter directory relative to either root | Reason |
|---|---|---|
| Enhanced Qwen3-4B, 75k | `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/final_adapter` | Main previous shared-adapter reference |
| Legacy Qwen3-4B, 151,727 | `models/adapters/qwen-qwen3-4b__20260919T125120Z/final_adapter` | Strong previous short-horizon reference |
| Enhanced Qwen3-4B, 6k | `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T071954Z/final_adapter` | Small-data reference; retains the earlier longer-horizon experiment |

All three final adapters exist locally, including weights, adapter config and tokenizers. Do not keep only `adapter_model.safetensors`: keep each complete `final_adapter` directory. An adapter is not the full base model.

Keep the new Qwen3.5-4B run as well. Notebook 06 prints its exact unique path and records it in `data/manifests/finetune_run_manifest.json`; the timestamped name cannot be known before execution.

## Keep provenance and reports

Preserve these corresponding evidence directories (all three exist locally):

- `reports/model_evaluations/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/`
- `reports/model_evaluations/qwen-qwen3-4b__20260919T125120Z/`
- `reports/model_evaluations/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T071954Z/`

Preserve `reports/model_evaluations/comparisons/`, `reports/finetune_runs/`, and `data/manifests/finetune_runs/` wherever present, plus the current panel/baseline manifests. Older original training manifests are missing locally; the evaluation manifests above are the current fallback evidence. Restore original training manifests from Drive if available. Do not delete fallback evidence before that restoration.

The explicit historical prompt/window declarations in `configs/comparison.yaml` preserve the preceding comparison protocol, but are **not independently verified training settings**. Notebook 07 labels the gap instead of silently inventing historical provenance. Treat comparisons as provisional where that metadata is missing.

The old comparison reports remain valuable audit records but must not be directly joined to the new metrics: cohorts, features and source checksums can differ. Notebook 07 generates fresh matched-case results in a new comparison directory and updates `reports/model_evaluations/latest_comparison_manifest.json`.

## Keep base caches on Drive

- `/content/drive/MyDrive/JobAI/models/base/Qwen--Qwen3-4B`
- `/content/drive/MyDrive/JobAI/models/base/Qwen--Qwen3.5-4B`

The first is needed to evaluate the old adapters; the second is created/reused by the new run. Neither base cache currently exists in the local project. Notebook 07 does not silently download a missing base. Preserve `data/cache/finetune/` and `data/cache/model_evaluations/` if you want to avoid re-tokenization and repeated forecast generation.

## Optional archives — not required by the new default comparison

- `models/adapters/qwen-qwen3-4b__20260919T083224Z/` — legacy 6k shared run.
- `models/adapters/qwen-qwen3-4b__qwen3_4b_h1__enhanced_v1__h1__20260920T184401Z/` — 60k H1-only experiment. Now present locally, including its final adapter. Preserve it as an optional archived reference; the new workflow does not require H1/H2/H4 adapters.

Only after backing them up and verifying the new comparison could these be removed locally. Their reports are small and worth retaining even if weights are archived. Existing user-staged model files are untouched.

## Checkpoints that can be archived if you will never resume those runs

Keep the complete final adapters above. These exact intermediate directories are not used by notebook 07, but contain optimizer/resume state:

- `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/checkpoint-6000`
- `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/checkpoint-6250`
- `models/adapters/qwen-qwen3-4b__20260919T125120Z/checkpoint-8000`
- `models/adapters/qwen-qwen3-4b__20260919T125120Z/checkpoint-8430`
- `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T071954Z/checkpoint-500`

Before any cleanup, verify the retained final adapter loads with its correct base model and the comparison can be reproduced. No deletion command is included intentionally.
