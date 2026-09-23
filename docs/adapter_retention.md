# Adapter retention — selected shared Qwen3-4B

Updated 2026-09-23. This change deletes no artifacts. The selected final adapter is pinned in `configs/final_model.yaml`, not the latest training pointer. Preserve remaining history; this guide is not permission to erase files. Paths below are relative to these roots:

- Local project: `/Users/Alan/Documents/Study/MLProject-JobAI`
- Colab project: `/content/drive/MyDrive/JobAI`

## Keep both adapters for the default comparison

| Reference | Adapter directory relative to either root | Reason |
|---|---|---|
| **Legacy Qwen3-4B, 151,727** | `models/adapters/qwen-qwen3-4b__20260919T125120Z/final_adapter` | **Selected final H1/H2/H4 model; required by default 07** |
| Enhanced Qwen3-4B, 75k | `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/final_adapter` | Required comparator in default 07 |

These two complete final-adapter directories are present locally. Keep the weights, adapter config and tokenizer together. An adapter still needs its matching base model. Smaller/horizon-specific adapter directories previously listed here are no longer required or assumed present.

Keep Qwen3.5 experiment artifacts wherever archived, but they are not needed by the default final-model evaluation. `data/manifests/finetune_run_manifest.json` tracks the latest training run only and cannot change the selected final model.

## Keep provenance and reports

Preserve the corresponding evidence directories wherever retained:

- `reports/model_evaluations/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/`
- `reports/model_evaluations/qwen-qwen3-4b__20260919T125120Z/`
- `reports/model_evaluations/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T071954Z/`

Preserve `reports/model_evaluations/comparisons/`, `reports/finetune_runs/`, and `data/manifests/finetune_runs/` wherever present, plus the current panel/baseline manifests. Older original training manifests are missing locally; the evaluation manifests above are the current fallback evidence. Restore original training manifests from Drive if available. Do not delete fallback evidence before that restoration.

The explicit historical prompt/window declarations in `configs/comparison_final.yaml` preserve the preceding comparison protocol (legacy: eight quarters; enhanced: twelve), but are **not independently verified training settings**. Notebook 07 recovers the extra past history from verified normalized sources, selects common eligible cases, and labels the metadata gap. Treat comparisons as provisional where that metadata is missing. The broader historical roster in `configs/comparison.yaml` remains available separately.

The old comparison reports remain valuable audit records but must not be directly joined to the new metrics: cohorts, features and source checksums can differ. Notebook 07 generates fresh matched-case results in a new comparison directory and updates `reports/model_evaluations/latest_comparison_manifest.json`.

## Keep base caches on Drive

- `/content/drive/MyDrive/JobAI/models/base/Qwen--Qwen3-4B`
- `/content/drive/MyDrive/JobAI/models/base/Qwen--Qwen3.5-4B`

The first is required for the final model; the second is historical/optional. Notebook 07 does not silently download a missing base. Preserve `data/cache/finetune/` and `data/cache/model_evaluations/`; cached results are reusable only when their data, prompt, code and model fingerprints match. Profile changes may require new caches without deleting old ones.

## Optional archives — not required by the new default comparison

- `models/adapters/qwen-qwen3-4b__20260919T083224Z/` — legacy 6k shared run.
- `models/adapters/qwen-qwen3-4b__qwen3_4b_h1__enhanced_v1__h1__20260920T184401Z/` — 60k H1-only experiment, if archived elsewhere. The default workflow does not require separate H1/H2/H4 adapters.

Their reports are small and worth retaining even if weights are archived. No existing model files are changed by this update.

## Checkpoints that can be archived if you will never resume those runs

Keep the complete final adapters above. These exact intermediate directories are not used by notebook 07, but contain optimizer/resume state:

- `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/checkpoint-6000`
- `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/checkpoint-6250`
- `models/adapters/qwen-qwen3-4b__20260919T125120Z/checkpoint-8000`
- `models/adapters/qwen-qwen3-4b__20260919T125120Z/checkpoint-8430`
- `models/adapters/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T071954Z/checkpoint-500`

Before any cleanup, verify the retained final adapter loads with its correct base model and the comparison can be reproduced. No deletion command is included intentionally.
