# MLProject-JobAI — Agent Context

## Project Overview
Finnish Job-Market Forecasting Pipeline: forecasts quarterly job vacancies (1Q/2Q/4Q horizons) using Statistics Finland PxWeb data (ATP + KEHA tables).

## Notebook Pipeline (per plan.md)
| Step | Notebook | Purpose |
|------|----------|---------|
| 0 | `00_environment_and_smoke_test.ipynb` | Env setup, GPU check |
| 1 | `01_pxweb_metadata_and_download.ipynb` | Fetch PxWeb metadata & raw data |
| 2 | `02_normalize_and_validate_data.ipynb` | Parse, normalize, validate, quality reports |
| 3 | `03_eda_and_series_selection.ipynb` | **TODO**: EDA + select forecast series |
| 4 | `04_baselines_and_rolling_evaluation.ipynb` | **TODO**: Seasonal naive, Ridge, rolling-origin eval |
| 5 | `05_prepare_forecasting_dataset.ipynb` | **TODO**: Build regression features for 1Q/2Q/4Q targets |
| 6 | `06_finetune_model.ipynb` | LLM fine-tuning (Qwen3.5 LoRA or QLoRA) |
| 7 | `07_evaluate_model.ipynb` | Compare fine-tuned vs baselines |
| 8 | `08_rag_index_and_evaluation.ipynb` | RAG with Statistics Finland sources |
| 9 | `09_final_demo.ipynb` | End-to-end demo |

## Current State
- `01_` and `02_` implemented
- `03_` exists but is **misplaced** (currently `03_classification.ipynb` does classification, not EDA)
- Need to create proper `03_eda_and_series_selection.ipynb`
- Need to create `04_baselines_and_rolling_evaluation.ipynb`

## Data Sources
- `StatFin/atp/11l1.px` — national quarterly vacancies (270 rows)
- `StatFin/atp/11n1.px` — regional quarterly vacancies
- `StatFin/tyonv/12tu.px`, `12tw.px`, `12r5.px` — KEHA monthly jobseeker/vacancy data
- `StatFin/atp/15ia.px` — industry vacancies (TOL2025)
- `StatFin/tyti/137l.px` — employment by industry (LFS)

## Key Configs
- `configs/data.yaml` — table definitions, queries, expected dtypes
- `configs/eval.yaml` — evaluation cutoffs, horizons
- `kilo.json` — Kilo CLI config (minimal)

## Conventions
- Chronological splits only (no random `train_test_split`)
- Rolling-origin evaluation with explicit info cutoff
- Metrics: MAE, RMSE, MASE, sMAPE by horizon/series
- Baselines: seasonal naive, last-value, Ridge regression
- All notebooks runnable top-to-bottom in fresh Colab
- Deterministic seeds (42)
- Outputs to `data/processed/`, `reports/`, `models/`

## Agent Instructions
- Prefer `bash` + `jupyter nbconvert` for notebook execution
- Use `read`/`glob`/`grep` for notebook inspection
- Don't create new notebooks unless explicitly asked
- Follow plan.md sequence strictly
- Validate with `jupyter nbconvert --execute` before committing