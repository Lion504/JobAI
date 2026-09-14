# JobAI

JobAI is a team project for forecasting Finnish labour-market indicators and producing source-grounded explanations. The first prototype targets quarterly job vacancies with 1-, 2-, and 4-quarter forecasts, using the Statistics Finland PxWeb API as the numeric source and official bulletins for the narrative layer.

## Principles

- Statistics Finland PxWeb data is authoritative for numeric inputs and labels.
- The fine-tuned model must beat the required baselines (seasonal naive, last-value naive, and Ridge regression) — training success is not enough.
- RAG explains results from official sources; it does not replace structured numeric data.
- Every external snapshot and model run produces a provenance manifest. PR review (not CI) is the quality gate.

## Repository layout

```text
jobai/
├── README.md
├── requirements.txt              # pinned Colab environment
├── notebooks/                    # ordered executable Python work
├── configs/                      # data.yaml, eval.yaml, model.yaml, rag.yaml
├── data/raw/                     # ignored; raw PxWeb responses
├── data/processed/               # ignored; normalized data, splits, JSONL
├── data/manifests/               # provenance and run manifests
├── reports/                      # ignored; generated evaluation outputs
└── models/                       # ignored; local adapters and checkpoints
```

## Workflow

1. Open a fresh Google Colab runtime, mount Drive, and `cd` to the project root. Set `JOBAI_REPO` to the root.
2. `!pip install -r requirements.txt`. Colab provides the CUDA build of `torch`; the pinned CPU version is for local smoke tests only.
3. Run `00_environment_and_smoke_test.ipynb` end-to-end. If it fails, stop.
4. Continue with `01_…` through `09_…` in order. Each notebook's first cell declares purpose, and expected inputs/outputs.

## Data and models

The pipeline draws from two tiers of sources: structured PxWeb tables for numeric inputs, and official bulletins for narrative explanation. All PxWeb queries are declared in `configs/data.yaml`; notebook `01_pxweb_metadata_and_download.ipynb` fetches them with slice-wise POST requests (each < 500k cells) and caches raw responses plus provenance manifests under `data/manifests/`.

### Data catalog

| Table | PxWeb path | Time range | Granularity | Raw cells | Post-agg rows |
|-------|-----------|------------|-------------|-----------|---------------|
| `11l1` National vacancies | `atp/11l1.px` | 2013Q1–2026Q2 | Quarterly | 270 (5 measures × 54 quarters) | 270 |
| `11n1` Regional vacancies | `atp/11n1.px` | 2013Q1–2026Q2 | Quarterly | 270 (5 regions × 1 current measure × 54 quarters) | 270 |
| `12tu` Jobs by occupation | `tyonv/12tu.px` | 2013M01–2026M07 | Monthly → Quarterly | 391,200 across 7 slices | 129,600 quarterly-grid rows |
| `12tw` Vacancies by industry | `tyonv/12tw.px` | 2013M01–2026M07 | Monthly → Quarterly | 1,232,280 across 14 slices | 408,240 quarterly-grid rows |
| `12r5` Jobs by region | `tyonv/12r5.px` | 2009M01–2026M07 | Monthly → Quarterly | 26,586 (1 slice, "all" time) | 8,820 quarterly-grid rows |

KEHA tables retain the M03/M06/M09/M12 quarter-end observations using the stock rule configured in `configs/data.yaml`. Missing source values remain explicit nulls so coverage can be audited. Notebook 02 hard-fails on wrong source months, duplicate keys, quarter-grid gaps, or a normalized quarter later than the latest complete source quarter. ATP tables are already quarterly and pass through unchanged.

Notebook 03 selects the official ATP benchmark targets and quality-filtered KEHA panel targets. Notebook 04 evaluates required baselines on the ATP benchmark. Notebook 05 only creates panel train/validation/test examples; LLM fine-tuning starts in notebook 06 and is gated on the baseline and panel reports.

### Current forecasting readiness

- Dataset A contains five benchmark targets: one `11l1` national series and four non-total `11n1` regional series.
- The current panel quality rules select 44 `12tu` vacancy series and no `12tw` series. Most fetched `12tw` top-level industry × selected employer-sector × selected duration combinations are structurally zero and are not duplicated into the training set.
- Notebook 05 currently produces 4,896 boundary-safe panel examples: 3,898 train, 465 validation, and 533 test.
- Fine-tuning is intentionally blocked by `data/manifests/panel_dataset_card.json` until both configured panel-size gates pass. Expanding or revising the `12tw` query is preferable to weakening the quality thresholds.

- First implementation uses `11l1` as the national quarterly vacancy target; all variable codes come from the table's runtime metadata. Provenance (raw response, query JSON, table metadata, retrieval date, response hash) lives in `data/manifests/`.
- Regional (`11n1`) and KEHA context tables are available from the start; additional ATP/KEHA tables are added as needed for the forecasting dataset.
- Fine-tuning uses either a Qwen3.5 bf16 LoRA branch (Unsloth) or a Qwen3 / Qwen2.5 4-bit QLoRA branch (transformers + peft + trl + bitsandbytes). The selected branch, base model id, commit/version, hardware, and package versions are recorded by `06_finetune_model.ipynb`.
- RAG sources are whitelisted in `configs/rag.yaml` §10.1: StatFin release pages, Job Market Finland and KEHA bulletins, and official TEM pages. Legacy `mol.fi` references are treated as unverified.

## Contribution

1. Create a focused branch.
2. Add or update notebook cells; keep outputs cleared (or only the relevant ones) before committing.
3. Edit `configs/*.yaml` only for genuine defaults — per-experiment overrides go in a copy.
4. Open a PR. Reviewers re-run the affected notebook on Colab and inspect the generated manifests and reports. Merge only after review and after any updated manifests or summary reports are committed.
