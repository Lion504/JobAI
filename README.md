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
├── data/raw/                     # raw PxWeb responses
├── data/processed/               # normalized data, splits, JSONL
├── data/manifests/               # provenance and run manifests
├── reports/                      # generated evaluation outputs
└── models/                       # ignored; local adapters and checkpoints
```

## Workflow

1. Open a fresh Google Colab runtime, mount Drive, and `cd` to the project root. Set `JOBAI_REPO` to the root.
2. `!pip install -r requirements.txt`. Colab provides the CUDA build of `torch`; the pinned CPU version is for local smoke tests only.
3. Run `00_environment_and_smoke_test.ipynb` end-to-end. If it fails, stop.
4. Continue with `01_…` through `09_…` in order. Each notebook's first cell declares purpose, and expected inputs/outputs.

## Data and models

The pipeline draws from two tiers of sources: structured PxWeb tables for numeric inputs, and official bulletins for narrative explanation. All PxWeb queries are declared in `configs/data.yaml`; notebook `01_pxweb_metadata_and_download.ipynb` fetches them with API-safe slice-wise POST requests and caches raw responses plus provenance manifests under `data/manifests/`.

### Data catalog

| Table                                       | PxWeb path      | Time range      | Granularity                      | Raw cells                                                                      | Post-agg rows               |
| ------------------------------------------- | --------------- | --------------- | -------------------------------- | ------------------------------------------------------------------------------ | --------------------------- |
| `11l1` National vacancies                   | `atp/11l1.px`   | 2013Q1–2026Q2   | Quarterly                        | 270 (5 measures × 54 quarters)                                                 | 270                         |
| `11n1` Regional vacancies                   | `atp/11n1.px`   | 2013Q1–2026Q2   | Quarterly                        | 270 (5 regions × 1 current measure × 54 quarters)                              | 270                         |
| `12tu` Vacancies by occupation and province | `tyonv/12tu.px` | 2013Q1–2026Q2   | Quarter-end monthly observations | 445,284 across 5 slices (19 provinces × all 434 occupations × 54 quarter ends) | 445,284 quarterly-grid rows |
| `12tw` Vacancies by industry and province   | `tyonv/12tw.px` | 2009M01–2026M07 | Monthly → Quarterly              | 453,017 across 18 slices (19 provinces × all 113 industries × 211 months)      | 150,290 quarterly-grid rows |
| `12r5` Vacancy measures by geography        | `tyonv/12r5.px` | 2009Q1–2026Q2   | Quarter-end monthly observations | 58,940 (all 421 official geographies × 2 vacancy measures × 70 quarter ends)   | 58,940 quarterly-grid rows  |

**Overall configured data volume:** 957,781 downloaded value cells across all five tables, producing 655,054 normalized quarterly rows before quality-based series selection and forecasting-window construction.

KEHA tables retain the M03/M06/M09/M12 quarter-end observations using the stock rule configured in `configs/data.yaml`. Missing source values remain explicit nulls so coverage can be audited. Notebook 02 hard-fails on wrong source months, duplicate keys, quarter-grid gaps, or a normalized quarter later than the latest complete source quarter. ATP tables are already quarterly and pass through unchanged.

Notebook 03 creates one direct-target catalog across all five tables. Notebook 04 evaluates the required baselines on those targets using common complete rolling-origin windows. Notebook 05 creates combined train/validation/test examples. Notebook 06 runs **once**, training one shared Qwen3.5-4B adapter for 1Q, 2Q and 4Q.

### Current forecasting readiness

- Dataset A remains a compact reference set with five ATP targets: one `11l1` national series and four non-total `11n1` major-region series.
- Direct forecast targets now cover national (`11l1`), broad-region (`11n1`), province-by-occupation (`12tu`), province-by-industry (`12tw`), and detailed geography (`12r5`) questions. Exact `11n1`/national and `12r5`/province duplicates are explicitly excluded.
- Notebook 05 uses twelve-quarter histories with source/scope metadata and compact origin-safe national or peer context. Combined files are the default; optional `export_horizon_files` in `configs/eval.yaml` preserves old experiments without requiring separate training runs. The old 1,679-series/151,727-example reports remain historical; regenerate 03–05 if the preflight reports stale inputs.
- `configs/model.yaml` selects Qwen3.5-4B with **75,000 total training examples**, 25,000 per horizon, one epoch and 1,000 validation examples. Sampling balances table/scope/size/volatility and retains ATP targets. The old Qwen3-4B horizon profiles are archived experiments, not active defaults.

- First implementation uses `11l1` as the national quarterly vacancy target; all variable codes come from the table's runtime metadata. Provenance (raw response, query JSON, table metadata, retrieval date, response hash) lives in `data/manifests/`.
- Regional (`11n1`) and KEHA context tables are available from the start; additional ATP/KEHA tables are added as needed for the forecasting dataset.
- The active experiment uses one cached Qwen3.5-4B base model and one shared 4-bit QLoRA adapter, with frozen vision and language-only LoRA. Every run gets a unique directory and immutable manifest. Batch-size changes reuse the tokenized dataset cache; changes to prompts, tokenizer, data or sample budget invalidate it.
- RAG sources are whitelisted in `configs/rag.yaml` §10.1: StatFin release pages, Job Market Finland and KEHA bulletins, and official TEM pages. Legacy `mol.fi` references are treated as unverified.

### Run the shared experiment in Colab

1. Sync the entire updated repository, including the new `jobai/` helper package and configs, not just the notebooks. Preserve your saved adapters, manifests, base caches and reports.
2. Run 03 → 04 → 05 if their current manifests do not match the five-table/twelve-quarter design. Notebook 04 now records normalized-source hashes. Notebook 03's selection logic is unchanged.
3. Use a fresh GPU runtime for 06. Run its dependency and fast-kernel setup, restarting the runtime after installs if requested. The notebook refuses the slow Qwen3.5 kernel fallback by default. Start with micro-batch 4, accumulation 4 and checkpointing enabled; inspect the measured speed before a long run. GPU fit and runtime are not guaranteed by the memory preflight.
4. Run 06 once. It masks prompt tokens, never truncates a training record, evaluates once per epoch and saves a new adapter without overwriting older ones. Checkpoints permit explicit recovery via `resume_from_checkpoint` with unchanged settings. After an interrupted run, restore its config and checkpoint path before rerunning.
5. Run 07 with the explicit roster in `configs/comparison.yaml`. It requires only one current shared adapter, compares 1,000 identical cases per horizon, loads base models sequentially and saves resumable prediction batches. Preserve cached Qwen3-4B weights for the historical comparisons; 07 does not silently download them.

The main criterion is sMAPE improvement by horizon against retained adapters and baselines, with MAE/RMSE/MASE alongside it. A larger or newer model is not a guarantee of sMAPE below 10%. These repeatedly inspected test cases are a development benchmark, not fresh final deployment evidence. Some historical prompt/window settings rely on explicit compatibility declarations because original manifests are absent locally; 07 records that limitation. This compares complete saved forecasting systems, not architecture alone: older runs used different data scopes, training budgets and prompts.

See [adapter retention inventory](docs/adapter_retention.md) before any cleanup. Nothing is deleted automatically. The existing 9B/27B notebooks/configs are preserved but not used by this workflow.

## Contribution

1. Create a focused branch.
2. Add or update notebook cells; keep outputs cleared (or only the relevant ones) before committing.
3. Edit `configs/*.yaml` only for genuine defaults — per-experiment overrides go in a copy.
4. Open a PR. Reviewers re-run the affected notebook on Colab and inspect the generated manifests and reports. Merge only after review and after any updated manifests or summary reports are committed.
