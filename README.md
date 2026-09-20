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

Notebook 03 now creates one direct-target catalog across all five tables. Notebook 04 evaluates the required baselines on exactly those targets using common complete rolling-origin windows. Notebook 05 creates combined and horizon-specific train/validation/test examples; Notebook 06 is then run once for each 1Q, 2Q, and 4Q adapter profile.

### Current forecasting readiness

- Dataset A remains a compact reference set with five ATP targets: one `11l1` national series and four non-total `11n1` major-region series.
- Direct forecast targets now cover national (`11l1`), broad-region (`11n1`), province-by-occupation (`12tu`), province-by-industry (`12tw`), and detailed geography (`12r5`) questions. Exact `11n1`/national and `12r5`/province duplicates are explicitly excluded.
- Notebook 05 writes separate `h1`, `h2`, and `h4` files, carries source/scope metadata, and adds compact origin-safe national or peer context. Because the target scope and history window changed, the old 1,679-series/151,727-example reports are historical and notebooks 03–05 must be rerun before new counts are quoted.
- Three Qwen3-4B pilot profiles live in `configs/model_qwen3_4b_h1.yaml`, `configs/model_qwen3_4b_h2.yaml`, and `configs/model_qwen3_4b_h4.yaml`. Each starts with 10,000 balanced examples and retains every available ATP national and broad-region example.

- First implementation uses `11l1` as the national quarterly vacancy target; all variable codes come from the table's runtime metadata. Provenance (raw response, query JSON, table metadata, retrieval date, response hash) lives in `data/manifests/`.
- Regional (`11n1`) and KEHA context tables are available from the start; additional ATP/KEHA tables are added as needed for the forecasting dataset.
- The active experiment uses one cached Qwen3-4B base model and three separate 4-bit QLoRA adapters, one per forecast horizon. Every immutable run records the profile, data checksums, sampling composition, package versions, hardware, training history, adapter, and validation results.
- RAG sources are whitelisted in `configs/rag.yaml` §10.1: StatFin release pages, Job Market Finland and KEHA bulletins, and official TEM pages. Legacy `mol.fi` references are treated as unverified.

## Contribution

1. Create a focused branch.
2. Add or update notebook cells; keep outputs cleared (or only the relevant ones) before committing.
3. Edit `configs/*.yaml` only for genuine defaults — per-experiment overrides go in a copy.
4. Open a PR. Reviewers re-run the affected notebook on Colab and inspect the generated manifests and reports. Merge only after review and after any updated manifests or summary reports are committed.
