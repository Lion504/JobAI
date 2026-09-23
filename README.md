# JobAI

JobAI is a team project that explores whether a fine-tuned language model can improve forecasts of Finnish job vacancies. It uses official Statistics Finland and KEHA data to predict vacancy counts **1, 2, and 4 quarters ahead**—roughly 3, 6, and 12 months.

The forecasting pipeline downloads and checks the data, builds training examples, and compares the model with simple forecasting methods. A planned explanation layer will use retrieval-augmented generation (RAG) to find relevant official publications and explain forecasts with citations.

## Current status

The selected final project model is **Qwen3-4B legacy, trained on 151,727 examples**, using **one shared adapter for H1/H2/H4**. Its exact saved run is pinned in [configs/final_model.yaml](configs/final_model.yaml), independently of any later training run:

`models/adapters/qwen-qwen3-4b__20260919T125120Z/final_adapter`

The saved [dataset card](data/manifests/panel_dataset_card.json) records **1,679 selected series**, an **8-quarter history**, and **151,727 training / 15,929 validation / 17,809 test examples** from `12tu` and `12tw`. The default workflow restores this shared legacy design. All five source tables remain available, but the selected adapter was not trained on ATP or `12r5` targets. Using the saved adapter requires **no retraining**. Selection as the final project model is not a deployment-accuracy guarantee; the existing comparisons remain development evidence.

## Data

The figures below describe the source snapshot downloaded on **14 September 2026**. Queries are defined in [configs/data.yaml](configs/data.yaml), and download records are saved in [data/manifests/](data/manifests/).

| Table     | Coverage                                               | Source period               | Downloaded value cells | Quarterly rows |
| --------- | ------------------------------------------------------ | --------------------------- | ---------------------: | -------------: |
| `11l1`    | National vacancies, 5 measures                         | 2013Q1–2026Q2               |                    270 |            270 |
| `11n1`    | Vacancies across 5 areas, including the national total | 2013Q1–2026Q2               |                    270 |            270 |
| `12tu`    | 19 provinces × 434 occupation categories               | Quarter ends, 2013Q1–2026Q2 |                445,284 |        445,284 |
| `12tw`    | 19 provinces × 113 industry categories                 | Monthly, 2009M01–2026M07    |                453,017 |        150,290 |
| `12r5`    | 421 geographic codes × 2 vacancy measures              | Quarter ends, 2009Q1–2026Q2 |                 58,940 |         58,940 |
| **Total** | **5 tables**                                           |                             |            **957,781** |    **655,054** |

ATP survey data (`11l1`, `11n1`) is already quarterly. For the monthly KEHA tables, the pipeline keeps March, June, September, and December observations; these are not quarterly sums. The normalized data ends at **2026Q2**.

These counts include missing values: **139,438** of the quarterly rows have no numeric value. Notebook 02 preserves them and checks for duplicate keys, missing quarters, and incorrect source months. Notebook 03 then filters out unsuitable series and known duplicates. The row counts above are therefore not the number of training examples.

ATP survey estimates and KEHA registered vacancies measure different things. The pipeline keeps their source and measure labels separate.

### Training, validation, and test splits

For the selected **Qwen3-4B legacy 151,727-example run**, all three splits came from **`12tu` and `12tw`**, separated **by time—not by table**. These are occupation/province and industry/province vacancy series. Notebook 05 creates the splits; Notebook 06 uses training and validation records; Notebook 07 evaluates test records against Notebook 04's baselines.

| Split | Available examples | Used for the selected run/comparison | Purpose |
| --- | ---: | --- | --- |
| Training | 151,727 | All 151,727 | Update the adapter's weights. |
| Validation | 15,929 | A 1,000-example sample during training | Monitor performance without updating weights from these answers. |
| Testing | 17,809 | Notebook 07 evaluates a matched subset; the current default is 1,000 per horizon (3,000 total) | Compare the saved adapter and baselines on the same forecast cases. |

The counts come from the saved [panel dataset card](data/manifests/panel_dataset_card.json); the training/validation sample settings are recorded in the [earlier shared-run configuration](https://github.com/Lion504/JobAI/blob/d8f7cf8/configs/model.yaml). Earlier evaluation reports used smaller test samples, so 17,809 is the **available test pool**, not the number scored in every report.

The chronological boundaries are:

- **Training:** forecast targets must be observed by **2022Q4**.
- **Validation:** forecast origins fall in **2023Q1–2024Q2**, and their targets must also fall by **2024Q2**.
- **Testing:** forecast origins run from **2024Q3** through the latest eligible origin, with observed targets available through **2026Q2**. The last origin is **2026Q1 for H1**, **2025Q4 for H2**, and **2025Q2 for H4**.

A *forecast origin* is the quarter when the prediction is made. Each example contains eight past quarters and one future target at H1, H2, or H4; overlapping history windows create many examples from each series. History may reach into an earlier split, but future answers are never included in a model's input.

**ATP (`11l1`, `11n1`) was not this adapter's validation or test dataset.** Those tables remain separate benchmark/EDA series, and `12r5` remains auxiliary geographic/context data. None supplies training targets or prompt context to the selected legacy adapter. The baseline methods used for its model comparison also forecast **`12tu`/`12tw`**; “baseline” means a forecasting method, not a separate source table. These roles match the earlier [Notebook 04](https://github.com/Lion504/JobAI/blob/d8f7cf8/notebooks/04_baselines_and_rolling_evaluation.ipynb) and [Notebook 05](https://github.com/Lion504/JobAI/blob/d8f7cf8/notebooks/05_prepare_panel_dataset.ipynb).

## Workflow

Run the main notebooks in order. Each saves the inputs needed by the next step.

| Step | Notebook                                                                  | Purpose                                         |
| ---- | ------------------------------------------------------------------------- | ----------------------------------------------- |
| 00   | [Environment check](notebooks/00_environment_and_smoke_test.ipynb)        | Check packages, configuration, and API access   |
| 01   | [Download data](notebooks/01_pxweb_metadata_and_download.ipynb)           | Fetch source tables and record their provenance |
| 02   | [Normalize and validate](notebooks/02_normalize_and_validate_data.ipynb)  | Prepare quarterly data and check its quality    |
| 03   | [Explore and select series](notebooks/03_eda_and_series_selection.ipynb)  | Choose suitable forecast targets                |
| 04   | [Evaluate baselines](notebooks/04_baselines_and_rolling_evaluation.ipynb) | Establish reference results                     |
| 05   | [Prepare the panel](notebooks/05_prepare_panel_dataset.ipynb)             | Build training, validation, and test examples   |
| 06   | [Fine-tune the model](notebooks/06_finetune_model.ipynb)                  | Train one adapter for all three horizons        |
| 07   | [Evaluate the model](notebooks/07_evaluate_model.ipynb)                   | Compare forecasts on matching test cases        |

The classification notebook, separate horizon profiles, and larger-model experiment are optional work outside this main sequence.

### Use the final model or run a new shared experiment in Colab

1. Copy the full repository to Google Drive. Mount it and set the working directory and `JOBAI_REPO` to its root; the notebooks default to `/content/drive/MyDrive/JobAI`.
2. To rebuild matching inputs, run **00 → 05**, or **03 → 05** if normalized data is already prepared. Those notebooks share the active legacy profile. If current baseline/panel files already pass checksum and profile checks, reuse them; do not rerun them merely to select an adapter. Back up generated outputs before switching experiment profiles.
3. To **compare both saved adapters**, skip training. Have the complete legacy 151,727 and enhanced 75,000 adapter directories and the `Qwen/Qwen3-4B` base cache available. In a fresh GPU runtime, use 06's dependency setup if needed (only the dependency-install step), then run **07** with [configs/comparison_final.yaml](configs/comparison_final.yaml). Evaluation never downloads missing weights or follows the latest-training pointer.

Reproduce that experiment only by explicitly setting `JOBAI_MODEL_CONFIG=configs/model.yaml` before 03–06. A historical 07 comparison also needs an explicit comparison config/model profile and every listed adapter; missing archived weights are not silently skipped.

Keep existing adapters, reports, and caches. See the [adapter retention guide](docs/adapter_retention.md) for the required files.

## Model and evaluation

The default model is **Qwen3-4B**, a text-only causal language model trained with 4-bit QLoRA: a small adapter is trained while the base model stays frozen. One adapter handles all three horizons; no separate H1/H2/H4 adapters are required.

Optional new training uses **all eligible mixed-horizon examples**, **one epoch**, and a deterministic **1,000-example validation sample**. Inputs contain eight quarters of history and series identifiers, without the enhanced/hierarchical prompt features. Settings live in [configs/model_qwen3_4b_shared.yaml](configs/model_qwen3_4b_shared.yaml) and [configs/eval_qwen3_4b_legacy.yaml](configs/eval_qwen3_4b_legacy.yaml). The restored batch-18, checkpointing-off setup targets an A100 40GB; smaller GPUs need lower batches/checkpointing. This restores the design, not bit-for-bit historical weights or missing training metadata.

The model is compared with **four baselines**:

- **Last value:** repeat the most recent observation.
- **Seasonal naive:** use the corresponding quarter from the previous year.
- **Ridge:** predict from past quarterly values.
- **Enhanced Ridge:** add seasonality, trend, volatility, and zero-value features.

Training and evaluation follow time order, with no random train/test split. Training targets end at **2022Q4**; validation uses origins from **2023Q1–2024Q2**, with targets inside that period. Test origins start at **2024Q3** and require an observed future target, so the latest usable origin differs by horizon.

Notebook 07 defaults to **legacy 151,727 versus enhanced 75,000 and four baselines** on **1,000 matching cases per horizon** (**3,000 total**). Legacy remains the pinned final model; comparison does not change that selection. Legacy uses eight past quarters, while enhanced retains the earlier comparison's declared twelve-quarter history and enhanced prompt. The original enhanced training manifest is missing, so that declaration is not independently verified training metadata. Notebook 07 reconstructs extra past quarters from checksum-verified normalized data, excludes incomplete histories for every model before sampling, and saves an exclusion report. Baselines keep their saved eight-quarter design. This compares complete saved systems, not training size alone.

Historical results remain unchanged in `reports/model_evaluations/comparisons/`, including [comparison_c64a6d3e5c17](reports/model_evaluations/comparisons/comparison_c64a6d3e5c17/comparison_manifest.json). New results include `adapter_and_baseline_metrics.csv` (both adapters and baselines) and `current_vs_previous_adapters.csv` (legacy versus enhanced), with sMAPE, MAE, RMSE, MASE and parsing coverage. No retraining is needed; if the input checks report stale baseline/panel files, refresh 03–05 first.

The existing test cases have been used repeatedly during development. A final performance claim needs untouched future data. Historical runs also differ in training data and prompts, and some original training records are missing; the comparison reports record those limitations.

## Repository layout

```text
MLProject-JobAI/
├── notebooks/                    # ordered pipeline and optional experiments
├── jobai/                        # shared forecasting and model helpers
├── configs/                      # data, evaluation, training, and RAG settings
├── data/raw/                     # saved PxWeb responses
├── data/processed/               # normalized data and forecasting examples
├── data/manifests/               # source records and dataset/run metadata
├── reports/                      # metrics, predictions, and figures
├── models/                       # base-model caches, adapters, and checkpoints
├── tests/                        # checks for the shared forecasting helpers
├── requirements.txt              # general project dependencies
└── requirements-train-colab.txt  # pinned training dependencies for notebook 06
```

## Contributing

Create a focused branch and keep notebook changes reproducible. Use seed **42**, preserve source and run records, and keep experiment-specific settings separate from shared defaults. Before committing notebook changes, run the affected notebooks with `jupyter nbconvert --execute` and keep only relevant outputs. Open a PR with the validation results for review.

## Fine-tuning history

### Runs with saved evaluation evidence

The retained reports identify **nine distinct trained runs**, not just six experiment groups. Links identify the exact run; reevaluating the same adapter does not count as another training run.

| Model and design | Horizons | Training examples | Short outcome / saved evidence |
| --- | --- | ---: | --- |
| Qwen3-4B legacy | H1/H2/H4 shared | 6,000 | [Initial small-data run](reports/model_evaluations/qwen-qwen3-4b__20260919T083224Z/evaluation_manifest.json). |
| **Qwen3-4B legacy** | **H1/H2/H4 shared** | **151,727** | **[Selected final model](reports/model_evaluations/qwen-qwen3-4b__20260919T125120Z/evaluation_manifest.json).** |
| Qwen3-4B enhanced | H1/H2/H4 shared | 6,000 | [Small-data enhanced-input trial](reports/model_evaluations/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T071954Z/evaluation_manifest.json). |
| Qwen3-4B enhanced | H1/H2/H4 shared | 75,000 | [Larger enhanced-input trial](reports/model_evaluations/qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z/evaluation_manifest.json). |
| Qwen3-4B enhanced | H1 only | 10,000 | [One-quarter specialization trial](reports/model_evaluations/comparisons/comparison_d68eeee6e93c/runs/qwen-qwen3-4b__qwen3_4b_h1__enhanced_v1__h1__20260920T141640Z/evaluation_manifest.json). |
| Qwen3-4B enhanced | H2 only | 10,000 | [Two-quarter specialization trial](reports/model_evaluations/comparisons/comparison_d68eeee6e93c/runs/qwen-qwen3-4b__qwen3_4b_h2__enhanced_v1__h2__20260920T143643Z/evaluation_manifest.json). |
| Qwen3-4B enhanced | H4 only | 10,000 | [Four-quarter specialization trial](reports/model_evaluations/comparisons/comparison_d68eeee6e93c/runs/qwen-qwen3-4b__qwen3_4b_h4__enhanced_v1__h4__20260920T150045Z/evaluation_manifest.json). |
| Qwen3-4B enhanced | H1 only | 60,000 | [Larger H1 specialization trial](reports/model_evaluations/comparisons/comparison_6cb50750fd76/runs/qwen-qwen3-4b__qwen3_4b_h1__enhanced_v1__h1__20260920T184401Z/evaluation_manifest.json). |
| Qwen3.5-4B shared enhanced | H1/H2/H4 shared | 75,000 | [Evaluated against retained Qwen3-4B adapters](reports/model_evaluations/comparisons/comparison_c64a6d3e5c17/comparison_manifest.json); not selected. |

### Earlier configurations and exploratory attempts

These are part of the development history, but **no completed evaluation for them was found in the retained run reports**. A configuration file alone does not prove training finished.

| Model / project branch | What was tried | Evidence |
| --- | --- | --- |
| Qwen3.5-9B | Early preferred QLoRA model; later replaced in the active workflow. | [Historical configuration, `0319d2f`](https://github.com/Lion504/JobAI/blob/0319d2f/configs/model.yaml). |
| Qwen3-8B | Text-only alternative before the Qwen3-4B runs. | [Historical configuration, `77b22bf`](https://github.com/Lion504/JobAI/blob/77b22bf/configs/model.yaml). |
| Archived 27B experiment, labelled `Qwen3.8-27B` in the project | Separate version-2 notebook and large-model configuration; not the selected model. | [Archived configuration](configs/model_v2_qwen38_27b.yaml). The name here records the project label, not independent verification of a released model ID. |

We also tested different batch sizes, gradient accumulation, checkpointing, validation budgets, caching and GPU setups. Those setup changes, interrupted attempts, and repeated evaluations are not counted as distinct completed training runs unless a separate run record supports them.

Runs differ in training budget, features, and sometimes evaluation cohorts. Older `pilot_passed`/`pilot_failed` labels apply only to that report's sample and criteria; they are not permanent rankings or deployment approval. Some original training manifests are missing, so evaluation records remain the available evidence. This history does not claim architecture-only superiority or under-10% sMAPE. No adapter or historical report is removed by documenting the history.
