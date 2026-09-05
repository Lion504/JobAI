# JobAI

JobAI is a team project for forecasting Finnish labour-market indicators and producing source-grounded explanations. The first prototype targets quarterly job vacancies with 1-, 2-, and 4-quarter forecasts, using the Statistics Finland PxWeb API as the numeric source and official bulletins for the narrative layer.

## Principles

- Statistics Finland PxWeb data is authoritative for numeric inputs and labels.
- The fine-tuned model must beat simple baselines (seasonal naive, last-value naive, linear regression, Prophet) — training success is not enough.
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

- First implementation uses `StatFin/atp/11l1.px` as the national quarterly vacancy target; all variable codes come from the table's runtime metadata. Provenance (raw response, query JSON, table metadata, retrieval date, response hash) lives in `data/manifests/`.
- Regional (`atp/11n1.px`) and industry (`atp/15ia.px`, TOL2025) sources are added only after the national baseline and rolling-origin evaluation are stable.
- Fine-tuning uses either a Qwen3.5 bf16 LoRA branch (Unsloth) or a Qwen3 / Qwen2.5 4-bit QLoRA branch (transformers + peft + trl + bitsandbytes). The selected branch, base model id, commit/version, hardware, and package versions are recorded by `06_finetune_model.ipynb`.
- RAG sources are whitelisted in `configs/rag.yaml` §10.1: StatFin release pages, Job Market Finland and KEHA bulletins, and official TEM pages. Legacy `mol.fi` references are treated as unverified.

## Contribution

1. Create a focused branch.
2. Add or update notebook cells; keep outputs cleared (or only the relevant ones) before committing.
3. Edit `configs/*.yaml` only for genuine defaults — per-experiment overrides go in a copy.
4. Open a PR. Reviewers re-run the affected notebook on Colab and inspect the generated manifests and reports. Merge only after review and after any updated manifests or summary reports are committed.
