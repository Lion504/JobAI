import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from jobai.forecasting import (KEYS, add_errors, balanced_sample, encode_record,
    matched_test_sample, metric_tables, parse_prediction, prompt_messages,
    sha256, verify_pipeline, write_json)
from jobai.evaluation import PredictionCache
from jobai.model_runtime import base_directory, discover_comparison_runs

ROOT = Path(__file__).resolve().parents[1]


def examples(n=20):
    return pd.DataFrame([dict(example_id=f'{h}-{i}', horizon_q=h,
        table_id='11l1' if i == 0 else ('12tu' if i % 2 else '12tw'),
        forecast_scope='province', target_size_band=str(i % 3),
        volatility_band=str(i % 2), series_id=f's{i}', origin_quarter='2024Q4')
        for h in (1, 2, 4) for i in range(n)])


def test_balanced_sampling_retains_atp_and_equal_horizons():
    frame = examples()
    result = balanced_sample(frame, 30, 42)
    assert result.horizon_q.value_counts().to_dict() == {1: 10, 2: 10, 4: 10}
    assert result.example_id.is_unique
    assert set(frame[frame.table_id == '11l1'].example_id) <= set(result.example_id)
    pd.testing.assert_frame_equal(result, balanced_sample(frame.sample(frac=1), 30, 42))


def test_actual_75k_budget():
    result = balanced_sample(examples(26000), 75000, 42)
    assert result.groupby('horizon_q').size().to_dict() == {1: 25000, 2: 25000, 4: 25000}


def test_sampling_fails_instead_of_repeating_short_horizons():
    with pytest.raises(AssertionError, match='TOTAL budget'):
        balanced_sample(examples(4), 15, 42)
    with pytest.raises(AssertionError, match='Duplicate'):
        balanced_sample(pd.concat([examples(), examples()]), 30, 42)
    result = balanced_sample(examples(), 10, 42, priority_tables=())
    assert result.groupby('horizon_q').size().to_dict() == {1: 4, 2: 3, 4: 3}


def test_matched_test_sample_is_repeatable_and_not_target_based():
    frame = examples()
    result = matched_test_sample(frame, 10, 42)
    assert result.groupby('horizon_q').size().eq(10).all()
    frame['target_value'] = np.arange(len(frame)) * 1000
    assert result.example_id.tolist() == matched_test_sample(frame, 10, 42).example_id.tolist()


def row():
    return dict(input_values_json=json.dumps(list(range(10, 130, 10))),
        dimensions_json='{"region": "01"}', origin_quarter='2024Q4',
        target_quarter='2025Q1', horizon_q=1, series_family='KEHA',
        table_id='12tu', forecast_scope='province', measure_code='AV',
        hierarchical_context_json='{}', target_value=123456789)


def test_prompt_does_not_contain_future_target():
    r = row()
    text = json.dumps(prompt_messages(r))
    assert '123456789' not in text
    assert '12 quarterly values' in text
    assert 'Eight quarterly vacancy values' in json.dumps(prompt_messages(r, 'legacy_v1', 8))


class Tokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['add_generation_prompt'] is True
        return 'prompt'
    def __call__(self, text, **kwargs):
        return {'input_ids': [10] * len(text)}


def test_completion_only_labels_and_no_silent_truncation():
    encoded = encode_record(Tokenizer(), [], 'answer', 20)
    assert encoded['labels'][:6] == [-100] * 6
    assert encoded['labels'][6:] == [10] * 6 + [2]
    assert encoded['input_ids'][-1] == 2
    with pytest.raises(AssertionError, match='no records were truncated'):
        encode_record(Tokenizer(), [], 'answer', 8)


@pytest.mark.parametrize('text, expected', [
    ('<think>hi</think>{"target_scaled_change": 0.5}', 0.5),
    ('{"target_scaled_change": true}', None),
    ('{"target_scaled_change": "NaN"}', None),
    ('no forecast', None)])
def test_parse(text, expected):
    assert parse_prediction(text) == expected


def test_macro_metrics_and_zero_smape():
    frame = pd.DataFrame([dict(model='m', table_id='t', forecast_scope='p', series_id=s,
        horizon_q=1, y_true=y, y_pred=p, mase_scale=10)
        for s, y, p in [('a', 10, 0), ('a', 10, 0), ('b', 0, 0)]])
    errors = add_errors(frame)
    assert errors.smape_component_pct.tolist() == [200, 200, 0]
    _, summary, _ = metric_tables(errors)
    assert summary.iloc[0].MAE == 5  # equal series weight, not 20/3
    assert summary.iloc[0].sMAPE_pct == 100


def test_prediction_cache_recovery_and_invalidation(tmp_path):
    spec = {'run_id': 'r', 'prompt': 'v1'}
    cache = PredictionCache(tmp_path, spec, ['a', 'b'])
    cache.save([{'example_id': 'a', 'run_id': 'r', 'parsed': False}])
    assert set(PredictionCache(tmp_path, spec, ['a', 'b']).load()) == {'a'}
    other = PredictionCache(tmp_path, {**spec, 'prompt': 'v2'}, ['a', 'b'])
    assert other.key != cache.key and other.load() == {}
    cache.save([{'example_id': 'a', 'run_id': 'r'}])
    with pytest.raises(AssertionError, match='Duplicate'):
        cache.load()


def fixture_pipeline(repo):
    cfg = dict(feature_window_quarters=12, horizons=[1, 2, 4], splits={'train': 'past'},
               engineered_feature_set='enhanced_v1', panel_selection={'target_tables': ['12tu']})
    catalog = repo / 'data/processed/selected_series.csv'
    catalog.parent.mkdir(parents=True)
    catalog.write_text('table_id\n12tu\n')
    source = repo / 'data/processed/12tu__normalized.csv'
    source.write_text('value\n1\n')
    output = repo / 'reports/baseline_predictions.csv'
    output.parent.mkdir(parents=True)
    output.write_text('prediction\n1\n')
    baseline = dict(feature_window_quarters=12, horizons_q=[1, 2, 4], splits=cfg['splits'],
        engineered_feature_set='enhanced_v1', selection_catalog_sha256=sha256(catalog),
        models=['last_value', 'seasonal_naive', 'ridge', 'ridge_enhanced'],
        normalized_inputs={str(source.relative_to(repo)): sha256(source)},
        outputs={str(output.relative_to(repo)): {'sha256': sha256(output)}})
    bp = repo / 'reports/baseline_run_manifest.json'
    write_json(bp, baseline)
    files = {}
    for split in ['train', 'validation', 'test']:
        path = repo / f'data/processed/panel_{split}.jsonl'
        path.write_text('{}\n')
        files[split] = {'path': str(path.relative_to(repo)), 'sha256': sha256(path)}
    card = dict(feature_window_quarters=12, horizons_q=[1, 2, 4], splits=cfg['splits'],
        gate={'finetuning_ready': True}, feature_set='enhanced_v1', selected_series_by_table={'12tu': 1},
        selection_catalog_sha256=sha256(catalog), baseline_manifest_sha256=sha256(bp), files=files)
    write_json(repo / 'data/manifests/panel_dataset_card.json', card)
    return cfg, card


def test_preflight_detects_stale_data_without_reading_test(tmp_path):
    cfg, card = fixture_pipeline(tmp_path)
    (tmp_path / card['files']['test']['path']).unlink()
    verify_pipeline(tmp_path, cfg)  # 06 never needs test records
    with pytest.raises(AssertionError, match='stale'):
        verify_pipeline(tmp_path, {**cfg, 'feature_window_quarters': 8})
    (tmp_path / 'data/processed/12tu__normalized.csv').write_text('changed')
    with pytest.raises(AssertionError, match='Normalized source changed'):
        verify_pipeline(tmp_path, cfg)


def test_shared_roster_requires_one_current_and_explicit_old_run(tmp_path):
    _, card = fixture_pipeline(tmp_path)
    for name in ['current', 'previous']:
        adapter = tmp_path / 'models/adapters' / name / 'final_adapter'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'fixture')
    current = dict(run_id='current', base_model='Qwen/Qwen3.5-4B', training_horizons=[1, 2, 4],
        prompt_schema='five_table_shared_v1', feature_window_quarters=12, train_examples_used=75000,
        model_config={'training': {'max_seq_length': 768}},
        adapter_path='models/adapters/current/final_adapter',
        **{f'{s}_split_sha256': card['files'][s]['sha256'] for s in ['train', 'validation']})
    write_json(tmp_path / 'data/manifests/finetune_run_manifest.json', current)
    previous = dict(run_id='previous', base_model='Qwen/Qwen3-4B', train_examples=6000,
                    adapter_path='models/adapters/previous/final_adapter')
    write_json(tmp_path / 'reports/model_evaluations/previous/evaluation_manifest.json', previous)
    spec = dict(run_id='previous', base_model='Qwen/Qwen3-4B', train_examples=6000, label='old',
                history_quarters=8, prompt_schema='legacy_v1', max_seq_length=768)
    runs = discover_comparison_runs(tmp_path, {'previous_runs': [spec]},
                                   {'candidate_to_run': 'Qwen/Qwen3.5-4B'}, card)
    assert len(runs) == 2 and runs[0]['role'] == 'current_shared'
    assert runs[1]['provenance'] == 'explicit_historical_compatibility_declaration'
    assert runs[1]['history_quarters'] == 8


def test_base_cache_never_downloads_in_evaluation(tmp_path):
    with pytest.raises(AssertionError, match='never silently downloads'):
        base_directory(tmp_path, 'Qwen/Qwen3-4B', download=False)
    base = tmp_path / 'models/base/Qwen--Qwen3-4B'
    base.mkdir(parents=True)
    (base / '.download_complete').write_text('ready')
    (base / 'config.json').write_text('{}')
    with pytest.raises(AssertionError, match='Missing base weight shards'):
        base_directory(tmp_path, 'Qwen/Qwen3-4B')
    (base / 'model.safetensors').write_bytes(b'fixture')
    assert base_directory(tmp_path, 'Qwen/Qwen3-4B') == base


def test_notebook_structure_and_shared_defaults():
    import nbformat
    for name in ['04_baselines_and_rolling_evaluation', '05_prepare_panel_dataset',
                 '06_finetune_model', '07_evaluate_model']:
        nb = nbformat.read(ROOT / f'notebooks/{name}.ipynb', as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                code = '\n'.join('pass' if line.lstrip().startswith(('%', '!')) else line
                                 for line in cell.source.splitlines())
                ast.parse(code)
    cfg = yaml.safe_load((ROOT / 'configs/model.yaml').read_text())
    assert cfg['candidate_to_run'] == 'Qwen/Qwen3.5-4B'
    assert cfg['training']['training_horizons'] == [1, 2, 4]
    assert cfg['training']['max_train_examples'] == 75000


def test_07_scoring_cells_on_common_and_failed_predictions():
    nb = json.loads((ROOT / 'notebooks/07_evaluate_model.ipynb').read_text())
    sample = pd.DataFrame([dict(example_id=f'e{h}-{i}', table_id='12tu', series_id=f's{i}',
        forecast_scope='province', horizon_q=h, origin_quarter='2024Q4', target_quarter='2025Q1',
        last_value=90, target_value=100) for h in [1, 2, 4] for i in [0, 1]])
    required = {'last_value', 'seasonal_naive', 'ridge', 'ridge_enhanced'}
    baseline = pd.DataFrame([{**r, 'model': m, 'y_true': 100, 'y_pred': 90, 'mase_scale': 20}
        for r in sample.to_dict('records') for m in required])
    adapters = pd.DataFrame([{**r, 'model': m, 'y_true': 100, 'y_pred': 95, 'parsed': True}
        for r in sample.to_dict('records') for m in ['new', 'old']])
    runs = [{'model_label': 'new', 'train_count': 75000, 'role': 'current_shared', 'provenance': 'training_manifest'},
            {'model_label': 'old', 'train_count': 6000, 'role': 'previous_shared', 'provenance': 'training_manifest'}]
    namespace = dict(pd=pd, np=np, model_predictions=adapters, sample=sample, matched_baselines=baseline,
        runs=runs, add_errors=add_errors, metric_tables=metric_tables, display=lambda *a: None,
        required_models=required, METRICS=['MAE', 'RMSE', 'MASE', 'sMAPE_pct'],
        COMPARE_CFG={'minimum_parse_success': .99})
    for fail in [False, True]:
        if fail:
            adapters.loc[0, ['parsed', 'y_pred']] = [False, np.nan]
        exec(''.join(nb['cells'][10]['source']), namespace)
        exec(''.join(nb['cells'][12]['source']), namespace)
        assert len(namespace['summary']) == 18
        assert namespace['complete_cohort'] == (not fail)
        assert namespace['fallback_summary'].n_forecasts.eq(2).all()


@pytest.mark.parametrize('export', [False, True])
def test_05_writes_combined_files_without_requiring_horizon_exports(tmp_path, export):
    import hashlib
    nb = json.loads((ROOT / 'notebooks/05_prepare_panel_dataset.ipynb').read_text())
    dataset = pd.DataFrame([dict(split=s, table_id='12tu', series_family='KEHA', forecast_scope='province',
        horizon_q=h, example_id=f'{s}-{h}', series_id='a', origin_quarter='2020Q4', target_quarter='2021Q1')
        for s in ['train', 'validation', 'test'] for h in [1, 2, 4]])
    cfg, _ = fixture_pipeline(tmp_path)
    cfg.update(export_horizon_files=export, finetuning_gate={'minimum_selected_panel_series': 1, 'minimum_train_examples': 1})
    namespace = dict(dataset=dataset, PRO=tmp_path / 'data/processed', REPORTS=tmp_path / 'reports',
        MAN=tmp_path / 'data/manifests', REPO=tmp_path, CFG=cfg, HORIZONS=[1, 2, 4], WINDOW=12,
        FEATURE_SET='enhanced_v1', normalization_assertions={'12tu': {'status': 'passed'}},
        baseline_manifest_path=tmp_path / 'reports/baseline_run_manifest.json',
        selection_path=tmp_path / 'data/processed/selected_series.csv',
        target_selection=pd.DataFrame([dict(table_id='12tu', series_id='a')]),
        hashlib=hashlib, json=json, display=lambda *a: None)
    for cell in nb['cells']:
        if cell.get('id') in ['persist', 'manifest']:
            exec(''.join(cell['source']), namespace)
    assert len(namespace['manifest']['files']) == 4
    assert len(namespace['horizon_paths']) == (9 if export else 0)
    assert len(namespace['manifest']['horizon_files']) == (3 if export else 0)


def test_training_cache_is_batch_independent_but_length_sensitive():
    from types import SimpleNamespace
    nb = json.loads((ROOT / 'notebooks/06_finetune_model.ipynb').read_text())
    node = next(n for n in ast.parse(''.join(nb['cells'][10]['source'])).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'cache_spec')
    cfg = yaml.safe_load((ROOT / 'configs/model.yaml').read_text())['training']
    namespace = dict(PROMPT_VERSION='v1', SEED=42, MODEL_ID='Qwen/Qwen3.5-4B',
        BASE_MODEL_DIR=Path('/base'), BASE_SNAPSHOT={'revision': 'abc'}, TOKENIZER_SHA='tok',
        REPO=ROOT, sha256=lambda p: str(p), HORIZONS=[1, 2, 4],
        panel_card={'files': {s: {'sha256': s} for s in ['train', 'validation']}},
        EVAL_CFG={'feature_window_quarters': 12}, TRAIN_CFG=cfg,
        importlib=SimpleNamespace(metadata=SimpleNamespace(version=lambda p: '1')))
    expression = compile(ast.Expression(node.value), '<cache spec>', 'eval')
    original = eval(expression, namespace)
    cfg['per_device_train_batch_size'] = 18
    cfg['gradient_accumulation_steps'] = 1
    assert eval(expression, namespace) == original
    cfg['max_seq_length'] = 1024
    assert eval(expression, namespace) != original
