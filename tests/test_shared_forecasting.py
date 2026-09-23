import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from jobai.forecasting import (KEYS, add_errors, balanced_sample, deterministic_sample, encode_record,
    matched_test_sample, metric_tables, parse_prediction, prompt_messages,
    sha256, verify_pipeline, write_json)
from jobai.evaluation import PredictionCache, comparison_input_identity, prepare_comparison_history
from jobai.model_runtime import (adapter_identity, base_directory, check_fast_kernels,
                                discover_comparison_runs, selected_final_run)
from jobai.configuration import load_profile, DEFAULT_COMPARISON_CONFIG

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


def test_legacy_prompt_matches_the_original_shared_format():
    r = row()
    r['input_values_json'] = json.dumps([10, 20, 30, 40, 50, 60, 70, 80])
    result = prompt_messages(r, 'legacy_v1', 8)
    assert result == [
        {'role': 'system', 'content': 'You forecast Finnish registered job vacancies. '
         'Return exactly one JSON object with one numeric field named target_scaled_change.'},
        {'role': 'user', 'content': '\n'.join([
            'Series family: KEHA', 'Table: 12tu', 'Dimensions: {"region": "01"}',
            'History start: 2023Q1',
            "Eight quarterly vacancy values, oldest to newest: ['10', '20', '30', '40', '50', '60', '70', '80']",
            'Forecast origin: 2024Q4', 'Forecast horizon: 1 quarter(s)',
            'Target quarter: 2025Q1', 'Scale: 80',
            'Predict target_scaled_change = (target vacancy value - latest history value) / scale.'
        ])}]


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
    write_json(repo / 'data/processed/normalization_assertions.json', {
        '12tu': {'status': 'passed', 'schema_version': 2,
                 'normalized_csv': {'path': str(source.relative_to(repo)), 'sha256': sha256(source)}}})
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
    for name in ['02_normalize_and_validate_data', '03_eda_and_series_selection', '04_baselines_and_rolling_evaluation', '05_prepare_panel_dataset',
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


@pytest.mark.parametrize('with_previous', [False, True])
def test_07_scoring_cells_on_common_and_failed_predictions(with_previous):
    nb = json.loads((ROOT / 'notebooks/07_evaluate_model.ipynb').read_text())
    sample = pd.DataFrame([dict(example_id=f'e{h}-{i}', table_id='12tu', series_id=f's{i}',
        forecast_scope='province', horizon_q=h, origin_quarter='2024Q4', target_quarter='2025Q1',
        last_value=90, target_value=100) for h in [1, 2, 4] for i in [0, 1]])
    required = {'last_value', 'seasonal_naive', 'ridge', 'ridge_enhanced'}
    baseline = pd.DataFrame([{**r, 'model': m, 'y_true': 100, 'y_pred': 90, 'mase_scale': 20}
        for r in sample.to_dict('records') for m in required])
    adapters = pd.DataFrame([{**r, 'model': m, 'y_true': 100, 'y_pred': 95, 'parsed': True}
        for r in sample.to_dict('records') for m in (['new', 'old'] if with_previous else ['new'])])
    runs = [{'model_label': 'new', 'train_count': 75000, 'role': 'current_shared', 'provenance': 'training_manifest'},
            {'model_label': 'old', 'train_count': 6000, 'role': 'previous_shared', 'provenance': 'training_manifest'}]
    if not with_previous:
        runs = runs[:1]
    namespace = dict(pd=pd, np=np, model_predictions=adapters, sample=sample, matched_baselines=baseline,
        runs=runs, add_errors=add_errors, metric_tables=metric_tables, display=lambda *a: None,
        required_models=required, METRICS=['MAE', 'RMSE', 'MASE', 'sMAPE_pct'],
        COMPARE_CFG={'minimum_parse_success': .99})
    for fail in [False, True]:
        if fail:
            adapters.loc[0, ['parsed', 'y_pred']] = [False, np.nan]
        exec(''.join(nb['cells'][10]['source']), namespace)
        exec(''.join(nb['cells'][12]['source']), namespace)
        assert len(namespace['summary']) == (18 if with_previous else 15)
        assert namespace['previous_comparison_performed'] == with_previous
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
    namespace = dict(PROMPT_SCHEMA='v1', SEED=42, MODEL_ID='Qwen/Qwen3.5-4B',
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


def test_default_and_explicit_experiment_profiles(monkeypatch):
    monkeypatch.delenv('JOBAI_MODEL_CONFIG', raising=False)
    path, model, eval_path, cfg = load_profile(ROOT)
    assert path.name == 'model_qwen3_4b_shared.yaml'
    assert eval_path.name == 'eval_qwen3_4b_legacy.yaml'
    assert model['candidate_to_run'] == 'Qwen/Qwen3-4B'
    assert model['training']['training_horizons'] == cfg['horizons'] == [1, 2, 4]
    assert model['training']['max_train_examples'] is None
    assert model['training']['prompt_schema'] == cfg['engineered_feature_set'] == 'legacy_v1'
    assert cfg['feature_window_quarters'] == 8
    assert cfg['panel_selection']['target_tables'] == ['12tu', '12tw']
    assert not cfg['hierarchical_context'] and not cfg['export_horizon_files']
    monkeypatch.setenv('JOBAI_MODEL_CONFIG', 'configs/model.yaml')
    _, experiment, _, experiment_eval = load_profile(ROOT)
    assert experiment['candidate_to_run'] == 'Qwen/Qwen3.5-4B'
    assert experiment_eval['feature_window_quarters'] == 12
    comparison = yaml.safe_load((ROOT / DEFAULT_COMPARISON_CONFIG).read_text())
    # The explicit final evaluation profile wins even with an old env override.
    _, final, _, _ = load_profile(ROOT, comparison['model_config'])
    assert final['candidate_to_run'] == 'Qwen/Qwen3-4B'
    assert comparison['selected_model_config'] == 'configs/final_model.yaml'
    assert comparison['history_source'] == 'normalized'
    assert len(comparison['previous_runs']) == 1
    other = comparison['previous_runs'][0]
    assert other['run_id'] == 'qwen-qwen3-4b__enhanced_v1__h1-2-4__20260920T082714Z'
    assert other['train_examples'] == 75000 and other['history_quarters'] == 12
    assert other['prompt_schema'] == 'enhanced_shared_v1'


def test_legacy_sampling_is_one_mixed_pool_without_replacement():
    frame = examples(20)
    frame = frame[(frame.horizon_q == 1) | (frame.horizon_q == 2) & (frame.index % 2 == 0) |
                  (frame.horizon_q == 4) & (frame.index % 4 == 0)]
    whole = deterministic_sample(frame, None, 42)
    assert whole.horizon_q.value_counts().to_dict() == {1: 20, 2: 10, 4: 5}
    subset = deterministic_sample(frame, 12, 42)
    assert len(subset) == 12 and subset.example_id.is_unique
    pd.testing.assert_frame_equal(subset, deterministic_sample(frame.sample(frac=1), 12, 42))
    with pytest.raises(AssertionError, match='budget'):
        deterministic_sample(frame, len(frame) + 1, 42)


def pinned_fixture(repo):
    adapter = repo / 'models/adapters/final/final_adapter'
    adapter.mkdir(parents=True)
    write_json(adapter / 'adapter_config.json', {'base_model_name_or_path': 'Qwen/Qwen3-4B'})
    (adapter / 'adapter_model.safetensors').write_bytes(b'final weights')
    spec = dict(run_id='final', base_model='Qwen/Qwen3-4B', label='final_qwen3',
        train_examples=151727, training_horizons=[1, 2, 4], target_tables=['12tu', '12tw'],
        prompt_schema='legacy_v1', history_quarters=8, max_seq_length=768,
        adapter_path=str(adapter.relative_to(repo)), adapter_sha256=adapter_identity(adapter),
        source_manifest='reports/model_evaluations/final/evaluation_manifest.json')
    write_json(repo / spec['source_manifest'], {**spec, 'test_data_used_for_training': False})
    (repo / 'configs').mkdir()
    (repo / 'configs/final_model.yaml').write_text(yaml.safe_dump(spec))
    # A later run must not replace the pinned selection.
    write_json(repo / 'data/manifests/finetune_run_manifest.json', {'base_model': 'Qwen/Qwen3.5-4B'})
    return spec, adapter


def test_final_selection_ignores_latest_and_needs_no_new_training(tmp_path):
    spec, _ = pinned_fixture(tmp_path)
    cfg = {'selected_model_config': 'configs/final_model.yaml', 'previous_runs': []}
    card = {'feature_window_quarters': 8, 'selected_series_by_table': {'12tu': 1, '12tw': 1}}
    runs = discover_comparison_runs(tmp_path, cfg, {'candidate_to_run': 'Qwen/Qwen3-4B'}, card)
    assert len(runs) == 1 and runs[0]['run_id'] == 'final'
    assert runs[0]['adapter_sha256'] == spec['adapter_sha256']
    assert runs[0]['provenance'] == 'explicit_historical_compatibility_declaration'
    assert runs[0]['selection_role'] == 'pinned_final'
    with pytest.raises(AssertionError, match='scope'):
        discover_comparison_runs(tmp_path, cfg, {'candidate_to_run': 'Qwen/Qwen3-4B'},
                                 {**card, 'selected_series_by_table': {'11l1': 1}})


@pytest.mark.parametrize('tamper', ['weights', 'base', 'count'])
def test_final_selection_rejects_changed_artifacts(tmp_path, tamper):
    spec, adapter = pinned_fixture(tmp_path)
    if tamper == 'weights':
        (adapter / 'adapter_model.safetensors').write_bytes(b'wrong weights')
    elif tamper == 'base':
        write_json(adapter / 'adapter_config.json', {'base_model_name_or_path': 'Qwen/Qwen3.5-4B'})
    else:
        write_json(tmp_path / spec['source_manifest'], {**spec, 'train_examples': 6000})
    with pytest.raises(AssertionError):
        selected_final_run(tmp_path)


def test_qwen3_does_not_require_multimodal_kernels(monkeypatch):
    import jobai.model_runtime as runtime
    monkeypatch.setattr(runtime.importlib, 'import_module', lambda _: pytest.fail('Qwen3 must not import Qwen3.5 kernels'))
    assert check_fast_kernels('Qwen/Qwen3-4B', required=True) == {}


def test_longer_comparator_requires_explicit_history_reconstruction(tmp_path):
    _, adapter = pinned_fixture(tmp_path)
    spec = dict(run_id='enhanced', base_model='Qwen/Qwen3-4B', label='enhanced',
                train_examples=75000, history_quarters=12,
                prompt_schema='enhanced_shared_v1', max_seq_length=768)
    write_json(tmp_path / 'reports/model_evaluations/enhanced/evaluation_manifest.json',
               {**spec, 'adapter_path': str(adapter.relative_to(tmp_path))})
    cfg = {'selected_model_config': 'configs/final_model.yaml', 'previous_runs': [spec]}
    card = {'feature_window_quarters': 8, 'selected_series_by_table': {'12tu': 1, '12tw': 1}}
    model = {'candidate_to_run': 'Qwen/Qwen3-4B'}
    with pytest.raises(AssertionError, match='more history'):
        discover_comparison_runs(tmp_path, cfg, model, card)
    runs = discover_comparison_runs(tmp_path, {**cfg, 'history_source': 'normalized'}, model, card)
    assert [r['history_quarters'] for r in runs] == [8, 12]


def comparison_history_fixture(repo):
    path = repo / 'data/processed/12tu__normalized.csv'
    path.parent.mkdir(parents=True)
    records = []
    for h in [1, 2, 4]:
        records.append({**row(), 'example_id': f'e{h}', 'series_id': 'region01',
                        'horizon_q': h, 'target_quarter': str(pd.Period('2024Q4', freq='Q') + h),
                        'input_values_json': json.dumps(list(range(50, 130, 10))),
                        'last_value': 120, 'scale': 120, 'target_value': 120 + h * 10})
    data = pd.DataFrame({'region': ['01'] * 16,
                         'timeperiod_q': pd.period_range('2022Q1', periods=16, freq='Q').astype(str),
                         'value': list(range(10, 170, 10))})
    data.to_csv(path, index=False)
    runs = [dict(model_label='legacy', prompt_schema='legacy_v1', history_quarters=8),
            dict(model_label='enhanced', prompt_schema='enhanced_shared_v1', history_quarters=12)]
    return path, pd.DataFrame(records), runs, data


def test_comparison_extends_only_past_and_preserves_original_eight(tmp_path):
    path, frame, runs, _ = comparison_history_fixture(tmp_path)
    original = frame.copy(deep=True)
    prepared, excluded, evidence = prepare_comparison_history(tmp_path, frame, runs, 'normalized',
        {str(path.relative_to(tmp_path)): sha256(path)})
    assert len(prepared) == 3 and excluded.empty
    pd.testing.assert_frame_equal(original, frame)  # No panel/training mutation.
    assert evidence['verified_against_baseline_sources']
    for before, after in zip(frame.to_dict('records'), prepared.to_dict('records')):
        assert json.loads(after['input_values_json']) == list(range(10, 130, 10))
        assert prompt_messages(before, 'legacy_v1', 8) == prompt_messages(after, 'legacy_v1', 8)
        assert after['last_value'] == before['last_value'] and after['scale'] == before['scale']
        assert after['target_value'] not in json.loads(after['input_values_json'])
    with pytest.raises(AssertionError):
        prepare_comparison_history(tmp_path, frame, runs, 'panel')
    panel, audit, _ = prepare_comparison_history(tmp_path, frame, runs[:1], 'panel')
    pd.testing.assert_frame_equal(panel, frame)
    assert audit.empty


@pytest.mark.parametrize('change', ['absent_quarter', 'null_value'])
def test_incomplete_extra_history_excludes_common_cases(tmp_path, change):
    path, frame, runs, data = comparison_history_fixture(tmp_path)
    if change == 'absent_quarter':
        data = data.iloc[1:]
    else:
        data.loc[0, 'value'] = np.nan
    data.to_csv(path, index=False)
    prepared, audit, evidence = prepare_comparison_history(tmp_path, frame, runs, 'normalized')
    assert prepared.empty and len(audit) == 3
    assert set(audit.example_id) == set(frame.example_id)
    assert audit.missing_quarters.eq('2022Q1').all()
    assert evidence['excluded_rows'] == 3
    assert not evidence['verified_against_baseline_sources']


@pytest.mark.parametrize('change', ['suffix', 'target', 'duplicate', 'source_hash', 'last_value'])
def test_changed_panel_or_source_is_not_silently_excluded(tmp_path, change):
    path, frame, runs, data = comparison_history_fixture(tmp_path)
    sources = None
    if change == 'suffix':
        data.loc[4, 'value'] = 999
    elif change == 'target':
        data.loc[12, 'value'] = 999
    elif change == 'duplicate':
        data = pd.concat([data, data.iloc[:1]])
    elif change == 'source_hash':
        sources = {str(path.relative_to(tmp_path)): 'wronghash'}
    else:
        frame.loc[0, 'last_value'] = 999
    data.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        prepare_comparison_history(tmp_path, frame, runs, 'normalized', sources)


def test_cache_identity_includes_added_history_not_just_case_ids(tmp_path):
    _, frame, runs, _ = comparison_history_fixture(tmp_path)
    prepared, _, _ = prepare_comparison_history(tmp_path, frame, runs, 'normalized')
    changed = prepared.copy()
    values = json.loads(changed.loc[0, 'input_values_json'])
    values[0] = 999
    changed.loc[0, 'input_values_json'] = json.dumps(values)
    assert prepared.example_id.tolist() == changed.example_id.tolist()
    assert comparison_input_identity(prepared, runs[1]) != comparison_input_identity(changed, runs[1])
    assert comparison_input_identity(prepared, runs[0]) == comparison_input_identity(changed, runs[0])


def test_07_case_selection_uses_reconstructed_common_histories(tmp_path):
    path, frame, runs, _ = comparison_history_fixture(tmp_path)
    frame['split'] = 'test'
    frame.to_json(tmp_path / 'data/processed/panel_test.jsonl', orient='records', lines=True)
    models = {'last_value', 'seasonal_naive', 'ridge', 'ridge_enhanced'}
    baseline = pd.DataFrame([{**{k: r[k] for k in KEYS}, 'split': 'test', 'model': model,
                              'y_true': r['target_value'], 'mase_scale': 10}
                             for r in frame.to_dict('records') for model in models])
    (tmp_path / 'reports').mkdir()
    baseline.to_csv(tmp_path / 'reports/baseline_predictions.csv', index=False)
    ns = dict(REPO=tmp_path, pd=pd, np=np, runs=runs, display=lambda *_: None,
              matched_test_sample=matched_test_sample, prepare_comparison_history=prepare_comparison_history,
              KEYS=KEYS, COMPARE_CFG={'history_source': 'normalized', 'test_examples_per_horizon': 1},
              EVAL_CFG={'seed': 42}, panel_card={'files': {'test': {'path': 'data/processed/panel_test.jsonl'}}},
              baseline_manifest={'normalized_inputs': {str(path.relative_to(tmp_path)): sha256(path)}})
    exec(code_cell('07_evaluate_model', 'shared-eval-6'), ns)
    sample, matched = ns['sample'], ns['matched_baselines']
    assert sample.groupby('horizon_q').size().to_dict() == {1: 1, 2: 1, 4: 1}
    assert sample.input_values_json.map(lambda x: len(json.loads(x))).eq(12).all()
    assert matched.groupby('example_id').model.nunique().eq(4).all()
    assert set(matched.example_id) == set(sample.example_id)
    assert ns['history_exclusions'].empty


def code_cell(notebook, cell_id):
    nb = json.loads((ROOT / 'notebooks' / f'{notebook}.ipynb').read_text())
    return ''.join(next(c['source'] for c in nb['cells'] if c.get('id') == cell_id))


def test_03_two_table_selection_and_tu_helper_arguments(tmp_path):
    _, _, _, cfg = load_profile(ROOT, 'configs/model_qwen3_4b_shared.yaml')
    frames = {
        '12tu': pd.DataFrame([dict(Alue='MK01', Ammattiryhmä='1', contentscode='AVPAIKATYHT',
            **{'Työmarkkina-asema': 'SSS'}, timeperiod_q=f'{2013+i//4}Q{i%4+1}', value=10+i) for i in range(54)]),
        '12tw': pd.DataFrame([dict(Alue='MK01', Toimiala='SSS', contentscode='AVPAIKATLOPUSSA',
            **{'Työnantajan sektori': 'SSS', 'Työpaikan työn kesto': 'SSS'},
            timeperiod_q=f'{2013+i//4}Q{i%4+1}', value=20+i) for i in range(54)]),
        '12r5': pd.DataFrame([dict(Alue='KU091', contentscode='UUDETAVP',
            timeperiod_q=f'{2013+i//4}Q{i%4+1}', value=30+i) for i in range(54)]),
    }
    for tid, frame in frames.items():
        frame.to_csv(tmp_path / f'{tid}__normalized.csv', index=False)
    namespace = dict(pd=pd, np=np, json=json, PRO=tmp_path, EVAL=cfg, SEL=cfg['panel_selection'],
        target_codes=cfg['panel_selection']['target_contentscodes'], catalog_rows=[], display=lambda *a: None)
    exec(code_cell('03_eda_and_series_selection', 'helpers'), namespace)
    exec(code_cell('03_eda_and_series_selection', 'panel-selection'), namespace)
    catalog = namespace['catalog']
    assert set(catalog.table_id) == set(frames)
    assert set(catalog.loc[catalog.is_forecast_target, 'table_id']) == {'12tu', '12tw'}
    assert not catalog[catalog.table_id == '12r5'].is_forecast_target.any()


def test_05_legacy_construction_needs_no_atp_or_geographic_table(tmp_path):
    import hashlib
    _, _, _, cfg = load_profile(ROOT, 'configs/model_qwen3_4b_shared.yaml')
    dimensions = {'Alue': 'MK01', 'Ammattiryhmä': '1', 'Työmarkkina-asema': 'SSS', 'contentscode': 'AVPAIKATYHT'}
    frame = pd.DataFrame([{**dimensions, 'timeperiod_q': f'{2020+i//4}Q{i%4+1}', 'value': 10+i}
                          for i in range(26)])
    frame.to_csv(tmp_path / '12tu__normalized.csv', index=False)
    targets = pd.DataFrame([dict(table_id='12tu', series_id='series', series_family='KEHA',
        dimensions_json=json.dumps(dimensions), forecast_scope='province_occupation', measure_code='AVPAIKATYHT')])
    namespace = dict(pd=pd, np=np, json=json, hashlib=hashlib, PRO=tmp_path, CFG=cfg,
        WINDOW=8, HORIZONS=[1, 2, 4], target_selection=targets, display=lambda *a: None)
    exec(code_cell('05_prepare_panel_dataset', 'construct'), namespace)
    dataset = namespace['dataset']
    assert not namespace['context_maps']
    assert set(dataset.horizon_q) == {1, 2, 4} and set(dataset.split) == {'train', 'validation', 'test'}
    assert dataset.input_values_json.map(lambda v: len(json.loads(v))).eq(8).all()
    assert dataset.engineered_features_json.eq('{}').all()
    assert dataset.hierarchical_context_json.eq('{}').all()
    assert dataset.feature_set.eq('legacy_v1').all()


def test_05_preflight_checks_baseline_normalized_fingerprints(tmp_path):
    import hashlib
    cfg, _ = fixture_pipeline(tmp_path)
    pro = tmp_path / 'data/processed'
    selection_path = pro / 'selected_series.csv'
    pd.DataFrame([dict(table_id='12tu', series_id='a', selected=True,
                       is_forecast_target=True, forecast_scope='province')]).to_csv(selection_path, index=False)
    baseline_path = tmp_path / 'reports/baseline_run_manifest.json'
    baseline = json.loads(baseline_path.read_text())
    baseline['selection_catalog_sha256'] = sha256(selection_path)
    write_json(baseline_path, baseline)
    ns = dict(REPO=tmp_path, PRO=pro, REPORTS=tmp_path / 'reports', CFG=cfg,
              WINDOW=12, HORIZONS=[1, 2, 4], pd=pd, json=json, hashlib=hashlib, display=lambda *a: None)
    source = code_cell('05_prepare_panel_dataset', 'preflight')
    exec(source, ns)
    baseline['normalized_inputs']['data/processed/12tu__normalized.csv'] = 'wrong hash'
    write_json(baseline_path, baseline)
    with pytest.raises(AssertionError, match='Normalized data differs from the baseline'):
        exec(source, ns)
