import copy
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jobai.chat import ForecastService, verified_quote
from jobai.question_routing import QuestionRouter


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/forecast_routing.yaml").write_text(
        (ROOT / "configs/forecast_routing.yaml").read_text())
    (tmp_path / "data/raw").mkdir(parents=True)
    processed = tmp_path / "data/processed"
    processed.mkdir()
    regions = {"MK01": "MK01 Uusimaa", "MK06": "MK06 Pirkanmaa", "MK02": "MK02 Southwest Finland"}
    targets = {"12tu": {"2142": "2142 Civil engineers", "2512": "2512 Software developers",
                        "2141": "2141 Industrial engineers", "3221": "3221 Nurses",
                        "1321": "1321 Manufacturing managers", "9999": "9999 Absent specialists"},
               "12tw": {"C": "C Manufacturing", "F": "F Construction"}}
    catalog = []
    for table, labels in targets.items():
        dimension = "Ammattiryhmä" if table == "12tu" else "Toimiala"
        def variable(code, mapping):
            return {"code": code, "values": list(mapping), "valueTexts": list(mapping.values())}
        (tmp_path / f"data/raw/{table}__meta.json").write_text(json.dumps(
            {"variables": [variable("Alue", regions), variable(dimension, labels)]}))
        observations = []
        for region in regions:
            for target in labels:
                dims = {"Alue": region, dimension: target, "contentscode": "AV"}
                sid = table + "::" + "|".join(f"{k}={v}" for k, v in dims.items())
                catalog.append({"series_id": sid, "table_id": table, "dimensions_json": json.dumps(dims),
                                "selected": target != "9999", "is_forecast_target": True,
                                "forecast_scope": "province_occupation" if table == "12tu" else "province_industry",
                                "series_family": "KEHA", "measure_code": "AV"})
                for i, quarter in enumerate(pd.period_range("2023Q1", periods=8, freq="Q")):
                    observations.append({**dims, "timeperiod_q": str(quarter), "value": (i + 1) * 8})
        pd.DataFrame(observations).to_csv(processed / f"{table}__normalized.csv", index=False)
    pd.DataFrame(catalog).to_csv(processed / "selected_series.csv", index=False)
    (processed / "normalization_assertions.json").write_text(json.dumps(
        {table: {"status": "passed"} for table in targets}))
    return tmp_path


@pytest.fixture
def router(repo):
    return QuestionRouter(repo)


@pytest.mark.parametrize("question, table, code, region, horizons", [
    ("Civil engineer vacancies in Uusimaa next year?", "12tu", "2142", "MK01", [4]),
    ("What is the outlook for software developers in Pirkanmaa in six months?", "12tu", "2512", "MK06", [2]),
    ("Manufacturing vacancies in Uusimaa next quarter", "12tw", "C", "MK01", [1]),
    ("Manufacturing managers in Uusimaa", "12tu", "1321", "MK01", [1, 2, 4]),
    ("Construction in Southwest Finland", "12tw", "F", "MK02", [1, 2, 4]),
    ("2142 in MK01, H2", "12tu", "2142", "MK01", [2]),
    ("Industry C in MK01 in 4q", "12tw", "C", "MK01", [4]),
    ("Civil engineers in Uusimaa for 1, 2 and 4 quarters", "12tu", "2142", "MK01", [1, 2, 4]),
    ("Civil engineers in Uusimaa for two and four quarters", "12tu", "2142", "MK01", [2, 4]),
    ("Civil engineers in Uusimaa in half a year", "12tu", "2142", "MK01", [2]),
    ("Civil engineers in Uusimaa in a year", "12tu", "2142", "MK01", [4]),
    ("Civil engineers in Uusimaa in the coming quarter", "12tu", "2142", "MK01", [1]),
])
def test_question_resolves_against_metadata(router, question, table, code, region, horizons):
    result = router.resolve(question)
    assert result["status"] == "ready", result
    assert result["context"] == {"region": region, "table_id": table, "target_code": code, "horizons": horizons}
    assert result["series_id"] in router.series


def test_followups_and_clarification_preserve_only_explicit_session_context(router):
    first = router.resolve("Civil engineers next year")
    assert first["status"] == "clarification" and "province" in first["message"]
    first = router.resolve("Uusimaa", first["context"])
    assert first["status"] == "ready"
    original = copy.deepcopy(first["context"])
    followup = router.resolve("What about six months?", first["context"])
    assert followup["series_id"] == first["series_id"] and followup["horizons"] == [2]
    assert first["context"] == original
    changed = router.resolve("What about software developers in Pirkanmaa?", followup["context"])
    assert changed["series_id"] != first["series_id"] and changed["horizons"] == [2]
    assert router.resolve("What about six months?")["status"] == "clarification"


@pytest.mark.parametrize("question", ["National vacancies next year", "Vacancies in Finland", "Civil engineers in Uusimaa for 9 months",
                                     "Civil engineers in Uusimaa for two years", "Civil engineers in Uusimaa in 2027Q1",
                                     "Civil engineers in Uusimaa next month", "Civil engineers in Uusimaa in 1 and 3 quarters"])
def test_unsupported_scope_and_horizon(router, question):
    assert router.resolve(question)["status"] == "unsupported"


@pytest.mark.parametrize("question", ["Engineers in Uusimaa", "Civil engineers in Uusimaa and Southwest Finland",
                                     "Software developers and nurses in Uusimaa", "Civil engineers and manufacturing in Uusimaa"])
def test_ambiguity_returns_choices_instead_of_picking_a_series(router, question):
    result = router.resolve(question)
    assert result["status"] == "clarification" and result["series_id"] is None
    assert len(result["choices"]) > 1


def test_unknown_new_location_never_reuses_old_series(router):
    old = router.resolve("Nurses in Pirkanmaa")
    unknown = router.resolve("Civil engineers in Atlantis next year", old["context"])
    assert unknown["status"] == "clarification" and "Atlantis".lower() in unknown["message"]
    assert "region" not in unknown["context"]
    fixed = router.resolve("Uusimaa", unknown["context"])
    assert fixed["context"]["target_code"] == "2142" and fixed["context"]["region"] == "MK01"
    assert router.resolve("Absent specialists in Uusimaa")["status"] == "unsupported"


@pytest.fixture
def service(repo, monkeypatch):
    run = {"run_id": "saved", "adapter_sha256": "pinned", "training_horizons": [1, 2, 4],
           "history_quarters": 8, "prompt_schema": "legacy_v1", "max_seq_length": 768,
           "target_tables": ["12tu", "12tw"]}
    monkeypatch.setattr("jobai.chat.selected_final_run", lambda *args: run)
    return ForecastService(repo)


def fake_predictions(inputs):
    return [{"series_id": row["series_id"], "run_id": "saved", "horizon_q": row["horizon_q"],
             "origin_quarter": row["origin_quarter"], "target_quarter": row["target_quarter"],
             "last_value": row["last_value"], "y_pred": row["last_value"] - row["horizon_q"]}
            for row in inputs]


def test_clarification_loads_no_models_or_index(service, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A clarification must not run inference")
    monkeypatch.setattr(service, "forecast", forbidden)
    monkeypatch.setattr(service, "load_retriever", forbidden)
    result = service.answer_question("Civil engineers next year")
    assert result["status"] == "clarification" and result["forecasts"] == [] and result["rag"] is None
    assert service.answer_question("National vacancies")["status"] == "unsupported"


def test_forecast_cache_reuses_all_horizons_but_invalidates_changed_history(service, monkeypatch):
    calls = []
    monkeypatch.setattr(service, "load_model", lambda: None)
    def predict(inputs):
        calls.append(inputs)
        return fake_predictions(inputs)
    monkeypatch.setattr(service, "_predict", predict)
    sid = service.resolve_question("Civil engineers in Uusimaa")["series_id"]
    first = service.forecast(sid, [4])
    assert first[0]["horizon_q"] == 4 and len(calls[0]) == 3
    first[0]["y_pred"] = 99999
    assert service.forecast(sid, [4])[0]["y_pred"] == 60
    assert len(service.forecast(sid, [1, 2])) == 2 and len(calls) == 1
    path = service.repo / "data/processed/12tu__normalized.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.timeperiod_q.eq("2024Q4"), "value"] = 80
    frame.to_csv(path, index=False)
    assert service.forecast(sid, [4])[0]["y_pred"] == 76 and len(calls) == 2


PASSAGE = {"doc_id": "a", "title": "Bulletin", "published": "2024-11-22", "url": "https://example.org/bulletin",
           "text": "Vacancies decreased in the professional occupational groups."}


def test_answer_returns_data_and_keeps_user_sessions_separate(service, monkeypatch):
    monkeypatch.setattr(service, "load_model", lambda: None)
    monkeypatch.setattr(service, "_predict", fake_predictions)
    monkeypatch.setattr(service, "retrieve_asof", lambda *args: [PASSAGE])
    captured = []
    def generate(messages, **kwargs):
        captured.append(messages)
        return json.dumps({"trend_sentence": "A decline is projected.", "context_sentence": "The bulletin reports weaker demand.",
                           "source_id": "1", "quote": PASSAGE["text"]})
    monkeypatch.setattr(service, "_generate_base_json", generate)
    one = service.answer_question("Civil engineers in Uusimaa next year")
    two = service.answer_question("Software developers in Pirkanmaa next quarter")
    followup = service.answer_question("What about six months?", one["context"])
    assert one["status"] == two["status"] == followup["status"] == "answered"
    assert followup["request"]["series_id"] == one["request"]["series_id"] != two["request"]["series_id"]
    assert [r["horizon_q"] for r in followup["forecasts"]] == [2]
    assert followup["rag"]["status"] == "verified" and followup["sources"][0]["quote"] == PASSAGE["text"]
    assert PASSAGE["text"] in captured[-1][1]["content"] and "62.00" in followup["answer"]
    json.dumps(followup)
    assert not (service.repo / "reports").exists()  # The caller, not this shared backend, owns output files.


def test_citation_failure_retry_and_missing_evidence(service, monkeypatch):
    sid = service.resolve_question("Civil engineers in Uusimaa")["series_id"]
    rows = fake_predictions(service.prepare_inputs(sid))
    original = copy.deepcopy(rows)
    monkeypatch.setattr(service, "retrieve_asof", lambda *args: [PASSAGE])
    valid = json.dumps({"source_id": 1, "quote": PASSAGE["text"], "trend_sentence": "A decline is projected."})
    responses = iter(['{"truncated":', valid])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: next(responses))
    result = service.explain("What is the outlook?", rows)
    assert result["rag"]["status"] == "verified" and len(result["rag"]["responses"]) == 2
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        {"source_id": 99, "quote": PASSAGE["text"], "context_sentence": "HALLUCINATED CONTEXT"}))
    result = service.explain("What is the outlook?", rows)
    assert result["rag"]["status"] == "invalid_citation" and result["sources"] == []
    assert "HALLUCINATED" not in result["answer"]
    monkeypatch.setattr(service, "retrieve_asof", lambda *args: [])
    assert service.explain("What is the outlook?", rows)["rag"]["status"] == "no_dated_passages"
    assert rows == original
    assert verified_quote({"source_id": True, "quote": PASSAGE["text"]}, [PASSAGE]) is None


def test_numeric_and_text_generation_use_the_right_adapter_state(service, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    class Inputs(dict):
        def to(self, device):
            return self
    class Tokenizer:
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "prompt"
        def __call__(self, prompts, **kwargs):
            return Inputs(input_ids=np.ones((len(prompts) if isinstance(prompts, list) else 1, 3)))
        def batch_decode(self, outputs, **kwargs):
            return ['{"target_scaled_change": -0.25}'] * len(outputs)
        def decode(self, *args, **kwargs):
            return '{"source_id": null}'
    class Model:
        device = "cpu"
        active = True
        def generate(self, **kwargs):
            assert self.active == (kwargs["max_new_tokens"] == 64)
            return np.ones((len(kwargs["input_ids"]), 4))
        @contextmanager
        def disable_adapter(self):
            self.active = False
            try:
                yield
            finally:
                self.active = True
    service.tokenizer, service.model = Tokenizer(), Model()
    sid = service.resolve_question("Civil engineers in Uusimaa")["series_id"]
    assert [r["y_pred"] for r in service.forecast(sid)] == [48.0, 48.0, 48.0]
    service._generate_base_json([], max_new_tokens=512)
    assert service.model.active


def test_retrieval_enforces_origin_cutoff_and_source_metadata(service):
    class Encoder:
        def encode(self, queries, **kwargs):
            assert "Civil engineers in Uusimaa" in queries[0]
            assert kwargs["normalize_embeddings"] is False
            return np.array([[1.0, 0.0]])
    class Collection:
        def query(self, **kwargs):
            assert kwargs["where"] == {"$and": [
                {"published_number": {"$gte": 20240701}}, {"published_number": {"$lte": 20241231}}]}
            return {"documents": [[PASSAGE["text"]] * 4], "metadatas": [[
                PASSAGE, {**PASSAGE, "doc_id": "future", "published": "2025-01-01"},
                {**PASSAGE, "doc_id": "old", "published": "2024-06-30"},
                {**PASSAGE, "doc_id": "no-url", "url": ""}]]}
    service.collection, service.embedding_model = Collection(), Encoder()
    sid = service.resolve_question("Civil engineers in Uusimaa")["series_id"]
    passages = service.retrieve_asof({"series_id": sid, "origin_quarter": "2024Q4"})
    assert [p["doc_id"] for p in passages] == ["a"]


def test_embedding_cache_is_persistent_and_never_downloads_by_default(service, monkeypatch):
    import sys
    (service.repo / "configs/rag.yaml").write_text((ROOT / "configs/rag.yaml").read_text())
    index = service.repo / "data/processed/rag/chroma"
    index.mkdir(parents=True)
    (index / "chroma.sqlite3").touch()
    collection = SimpleNamespace(count=lambda: 2)
    client = SimpleNamespace(get_collection=lambda *args, **kwargs: collection)
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client))
    calls = []
    class Encoder:
        runtime_available = True
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))
            assert kwargs["local_files_only"] is True
            if model == "BAAI/bge-m3" and not self.runtime_available:
                raise OSError("No cached weights")
        def save(self, path):
            Path(path, "modules.json").write_text("[]")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    service.load_retriever()
    assert calls[0][0] == "BAAI/bge-m3"
    service.load_retriever()
    assert len(calls) == 1
    restored = ForecastService(service.repo)
    restored.load_retriever()
    assert calls[-1][0] == str(service.repo / "models/embeddings/BAAI--bge-m3")
    (service.repo / "models/embeddings/BAAI--bge-m3/modules.json").unlink()
    Encoder.runtime_available = False
    with pytest.raises(FileNotFoundError, match="allow_embedding_download=True"):
        ForecastService(service.repo).load_retriever()
