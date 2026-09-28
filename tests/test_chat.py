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
    targets = {"12tu": {"SSS": "All occupations", "2142": "2142 Civil engineers", "2512": "2512 Software developers",
                        "2511": "2511 Systems analysts",
                        "2141": "2141 Industrial engineers", "3221": "3221 Nurses",
                        "1321": "1321 Manufacturing managers", "4227": "4227 Survey and market research interviewers",
                        "9999": "9999 Absent specialists"},
               "12tw": {"SSS": "All industries", "C": "C Manufacturing", "F": "F Construction",
                        "M73": "M73 Advertising and market research"}}
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


def test_job_market_is_not_treated_as_an_industry_and_numbered_choices_work(router):
    general = router.resolve("What is the trend of the job market next quarter in Uusimaa?")
    assert general["status"] == "clarification" and not general["choices"]
    assert general["context"]["region"] == "MK01" and general["horizons"] == [1]
    menu = router.resolve("Engineers in Uusimaa")
    assert len(menu["choices"]) == 2
    chosen = router.resolve("2", menu["context"])
    assert chosen["status"] == "ready" and chosen["series_id"] in router.series
    assert chosen["context"]["target_code"] == menu["choices"][1]["target_code"]
    assert router.resolve("Helsinki next quarter")["status"] == "unsupported"


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


def forecast_action(region="MK01", targets=None, horizons=None, needs_choice=False):
    return {"action": "forecast", "forecast": {
        "region": region, "targets": ["12tu:2142"] if targets is None else targets,
        "horizons": [4] if horizons is None else horizons, "needs_choice": needs_choice}}


def fake_predictions(inputs):
    return [{"series_id": row["series_id"], "run_id": "saved", "horizon_q": row["horizon_q"],
             "origin_quarter": row["origin_quarter"], "target_quarter": row["target_quarter"],
             "last_value": row["last_value"], "y_pred": row["last_value"] - row["horizon_q"]}
            for row in inputs]


def test_clarification_runs_no_numeric_inference_or_retrieval(service, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A clarification must not run inference")
    monkeypatch.setattr(service, "forecast", forbidden)
    monkeypatch.setattr(service, "load_retriever", forbidden)
    actions = iter([forecast_action(region=None), forecast_action(region="Finland")])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(next(actions)))
    result = service.answer_question("Civil engineers next year")
    assert result["status"] == "clarification" and result["forecasts"] == [] and result["rag"] is None
    assert service.answer_question("National vacancies")["status"] == "unsupported"


def test_base_model_leads_chat_and_receives_recent_history(service, monkeypatch):
    monkeypatch.setattr(service, "_decide_action", ForecastService._decide_action.__get__(service))
    captured = []
    def generate(messages, **kwargs):
        captured.append(messages)
        return '{"action": "reply", "answer": "I can help with vacancy forecasts and ordinary questions."}'
    monkeypatch.setattr(service, "_generate_base_json", generate)
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("A chat reply needs no forecast"))
    result = service.answer_question("what you know", history=[
        {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}])
    assert result["status"] == "answered" and result["answer"].startswith("I can help")
    assert [message["role"] for message in captured[0][-3:]] == ["user", "assistant", "user"]
    assert captured[0][-1]["content"] == "what you know"


def test_helsinki_general_outlook_then_yes_forecasts_the_total(service, monkeypatch):
    actions = iter([forecast_action(region="Helsinki", targets=["12tu:SSS"]),
                    forecast_action(targets=["12tu:SSS"])])
    prompts, calls = [], []
    def generate(messages, **kwargs):
        prompts.append(copy.deepcopy(messages))
        return json.dumps(next(actions))
    def forecast(sid, horizons):
        calls.append((sid, horizons))
        return fake_predictions(service.prepare_inputs(sid, horizons))
    monkeypatch.setattr(service, "_generate_base_json", generate)
    monkeypatch.setattr(service, "forecast", forecast)
    monkeypatch.setattr(service, "resolve_question", lambda *args, **kwargs:
                        pytest.fail("The chat must not reparse the user's wording"))
    monkeypatch.setattr(service, "explain", lambda *args: {"answer": "The total vacancy forecast.", "sources": [], "rag": None})
    question = "what is the trend of job market in helsinki for next year"
    first = service.answer_question(question)
    assert first["status"] == "unsupported" and not calls
    assert first["context"]["suggested_request"] == forecast_action(targets=["12tu:SSS"])["forecast"]
    second = service.answer_question("yes", first["context"], history=[
        {"role": "user", "content": question}, {"role": "assistant", "content": first["answer"]}])
    assert second["status"] == "answered" and len(calls) == 1
    assert calls[0][1] == [4] and "Ammattiryhmä=SSS" in calls[0][0]
    assert second["request"]["choices"] == []
    assert second["forecasts"][0]["target_quarter"] == "2025Q4"
    assert "12tu:SSS: All occupations" in prompts[0][0]["content"]
    assert "suggested_request" in prompts[1][0]["content"]


def test_explicit_helsinki_cannot_be_silently_mapped_to_uusimaa(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=["12tu:SSS"])))
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("The province substitution needs a choice"))
    result = service.answer_question("what is the job market trend for next year in helsinki")
    assert result["status"] == "unsupported" and not result["forecasts"]
    assert result["context"]["forecast_request"]["region"] == "Helsinki"
    assert result["context"]["suggested_request"]["region"] == "MK01"


def test_completed_forecast_is_passed_to_followups_even_without_history(service, monkeypatch):
    captured = []
    responses = iter([json.dumps(forecast_action(targets=["12tu:SSS"])),
                      json.dumps({"action": "reply", "answer": "The forecast shown was for Uusimaa."}),
                      json.dumps(forecast_action(region="Atlantis"))])
    def generate(messages, **kwargs):
        captured.append(copy.deepcopy(messages))
        return next(responses)
    monkeypatch.setattr(service, "_generate_base_json", generate)
    monkeypatch.setattr(service, "forecast", lambda sid, hs: fake_predictions(service.prepare_inputs(sid, hs)))
    monkeypatch.setattr(service, "explain", lambda *args: {"answer": "60 vacancies in 2025Q4, down from 64 in 2024Q4.",
                                                          "sources": [], "rag": None})
    first = service.answer_question("show the job market next year in Uusimaa")
    saved = copy.deepcopy(first["context"]["last_forecast"])
    assert saved["status"] == "completed" and saved["scope"] == "All occupations in Uusimaa"
    assert saved["values"][0]["y_pred"] == 60
    # Caller-side edits to displayed rows must not corrupt the session's saved result.
    first["forecasts"][0]["y_pred"] = 999
    followup = service.answer_question("what forecast did you just show me", first["context"])
    assert followup["context"]["last_forecast"] == saved
    assert json.dumps(saved, ensure_ascii=False) in captured[1][0]["content"]
    unsupported = service.answer_question("what about Atlantis", followup["context"])
    assert unsupported["context"]["last_forecast"] == saved
    assert unsupported["status"] == "unsupported"


def test_conversational_forecasts_use_arguments_and_can_reset_to_total(service, monkeypatch):
    actions = iter([forecast_action(targets=["12tu:2512"], horizons=[2]),
                    forecast_action(targets=["12tu:SSS"], horizons=[2])])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(next(actions)))
    monkeypatch.setattr(service, "forecast", lambda sid, hs: fake_predictions(service.prepare_inputs(sid, hs)))
    monkeypatch.setattr(service, "explain", lambda *args: {"answer": "Forecast", "sources": [], "rag": None})
    first = service.answer_question("Could you check how things might look for people writing software around Uusimaa six months from now?")
    assert first["status"] == "answered" and first["context"]["target_code"] == "2512"
    second = service.answer_question("How about the whole job market there, over the same period?", first["context"])
    assert second["status"] == "answered" and second["context"]["target_code"] == "SSS"
    assert second["context"]["horizons"] == [2]


def test_named_choice_uses_saved_context_without_requiring_exact_phrase(service, monkeypatch):
    actions = iter([forecast_action(targets=["12tu:3221"], needs_choice=True),
                    forecast_action(targets=["12tu:3221"])])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(next(actions)))
    calls = []
    def forecast(sid, hs):
        calls.append(sid)
        return fake_predictions(service.prepare_inputs(sid, hs))
    monkeypatch.setattr(service, "forecast", forecast)
    monkeypatch.setattr(service, "explain", lambda *args: {"answer": "Forecast", "sources": [], "rag": None})
    first = service.answer_question("Healthcare work around Uusimaa a year ahead?")
    assert first["status"] == "clarification" and not calls
    second = service.answer_question("The nursing one would be useful, thanks", first["context"])
    assert second["status"] == "answered" and len(calls) == 1
    assert second["context"]["target_code"] == "3221"


def test_choice_is_remembered_while_waiting_for_the_province(service, monkeypatch):
    actions = iter([forecast_action(region=None, targets=["12tu:2512", "12tu:2511"], needs_choice=True),
                    forecast_action(targets=["12tu:2511"])])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(next(actions)))
    monkeypatch.setattr(service, "forecast", lambda sid, hs: fake_predictions(service.prepare_inputs(sid, hs)))
    monkeypatch.setattr(service, "explain", lambda *args: {"answer": "Forecast", "sources": [], "rag": None})
    first = service.answer_question("IT jobs next year?")
    before = copy.deepcopy(first["context"])
    second = service.answer_question("2", first["context"])
    assert first["context"] == before
    assert second["status"] == "clarification" and "province" in second["answer"]
    assert second["context"]["forecast_request"]["targets"] == ["12tu:2511"]
    assert second["context"]["forecast_request"]["needs_choice"] is False
    third = service.answer_question("Uusimaa, please", second["context"])
    assert third["status"] == "answered" and third["context"]["target_code"] == "2511"


def test_choices_exclude_series_missing_in_the_requested_province(service, monkeypatch):
    service.router.series = {sid: row for sid, row in service.router.series.items()
                             if not (service.router.dimensions[sid]["Alue"] == "MK01"
                                     and service.router.dimensions[sid].get("Ammattiryhmä") == "2512")}
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=["12tu:2512", "12tu:2511"], needs_choice=True)))
    result = service.answer_question("IT jobs in Uusimaa next year")
    assert result["status"] == "clarification"
    assert [choice["target_code"] for choice in result["request"]["choices"]] == ["2511"]


@pytest.mark.parametrize("action", [
    forecast_action(region="Atlantis"), forecast_action(targets=["12tu:9999"]),
    forecast_action(horizons=[3]), forecast_action(horizons=[True]), forecast_action(targets=[]),
])
def test_tool_arguments_cannot_bypass_catalog_or_horizon_checks(service, monkeypatch, action):
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(action))
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("Unsupported arguments must not run the adapter"))
    old = service.resolve_question("Civil engineers in Uusimaa")["context"]
    result = service.answer_question("Check this new request", old)
    assert result["status"] in {"clarification", "unsupported"} and not result["forecasts"]
    assert "target_code" not in result["context"]


def test_incomplete_tool_request_is_retried_before_inference(service, monkeypatch):
    responses = iter([json.dumps({"action": "forecast"}), json.dumps(forecast_action(region=None))])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("The province still needs clarification"))
    result = service.answer_question("Civil engineer vacancies next year")
    assert result["status"] == "clarification" and "province" in result["answer"]
    assert result["context"]["target_code"] == "2142"


def test_model_selected_catalog_uses_only_saved_series(service, monkeypatch):
    monkeypatch.setattr(service, "_decide_action", lambda *args: {"action": "catalog", "kind": "occupation"})
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("A catalog question needs no forecast"))
    result = service.answer_question("what occupation you know")
    assert result["status"] == "answered" and not result["forecasts"]
    assert "Software developers" in result["answer"] and "Absent specialists" not in result["answer"]
    scoped = service.answer_question("list all occupations in Uusimaa")
    assert scoped["context"]["region"] == "MK01" and "Software developers" in scoped["answer"]


def test_pending_choice_cannot_be_answered_as_unrelated_chat(service, monkeypatch):
    menu = service.resolve_question("Engineers in Uusimaa")
    monkeypatch.setattr(service, "_decide_action", lambda *args: {"action": "reply", "answer": "Hello"})
    monkeypatch.setattr(service, "forecast", lambda sid, horizons: fake_predictions(
        service.prepare_inputs(sid, horizons)))
    seen = []
    def explain(question, forecasts):
        seen.append(question)
        return {"answer": "A verified forecast response", "sources": [], "rag": None}
    monkeypatch.setattr(service, "explain", explain)
    result = service.answer_question("1", menu["context"], history=[
        {"role": "user", "content": "Engineers in Uusimaa"}])
    assert result["status"] == "answered" and result["forecasts"]
    assert result["request"]["series_id"] in service.router.series
    assert seen == ["Engineers in Uusimaa"]
    assert result["context"]["forecast_request"]["targets"] == [
        f"{result['context']['table_id']}:{result['context']['target_code']}"]


def test_natural_language_target_choices_are_catalog_checked(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=["12tu:2512", "12tu:2511"], needs_choice=True)))
    question = "I want to know the next year IT jobs trend in Uusimaa area"
    route = service.answer_question(question)["request"]
    assert route["status"] == "clarification" and route["horizons"] == [4]
    assert route["context"]["region"] == "MK01" and len(route["choices"]) == 2
    assert [choice["target_code"] for choice in route["choices"]] == ["2512", "2511"]
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("Choose a target before forecasting"))
    assert service.answer_question(question)["status"] == "clarification"
    selected = service.resolve_question("1", route["context"])
    assert selected["status"] == "ready" and selected["horizons"] == [4]
    assert selected["context"]["target_code"] == "2512"
    previous = service.router.resolve("Civil engineers in Uusimaa")
    changed = service.answer_question(question, previous["context"])["request"]
    assert changed["status"] == "clarification" and len(changed["choices"]) == 2
    assert "target_code" not in changed["context"]


def test_helsinki_suggestion_keeps_the_requested_target_and_horizon(service, monkeypatch):
    actions = iter([
        forecast_action(region="Helsinki", targets=["12tu:2512", "12tu:2511"], needs_choice=True),
        forecast_action(targets=["12tu:2512", "12tu:2511"], needs_choice=True)])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(next(actions)))
    first = service.answer_question("what is next year it job trend in helsinki")["request"]
    assert first["status"] == "unsupported" and first["horizons"] == [4]
    assert "Uusimaa" in first["message"] and "next quarter" not in first["message"]
    second = service.answer_question("ok, as your suggested check for me", first["context"])["request"]
    assert second["status"] == "clarification" and second["horizons"] == [4]
    assert second["context"]["region"] == "MK01" and len(second["choices"]) == 2
    chosen = service.resolve_question("1", second["context"])
    assert chosen["status"] == "ready" and chosen["horizons"] == [4]


def test_natural_language_target_must_be_selected_and_unambiguous(service, monkeypatch):
    question = "Tech jobs in Uusimaa next year"
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=["12tu:9999"])))
    assert service.answer_question(question)["status"] == "clarification"
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=["12tu:2512"], needs_choice=True)))
    route = service.answer_question(question)["request"]
    assert route["status"] == "clarification" and route["choices"][0]["target_code"] == "2512"
    selected = service.resolve_question("1", route["context"])
    assert selected["status"] == "ready" and selected["context"]["target_code"] == "2512"


def test_missing_latest_quarter_returns_data_limit_before_model_load(service, monkeypatch):
    path = service.repo / "data/processed/12tu__normalized.csv"
    frame = pd.read_csv(path)
    missing = (frame.Alue.eq("MK01") & frame["Ammattiryhmä"].astype(str).eq("2142")
               & frame.timeperiod_q.eq("2024Q4"))
    assert missing.sum() == 1
    frame.loc[missing, "value"] = np.nan
    frame.to_csv(path, index=False)
    monkeypatch.setattr(service, "load_model", lambda: pytest.fail("Missing data must not load the GPU model"))
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(horizons=[1])))
    result = service.answer_question("Civil engineers in Uusimaa next quarter")
    assert result["status"] == "unsupported" and "2024Q4" in result["answer"]
    assert result["forecasts"] == []


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


def test_model_selected_bulletin_tool_checks_citation(service, monkeypatch):
    monkeypatch.setattr(service, "_decide_action", lambda *args: {"action": "bulletins"})
    monkeypatch.setattr(service, "retrieve_bulletins", lambda *args: [PASSAGE])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        {"answer": "The bulletin reports fewer vacancies.",
         "citations": [{"source_id": 1, "quote": PASSAGE["text"]}]}))
    result = service.answer_question("What do the bulletins say about vacancies?")
    assert result["status"] == "answered" and result["rag"]["status"] == "verified"
    assert result["sources"][0]["doc_id"] == "a" and PASSAGE["text"] in result["answer"]
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        {"answer": "An unsupported claim", "citations": [{"source_id": 1, "quote": "Invented quote"}]}))
    invalid = service.answer_question("What do the bulletins say about vacancies?")
    assert invalid["rag"]["status"] == "invalid_citation" and "unsupported claim" not in invalid["answer"]


def test_answer_returns_data_and_keeps_user_sessions_separate(service, monkeypatch):
    monkeypatch.setattr(service, "load_model", lambda: None)
    monkeypatch.setattr(service, "_predict", fake_predictions)
    monkeypatch.setattr(service, "retrieve_asof", lambda *args: [PASSAGE])
    actions = iter([forecast_action(), forecast_action(region="MK06", targets=["12tu:2512"], horizons=[1]),
                    forecast_action(horizons=[2])])
    monkeypatch.setattr(service, "_decide_action", lambda *args: next(actions))
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
    assert "The saved model forecasts" in one["answer"] and "down 4.00" in one["answer"]
    assert one["context"]["last_forecast"]["scope"] == "Civil engineers in Uusimaa"
    assert two["context"]["last_forecast"]["scope"] == "Software developers in Pirkanmaa"
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


def test_unemployment_quote_cannot_be_cited_as_vacancy_context(service, monkeypatch):
    unrelated = {**PASSAGE, "doc_id": "unemployment",
                 "text": "The number of unemployed job seekers increased in Uusimaa compared with last year."}
    sid = service.resolve_question("Civil engineers in Uusimaa")["series_id"]
    rows = fake_predictions(service.prepare_inputs(sid))[:1]
    monkeypatch.setattr(service, "retrieve_asof", lambda *args: [unrelated, PASSAGE])
    responses = iter([
        json.dumps({"source_id": 1, "quote": unrelated["text"], "trend_sentence": "The forecast falls."}),
        json.dumps({"source_id": 2, "quote": PASSAGE["text"], "trend_sentence": "The forecast falls."}),
    ])
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: next(responses))
    result = service.explain("What is the outlook?", rows)
    assert result["rag"]["status"] == "verified" and len(result["rag"]["responses"]) == 2
    assert result["sources"][0]["doc_id"] == "a"
    assert "unemployed job seekers" not in result["answer"]
    assert "The saved model forecasts" in result["answer"]


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


def test_bulletin_retrieval_rejects_future_and_unlinked_passages(service):
    today = pd.Timestamp.now(tz="UTC").date()
    tomorrow = (pd.Timestamp(today) + pd.Timedelta(days=1)).date().isoformat()
    class Encoder:
        def encode(self, queries, **kwargs):
            return np.array([[1.0, 0.0]])
    class Collection:
        def query(self, **kwargs):
            assert kwargs["where"] == {"published_number": {"$lte": int(today.strftime("%Y%m%d"))}}
            return {"documents": [[PASSAGE["text"]] * 4], "metadatas": [[
                PASSAGE, {**PASSAGE, "doc_id": "future", "published": tomorrow},
                {**PASSAGE, "doc_id": "no-url", "url": ""}, PASSAGE]]}
    service.collection, service.embedding_model = Collection(), Encoder()
    assert [passage["doc_id"] for passage in service.retrieve_bulletins("vacancies")] == ["a"]


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


@pytest.mark.parametrize("target, question, expected", [
    ("3221", "What is the vacancy outlook for nurses in Uusimaa?", "Ammattiryhmä=3221"),
    ("Nurses", "What is the vacancy outlook for nurses in Uusimaa?", "Ammattiryhmä=3221"),
    ("12tu:3221: Nurses", "What is the vacancy outlook for nurses in Uusimaa?", "Ammattiryhmä=3221"),
    ("Software developers", "How are software development jobs looking in Uusimaa?", "Ammattiryhmä=2512"),
    ("F", "What is the outlook for the construction industry in Uusimaa?", "Toimiala=F"),
    ("12tw:F: Construction", "What is the outlook for the construction industry in Uusimaa?", "Toimiala=F"),
])
def test_unambiguous_target_codes_and_labels_still_forecast(service, monkeypatch, target, question, expected):
    monkeypatch.setattr(service, "forecast", lambda sid, hs: fake_predictions(service.prepare_inputs(sid, hs)))
    monkeypatch.setattr(service, "explain", lambda *args: {"answer": "Forecast generated.", "sources": [], "rag": None})
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=[target])))
    result = service.answer_question(question)
    assert result["status"] == "answered"
    assert expected in result["request"]["series_id"]


@pytest.mark.parametrize("question", [
    "Explain why the forecast for nurses in Uusimaa is uncertain",
    "Forecast nurses in Uusimaa in 9 months",
])
def test_reply_is_not_overridden_by_forecast_keywords(service, monkeypatch, question):
    answer = "I can explain the forecast and its supported horizons."
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        {"action": "reply", "answer": answer}))
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("A reply must not trigger a forecast"))
    result = service.answer_question(question)
    assert result["answer"] == answer and result["forecasts"] == []


def test_mentioning_series_id_in_explanation_does_not_run_forecast(service, monkeypatch):
    sid = next(iter(service.router.series))
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs:
                        '{"action":"reply","answer":"That is the saved forecast series."}')
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("A mentioned ID is not a tool request"))
    result = service.answer_question(f"Explain the forecast for {sid}")
    assert result["answer"] == "That is the saved forecast series." and result["forecasts"] == []


@pytest.mark.parametrize("targets, question", [
    (["12tu:1321"], "What is the outlook for managers in Uusimaa next year?"),
    (["12tu:2142", "12tu:1321"], "What is the outlook for engineers in Uusimaa next year?"),
])
def test_target_choices_are_not_narrowed_by_question_words(service, monkeypatch, targets, question):
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=targets, needs_choice=True)))
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("The user must choose a target"))
    result = service.answer_question(question)
    assert result["status"] == "clarification" and result["forecasts"] == []
    assert [f"{c['table_id']}:{c['target_code']}" for c in result["request"]["choices"]] == targets


@pytest.mark.parametrize("targets, question", [
    (["construction workers"], "What is the trend for construction workers in Uusimaa?"),
    (["software development"], "How are software development jobs looking in Uusimaa?"),
    (["12tu:9999"], "What is the vacancy outlook for nurses in Uusimaa?"),
    (["12tu:3221", "12tu:9999"], "What is the vacancy outlook for nurses in Uusimaa?"),
    (["12tu:3221: Software developers"], "What is the vacancy outlook for nurses in Uusimaa?"),
    (["12tu:9999: Nurses"], "What is the vacancy outlook for nurses in Uusimaa?"),
    ([], "What is the vacancy outlook for IT in Uusimaa?"),
    (["SSS"], "What is the vacancy outlook in Uusimaa?"),
])
def test_unresolved_targets_are_not_replaced_with_other_series(service, monkeypatch, targets, question):
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=targets)))
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("An unresolved target must not run inference"))
    result = service.answer_question(question)
    assert result["status"] == "clarification" and result["forecasts"] == []
    assert result["request"]["choices"] == []


def test_unsupported_forecast_horizon_is_not_replaced_with_defaults(service, monkeypatch):
    monkeypatch.setattr(service, "_generate_base_json", lambda *args, **kwargs: json.dumps(
        forecast_action(targets=["3221"], horizons=[3])))
    monkeypatch.setattr(service, "forecast", lambda *args: pytest.fail("Nine months is unsupported"))
    result = service.answer_question("Forecast nurses in Uusimaa in 9 months")
    assert result["status"] == "unsupported" and result["forecasts"] == []
