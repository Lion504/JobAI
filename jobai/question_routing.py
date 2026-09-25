"""Resolve forecast questions against the selected series, without loading models."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml


TARGET_DIMENSIONS = {"12tu": "Ammattiryhmä", "12tw": "Toimiala"}
NUMBER_WORDS = dict(zip(
    "one two three four five six seven eight nine ten eleven twelve".split(), range(1, 13)))
FILLER = set("""a an the and or of for in on at to from with by about what how why which
    is are was were be will would could can do does did i we you me my please give show tell
    forecast forecasting predict prediction projected projection outlook expected expect
    vacancy job demand employment work opportunity number count trend change rise fall
    increase decrease growth decline future next coming over within during this that these
    those it them same also then compared compare than versus vs explain explanation
    reason source bulletin say context support evidence latest current previous value
    result help more detail reliable reliability uncertain uncertainty likely happen
    there available provide summarize summary tell us use looking look expected
    rising falling increasing decreasing declining please horizon data information
    region province occupation industry sector professional group total all country
    month quarter year h1 h2 h4 half ahead""".split()) | set(NUMBER_WORDS)


def normalized(text):
    text = unicodedata.normalize("NFKD", str(text).casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = re.findall(r"[a-z0-9]+", text)
    return " ".join(w[:-3] + "y" if w.endswith("ies") else
                    w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is"))
                    else w for w in words)


def clean_label(code, label):
    label = re.sub(rf"^{re.escape(code)}\s+", "", label).strip()
    return re.sub(r"^\d+\s+", "", label)


class QuestionRouter:
    """Conservative English label/code matching; unknown requests need clarification.

    Context is returned to the caller, never stored on this shared object. Pass it
    back for follow-ups; this also supports resolving a partially specified request.
    """

    def __init__(self, repo):
        self.repo = Path(repo)
        self.config = yaml.safe_load((self.repo / "configs/forecast_routing.yaml").read_text())
        self.horizons = sorted(int(h) for h in self.config["adapter_profiles"])
        self.tables = {v["preferred_table"] for v in self.config["target_scopes"].values()
                       if v["supported_by_final_adapter"]}
        catalog = pd.read_csv(self.repo / "data/processed/selected_series.csv")
        eligible = catalog.loc[catalog.selected.astype(str).str.lower().eq("true")
                               & catalog.is_forecast_target.astype(str).str.lower().eq("true")
                               & catalog.table_id.isin(self.tables)]
        self.series = {r["series_id"]: r for r in eligible.to_dict("records")}
        self.dimensions = {sid: json.loads(row["dimensions_json"]) for sid, row in self.series.items()}
        self.regions, self.targets = {}, {}
        for table in sorted(self.tables):
            path = self.repo / f"data/raw/{table}__meta.json"
            if not path.is_file():
                raise FileNotFoundError(f"Restore {path}; question routing needs the saved names and codes.")
            for var in json.loads(path.read_text())["variables"]:
                labels = {code: clean_label(code, label)
                          for code, label in zip(var["values"], var["valueTexts"])}
                if var["code"] == "Alue":
                    self.regions.update({k: v for k, v in labels.items() if k.startswith("MK") and k[2:].isdigit()})
                elif var["code"] == TARGET_DIMENSIONS[table]:
                    self.targets[table] = labels

    def label(self, series_id):
        row, dims = self.series[series_id], self.dimensions[series_id]
        target = self.targets[row["table_id"]][dims[TARGET_DIMENSIONS[row["table_id"]]]]
        return f"{target} in {self.regions.get(dims['Alue'], dims['Alue'])}"

    @staticmethod
    def _matches(text, labels, partial=False):
        tokens = set(text.split())
        full, pieces = [], []
        for code, label in labels.items():
            if code == "SSS":
                continue
            name = normalized(label)
            code_match = (len(code) > 1 or "industry" in tokens) and normalized(code) in tokens
            if code_match or f" {name} " in f" {text} ":
                words = (set(name.split()) & tokens) | ({normalized(code)} if code_match else set())
                full.append((code, len(name.split()), words))
            elif partial:
                overlap = (set(name.split()) - FILLER) & tokens
                if overlap:
                    pieces.append((code, len(overlap), overlap))
        matches = full or pieces
        if not matches:
            return [], set(), False
        if full:
            winners = [m for m in full if not any(m[2] < other[2] for other in full)]
        else:
            score = max(m[1] for m in matches)
            winners = [m for m in matches if m[1] == score]
        consumed = set().union(*(m[2] | {normalized(m[0])} for m in winners))
        return [m[0] for m in winners], consumed, bool(full)

    def _time(self, question, fallback):
        text = question.casefold()
        if re.search(r"\b(?:20\d{2}(?:\s*q[1-4])?|q[1-4])\b", text):
            return None, "Please use a horizon of 1, 2, or 4 quarters from the latest complete observation.", question
        text = re.sub(r"\bhalf\s+(?:a\s+)?year\b", "6 months", text)
        found = []
        word_pattern = "|".join(NUMBER_WORDS)
        number = rf"(?:\d+|{word_pattern}|a)"
        pattern = rf"\b({number}(?:(?:\s*[,/]\s*(?:and\s+)?|\s+and\s+){number})*)\s*[- ]?\s*(quarters?|months?|years?|q)\b"
        for match in re.finditer(pattern, text):
            unit = match.group(2)
            for item in re.findall(rf"\b{number}\b", match.group(1)):
                value = int(item) if item.isdigit() else 1 if item == "a" else NUMBER_WORDS[item]
                found.append(value / 3 if unit.startswith("month") else value * 4 if unit.startswith("year") else value)
        found.extend(int(h) for h in re.findall(r"\bh(\d+)\b", text))
        remaining = re.sub(pattern, " ", text)
        remaining = re.sub(r"\bh\d+\b", " ", remaining)
        if re.search(r"\b(?:next|coming)\s+quarter\b", text):
            found.append(1)
        if re.search(r"\b(?:next|coming)\s+year\b", text):
            found.append(4)
        remaining = re.sub(r"\b(?:next|coming)\s+(?:quarter|year)\b", " ", remaining)
        horizon_message = "Choose 1, 2, or 4 quarters (3, 6, or 12 months) from the latest complete observation."
        if re.search(r"\b(?:next\s+month|weeks?|weekly|daily|days?|decades?)\b", text):
            return None, horizon_message, question
        values = found or list(fallback)
        if not values or any(isinstance(h, bool) or h not in self.horizons for h in values):
            return None, horizon_message, question
        return sorted({int(h) for h in values}), None, remaining

    def resolve(self, question, context=None, *, series_id=None, horizons=None):
        """Return ready/clarification/unsupported, a validated request, and new context."""
        question = question.strip()
        state = dict(context or {})
        choices = []

        def result(status, message, sid=None):
            return {"status": status, "message": message, "series_id": sid,
                    "horizons": list(state.get("horizons", self.horizons)),
                    "context": dict(state), "choices": choices}

        if not question:
            return result("clarification", "Enter a vacancy forecast question.")
        explicit_ids = re.findall(r"12[a-z0-9]+::[^\s?]+", question)
        if len(explicit_ids) > 1:
            return result("clarification", "Choose one series at a time.")
        if explicit_ids:
            series_id = explicit_ids[0].rstrip(".,")
            question_for_matching = question.replace(explicit_ids[0], "")
        else:
            question_for_matching = question
        if series_id is not None:
            if series_id not in self.series:
                return result("unsupported", "That series is not a selected 12tu/12tw forecast target.")
            row, dims = self.series[series_id], self.dimensions[series_id]
            state.update(region=dims["Alue"], table_id=row["table_id"],
                         target_code=dims[TARGET_DIMENSIONS[row["table_id"]]])
        hs, error, without_time = self._time(question_for_matching, horizons if horizons is not None else state.get("horizons", self.horizons))
        if error:
            return result("unsupported", error)
        state["horizons"] = hs
        text = normalized(without_time)
        regions, region_words, _ = self._matches(text, self.regions)
        if re.search(r"\b(national|nationwide|whole country|all finland)\b", text) or (
                "finland" in text.split() and not regions):
            state.pop("region", None)
            return result("unsupported", "The saved model supports province-by-occupation or province-by-industry vacancies. It cannot provide a national forecast.")
        if len(regions) > 1:
            state.pop("region", None)
            choices.extend({"region": c, "label": self.regions[c]} for c in regions)
            return result("clarification", "Which single province should I forecast?")
        if regions:
            state["region"] = regions[0]
        target_text = " ".join(t for t in text.split() if t not in region_words)
        candidates = []
        for table, labels in self.targets.items():
            codes, consumed, exact = self._matches(target_text, labels, partial=True)
            candidates.extend((table, code,
                               (set(normalized(labels[code]).split()) | {normalized(code)}) & set(target_text.split()),
                               exact) for code in codes)
        if any(c[3] for c in candidates):
            candidates = [c for c in candidates if c[3]]
            candidates = [c for c in candidates if not any(c[2] < other[2] for other in candidates)]
        if "occupation" in text.split() and "industry" not in text.split():
            candidates = [c for c in candidates if c[0] == "12tu"]
        elif "industry" in text.split() and "occupation" not in text.split():
            candidates = [c for c in candidates if c[0] == "12tw"]
        if re.search(r"\b(?:all|total) occupation\b", text):
            candidates = [("12tu", "SSS", {"all", "total", "occupation"}, True)]
        elif re.search(r"\b(?:all|total) industr(?:y|ies)\b", text):
            candidates = [("12tw", "SSS", {"all", "total", "industry"}, True)]
        target_words = set().union(*(c[2] for c in candidates)) if candidates else set()
        unknown = set(text.split()) - FILLER - region_words - target_words
        if unknown:
            # Do not silently reuse an earlier occupation or province for an unrecognized new request.
            state.clear()
            state["horizons"] = hs
            if len(regions) == 1:
                state["region"] = regions[0]
            if len(candidates) == 1:
                state.update(table_id=candidates[0][0], target_code=candidates[0][1])
            return result("clarification", "I could not match " + ", ".join(sorted(unknown)) +
                          ". Please use a province and occupation/industry name from the saved catalog, or select a series ID.")
        if len(candidates) > 1:
            state.pop("table_id", None)
            state.pop("target_code", None)
            choices.extend({"table_id": t, "target_code": c, "label": self.targets[t][c]}
                           for t, c, _, _ in candidates)
            return result("clarification", "Which occupation or industry do you mean? Choose one of the matches.")
        if candidates:
            state.update(table_id=candidates[0][0], target_code=candidates[0][1])
        if state.get("region") not in self.regions:
            state.pop("region", None)
            return result("clarification", "Which province, for example Uusimaa or Pirkanmaa?")
        if state.get("table_id") not in self.targets or state.get("target_code") not in self.targets[state["table_id"]]:
            return result("clarification", "Which occupation or industry should I forecast?")
        table, code, region = state["table_id"], state["target_code"], state["region"]
        matches = [sid for sid, row in self.series.items() if row["table_id"] == table
                   and self.dimensions[sid].get("Alue") == region
                   and self.dimensions[sid].get(TARGET_DIMENSIONS[table]) == code]
        if not matches:
            return result("unsupported", f"No selected forecast series is available for {self.targets[table][code]} in {self.regions[region]}.")
        if len(matches) > 1:
            choices.extend({"series_id": sid, "label": sid} for sid in matches)
            return result("clarification", "Choose a specific series; this request matches more than one target.")
        return result("ready", self.label(matches[0]), matches[0])
