import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from nl_assistant import answer, parse_question
from model_utils import get_model

def test_rain_parsed_and_changes_forecast():
    qy = parse_question("How many bikes do we need Friday at 8am if it rains?")
    assert (qy.intent, qy.weekday, qy.hour, qy.weathersit) == ("need", 5, 8, 3)
    rain = answer("How many bikes do we need Friday at 8am if it rains?")["text"]
    clear = answer("How many bikes do we need Friday at 8am if it is clear?")["text"]
    assert rain != clear

def test_am_pm_conversion():
    assert parse_question("demand at 12am").hour == 0
    assert parse_question("demand at 12pm").hour == 12
    assert parse_question("demand at 5 pm").hour == 17

def test_temperature_not_mistaken_for_hour():
    qy = parse_question("how many bikes at 25 degrees Wednesday 5pm")
    assert qy.temp_c == 25 and qy.hour == 17

def test_invalid_hour_rejected():
    assert answer("how many bikes at 30pm")["status"] == "error"

def test_out_of_range_temperature_rejected():
    assert answer("how many bikes Monday 8am at 60 degrees")["status"] == "error"

def test_unsupported_refused_not_guessed():
    for q in ["Which station runs out first?", "What will demand be next year?", "revenue forecast"]:
        assert answer(q)["status"] == "refused"

def test_gibberish_gets_help():
    assert answer("hello")["status"] == "help"

def test_peak_hours_are_commute_hours_on_working_day():
    t = answer("When is the peak on a working day in October?")["table"]
    top = set(t.nlargest(3, "forecast_rentals").hour)
    assert top <= {7, 8, 9, 16, 17, 18, 19}

def test_planning_never_exceeds_record():
    p = get_model().day_profile(1, 9, 1)
    assert p.planning_rentals.max() <= get_model().record

def test_weekend_peak_is_midday_not_commute():
    t = answer("When is the peak on a weekend in July?")["table"]
    assert int(t.loc[t.forecast_rentals.idxmax(), "hour"]) in range(11, 17)

def test_buy_answer_is_cautious():
    txt = answer("Should we buy more bikes?")["text"].lower()
    assert "not yet" in txt and "cannot prove" in txt


# ---- Gemini path (the API call is mocked; no key or network needed) ----
import llm_parser as llm_parser

def _fake(q):
    return llm_parser.LLMQuery(intent="need", hour=8, weekday="Friday", weather="rain")

def test_llm_path_builds_same_query(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x"); monkeypatch.setattr(llm_parser, "_call_gemini", _fake)
    r = answer("will riders struggle friday morning in the wet?", use_llm=True)
    assert r["parser"] == "gemini" and r["status"] == "ok"
    assert (r["query"].weekday, r["query"].hour, r["query"].weathersit) == (5, 8, 3)

def test_llm_failure_falls_back_to_rules(monkeypatch):
    def boom(q): raise RuntimeError("network")
    monkeypatch.setenv("GEMINI_API_KEY", "x"); monkeypatch.setattr(llm_parser, "_call_gemini", boom)
    r = answer("How many bikes do we need Friday at 8am if it rains?", use_llm=True)
    assert r["status"] == "ok" and r["parser"].startswith("rules (Gemini failed")

def test_llm_bad_output_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(llm_parser, "_call_gemini", lambda q: llm_parser.LLMQuery(intent="need", hour=40))
    r = answer("How many bikes do we need Friday at 8am?", use_llm=True)
    assert r["parser"].startswith("rules (Gemini failed")          # falls back instead of trusting hour=40

def test_unsupported_never_reaches_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(llm_parser, "_call_gemini", lambda q: (_ for _ in ()).throw(AssertionError("LLM called")))
    assert answer("Which station runs out first?", use_llm=True)["status"] == "refused"

def test_no_key_uses_rules(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert answer("How many bikes Monday 8am", use_llm=True)["parser"] == "rules (no GEMINI_API_KEY)"

def test_schema_builds():
    assert "intent" in llm_parser.LLMQuery.model_json_schema()["properties"]
