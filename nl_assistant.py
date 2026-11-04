"""Natural-language layer. Pipeline: question -> parse (rules) -> validate -> call model tool -> template answer.
The answer text is filled from the tool output, so every number comes from the model, never from free text.
A future LLM parser can replace `parse_question` as long as it returns the same `Query` object."""
import re
import sys
from dataclasses import dataclass, field

import pandas as pd
from model_utils import DEFAULTS, WEATHER_NAMES, WEEKDAY_NAMES, get_model

DAYS = {n.lower(): i for i, n in enumerate(WEEKDAY_NAMES)}
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                          "august", "september", "october", "november", "december"])}
UNSUPPORTED = r"\b(station|stations|zone|zones|location|where|which area|neighbou?rhood|price|pricing|cost|revenue|profit|next year|2013|tomorrow|yesterday|next week)\b"


@dataclass
class Query:
    intent: str = "unknown"            # need | demand | peak | buy | unsupported | unknown
    hour: int | None = None
    weekday: int | None = None
    month: int | None = None
    weathersit: int | None = None
    temp_c: float | None = None
    holiday: bool = False
    notes: list = field(default_factory=list)   # assumptions made while parsing


def _parse_hour(t: str):
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        n = int(m.group(1))
        if not 1 <= n <= 12:
            return 99                      # e.g. '30pm' -> rejected by validate()
        return n % 12 + (12 if m.group(3) == "pm" else 0)
    m = re.search(r"\b(\d{1,2}):00\b", t) or re.search(r"\bat (\d{1,2})\b", t)
    if m:
        return int(m.group(1))
    for k, v in {"morning rush": 8, "morning peak": 8, "evening rush": 17, "evening peak": 17,
                 "midnight": 0, "noon": 12, "lunch": 12}.items():
        if k in t:
            return v
    return None


def parse_question(q: str, settings: dict | None = None) -> Query:
    t = q.lower().strip()
    s = {**DEFAULTS, **(settings or {})}
    qy = Query()

    if re.search(UNSUPPORTED, t):
        qy.intent = "unsupported"; return qy

    # temperature first, so '25 degrees' is not mistaken for an hour
    m = re.search(r"(-?\d{1,2}(?:\.\d)?)\s*(?:°\s*c|degrees|deg|celsius|c\b)", t)
    if m:
        qy.temp_c = float(m.group(1)); t = t.replace(m.group(0), " ")

    qy.hour = _parse_hour(t)
    for name, i in DAYS.items():
        if re.search(rf"\b{name}s?\b", t):
            qy.weekday = i; break
    if qy.weekday is None:
        if "weekend" in t:
            qy.weekday = 6; qy.notes.append("weekend = Saturday")
        elif re.search(r"working day|weekday|workday", t):
            qy.weekday = 3; qy.notes.append("working day = Wednesday")
    for name, i in MONTHS.items():
        if name in t:
            qy.month = i; break
    if re.search(r"\b(rain|rains|rainy|raining|storm|snow|snows|snowy|drizzle|thunder)\b", t): qy.weathersit = 3
    elif re.search(r"\b(mist|misty|cloud|cloudy|fog|foggy|overcast)\b", t): qy.weathersit = 2
    elif re.search(r"\b(clear|sunny|sun|fine weather)\b", t): qy.weathersit = 1
    qy.holiday = bool(re.search(r"\bholiday\b", t))

    if re.search(r"(buy|purchase|more bikes|add bikes|expand|bigger fleet|enough bikes)", t):
        qy.intent = "buy"
    elif re.search(r"\b(peak|busiest|rush|when)\b", t) and qy.hour is None:
        qy.intent = "peak"
    elif re.search(r"\b(need|needed|require|required|fleet|supply|have available|stock)\b|how many bikes", t):
        qy.intent = "need"
    elif re.search(r"\b(demand|rentals?|riders?|rent|busy|how many people|usage)\b", t):
        qy.intent = "demand"
    return qy


def validate(qy: Query):
    """Return an error message or None."""
    if qy.hour is not None and not 0 <= qy.hour <= 23:
        return f"Hour {qy.hour} is not valid. Use 0-23 (or e.g. 8am)."
    if qy.temp_c is not None:
        lo, hi = get_model().temp_range
        if not lo - 2 <= qy.temp_c <= hi + 2:
            return f"Temperature {qy.temp_c} C is outside the range the model has seen ({lo:.0f} to {hi:.0f} C)."
    return None


def _assumption_line(qy, prof, s):
    wd = WEEKDAY_NAMES[qy.weekday]
    return (f"Assumes: {wd} ({'working day' if prof.attrs['workingday'] else 'non-working day'}), "
            f"{list(MONTHS)[qy.month - 1].title()}, {WEATHER_NAMES[qy.weathersit]} weather "
            f"({prof.attrs['temp_c']:.0f} C), 2012 demand level, {s['trip_min']}-min trips, "
            f"P{int(s['pctl'] * 100)} service level.")


CAVEAT = ("This is a city-wide scenario estimate from past rentals (not true demand, no station data); "
          "it cannot say where bikes are needed.")

HELP = ("I can answer questions like:\n"
        "- How many bikes do we need on Friday at 8am if it rains?\n"
        "- What demand do we expect Saturday at 2pm in July?\n"
        "- When is the peak on a working day in October?\n"
        "- Should we buy more bikes?")


def answer(question: str, settings: dict | None = None, use_llm: bool = False) -> dict:
    """Returns {'text', 'table', 'query', 'status': ok|refused|help|error, 'parser': str}.
    use_llm=True parses with Gemini when a key is set; any failure falls back to the rule parser."""
    s = {**DEFAULTS, **(settings or {})}
    qy, parser = parse_question(question, s), "rules"
    if use_llm and qy.intent != "unsupported":      # cheap rule guard first: unsupported topics never reach the LLM
        import llm_parser as llm_parser
        if llm_parser.available():
            try:
                qy, parser = llm_parser.parse_with_llm(question), "gemini"
            except Exception as e:
                parser = f"rules (Gemini failed: {type(e).__name__})"
        else:
            parser = "rules (no GEMINI_API_KEY)"
    res = {"query": qy, "table": None, "parser": parser}

    if qy.intent == "unsupported":
        return {**res, "status": "refused", "text":
                "I can't answer that from this data. It has city-wide hourly rentals only: no stations, "
                "locations, prices or future dates, so I would be guessing.\n\n" + HELP}
    if qy.intent == "unknown":
        return {**res, "status": "help", "text": "I didn't understand that as a demand or fleet question.\n\n" + HELP}
    if (err := validate(qy)):
        return {**res, "status": "error", "text": err}

    M = get_model()
    if qy.intent == "buy":
        b = M.buy_summary(s["trip_min"], s["pctl"])
        text = (f"Not yet, based on this data. On working days the average hour has about {b['rent_5pm']:.0f} rentals at 5pm "
                f"but only about {b['rent_3am']:.0f} at 3am. In Jul-Dec 2012 the P{int(s['pctl']*100)} busy hour needed about "
                f"{b['peak_in_use']:.0f} bikes in use versus {b['mean_in_use']:.0f} on average ({b['ratio']:.1f}x), so a fleet sized "
                f"for that peak would be only about {1 - b['idle_share']:.0%} used in an average hour. That fits a timing/distribution "
                f"problem, but this data cannot prove it. Suggested next step: pilot rebalancing before commuter peaks and log failed searches.\n\n"
                f"Assumes {s['trip_min']}-min trips. {CAVEAT}")
        return {**res, "status": "ok", "text": text}

    # fill defaults and say so
    if qy.weekday is None:
        qy.weekday = 3; qy.notes.append("no day given, used a working day (Wednesday)")
    if qy.month is None:
        qy.month = s["month"]
    if qy.weathersit is None:
        qy.weathersit = 1
    prof = M.day_profile(qy.weekday, qy.month, qy.weathersit, qy.temp_c, qy.holiday, s["trip_min"], s["pctl"])
    line = _assumption_line(qy, prof, s)
    if qy.notes:
        line += " (" + "; ".join(qy.notes) + ")"

    if qy.intent == "need" and qy.hour is not None:
        r = prof.iloc[qy.hour]
        text = (f"About {r.bikes_needed:.0f} bikes need to be available around {qy.hour:02d}:00 "
                f"(forecast {r.forecast_rentals:.0f} rentals/hr, planning level {r.planning_rentals:.0f}).")
        if r.capped: text += f" The planning level is capped at the highest hour ever recorded ({M.record:.0f} rentals)."

    elif qy.intent == "need":
        top = prof.loc[prof.bikes_needed.idxmax()]
        avg = prof.bikes_in_use_avg.mean()
        text = (f"The busiest hour is {int(top.hour):02d}:00, needing about {top.bikes_needed:.0f} bikes available "
                f"(planning level {top.planning_rentals:.0f} rentals/hr). Over the whole day only about {avg:.0f} bikes are in use on average, "
                f"so a fleet sized for that hour is about {avg / top.bikes_needed:.0%} used on average.")
        if top.capped: text += f" The planning level is capped at the highest hour ever recorded ({M.record:.0f} rentals)."

    elif qy.intent == "demand" and qy.hour is not None:
        r = prof.iloc[qy.hour]
        text = (f"Forecast: about {r.forecast_rentals:.0f} rentals at {qy.hour:02d}:00, "
                f"roughly {r.bikes_in_use_avg:.0f} bikes in use on average that hour.")
    else:  # peak, or demand without an hour
        top3 = prof.nlargest(3, "forecast_rentals").sort_values("hour")
        hrs = ", ".join(f"{int(h):02d}:00 ({v:.0f})" for h, v in zip(top3.hour, top3.forecast_rentals))
        text = (f"Peak hours (forecast rentals/hr): {hrs}. The quietest hour is about "
                f"{prof.forecast_rentals.min():.0f} rentals/hr, so demand varies by roughly "
                f"{prof.forecast_rentals.max() / max(prof.forecast_rentals.min(), 1):.0f}x within the day.")
    table = prof.round(0).astype({"hour": int})
    return {**res, "status": "ok", "table": table, "text": f"{text}\n\n{line} {CAVEAT}"}


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:]) or "How many bikes do we need Friday 8am if it rains?")["text"])
