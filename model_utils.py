"""Model layer for the prototype: loads data, trains the demand model, and turns
predictions into 'bikes needed' scenarios. Mirrors bike_sharing_analysis.ipynb."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_PATH = Path(__file__).parent / "data" / "hour.csv"
TEST_START = "2012-07-01"          # train before, calibrate/test from here (same split as notebook)
FEATURES = ["hr", "weekday", "workingday", "holiday", "mnth", "yr",
            "weathersit", "temp_c", "hum_pct", "wind_kmh"]
DEFAULTS = {"trip_min": 20, "pctl": 0.90, "month": 9}   # assumptions, adjustable in the app
WEATHER_NAMES = {1: "clear", 2: "misty/cloudy", 3: "rain/snow"}
WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]  # dataset: 0 = Sunday
RUSH_HOURS = (7, 8, 9, 17, 18, 19)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    df["temp_c"], df["hum_pct"], df["wind_kmh"] = df.temp * 41, df.hum * 100, df.windspeed * 67
    df["is_rush"] = (df.hr.isin(RUSH_HOURS) & (df.workingday == 1)).astype(int)
    return df


class DemandModel:
    def __init__(self):
        self.df = load_data()
        train = self.df[self.df.dteday < TEST_START]
        self.test = self.df[self.df.dteday >= TEST_START].copy()
        self.model = HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.08, max_iter=400,
            min_samples_leaf=30, random_state=42).fit(train[FEATURES], train.cnt)
        self.test["pred"] = self.model.predict(self.test[FEATURES])
        # Safety multiplier = how much actual demand exceeded the forecast, per group (rush vs other),
        # measured on the held-out period. It also absorbs the model's known under-prediction at peaks.
        ratio = self.test.cnt / self.test.pred.clip(lower=1)
        self.ratio = {1: ratio[self.test.is_rush == 1].values, 0: ratio[self.test.is_rush == 0].values}
        self.record = float(self.df.cnt.max())      # highest hourly rentals ever recorded
        self.temp_range = (float(self.df.temp_c.min()), float(self.df.temp_c.max()))

    # ---- weather defaults from history ----
    def typical_weather(self, month: int, weathersit: int):
        d = self.df[(self.df.mnth == month) & (self.df.weathersit == weathersit)]
        if len(d) < 30:                      # rare combos (e.g. snow in July): fall back to the month overall
            d = self.df[self.df.mnth == month]
        return d.temp_c.mean(), d.hum_pct.mean(), d.wind_kmh.mean()

    # ---- main scenario function ----
    def day_profile(self, weekday: int, month: int, weathersit: int = 1, temp_c=None,
                    holiday: bool = False, trip_min=DEFAULTS["trip_min"], pctl=DEFAULTS["pctl"]) -> pd.DataFrame:
        workingday = int(weekday in (1, 2, 3, 4, 5) and not holiday)
        t, h, w = self.typical_weather(month, weathersit)
        if temp_c is not None:
            t = temp_c
        X = pd.DataFrame({"hr": range(24), "weekday": weekday, "workingday": workingday,
                          "holiday": int(holiday), "mnth": month, "yr": 1, "weathersit": weathersit,
                          "temp_c": t, "hum_pct": h, "wind_kmh": w})
        out = pd.DataFrame({"hour": range(24), "forecast_rentals": self.model.predict(X[FEATURES])})
        rush = out.hour.isin(RUSH_HOURS) & (workingday == 1)
        mult = np.where(rush, np.quantile(self.ratio[1], pctl), np.quantile(self.ratio[0], pctl))
        raw = out.forecast_rentals * mult
        out["capped"] = raw > self.record            # never plan above the record hour: the multiplier is a rough calibration
        out["planning_rentals"] = raw.clip(upper=self.record)
        out["bikes_in_use_avg"] = out.forecast_rentals * trip_min / 60
        out["bikes_needed"] = out.planning_rentals * trip_min / 60
        out.attrs.update(workingday=workingday, temp_c=t, hum_pct=h, wind_kmh=w)
        return out

    # ---- facts from observed data (for the 'should we buy bikes' question) ----
    def buy_summary(self, trip_min=DEFAULTS["trip_min"], pctl=DEFAULTS["pctl"]) -> dict:
        use = self.test.cnt * trip_min / 60
        wd = self.df[self.df.workingday == 1].groupby("hr").cnt.mean()
        return {"mean_in_use": use.mean(), "peak_in_use": float(np.quantile(use, pctl)),
                "ratio": float(np.quantile(use, pctl) / use.mean()),
                "idle_share": 1 - use.mean() / float(np.quantile(use, pctl)),
                "rent_5pm": wd[17], "rent_3am": wd[3]}


@lru_cache(maxsize=1)
def get_model() -> DemandModel:
    return DemandModel()
