# Bike Sharing Demand Analysis

A data science project that investigates whether a bike-sharing operator should expand its fleet or improve the allocation of its existing bikes. It combines exploratory data analysis, hourly demand forecasting, fleet-capacity scenarios, and a small natural-language assistant for operational questions.

## Business problem

The CEO reports two apparently conflicting problems:

- Riders sometimes cannot find an available bike.
- Bikes are also left unused during quieter periods.

The analysis tests whether these problems could be explained by demand varying substantially by hour and day type. It uses historical completed rentals to identify demand patterns and estimate bikes in use under explicit planning assumptions.

**Recommendation:** do not purchase additional bikes yet. First run a 4–6 week pilot of proactive rebalancing before commuter peaks, while collecting station-level availability and failed-search data. The available dataset cannot establish whether shortages are caused by poor distribution or an insufficient total fleet.

## Project contents

| File / component | Purpose |
|---|---|
| `bike_sharing_analysis.ipynb` | Data checks, feature engineering, exploratory analysis, hypothesis design, demand models, and fleet scenarios |
| `DECISION_LOG.md` | Records assumptions, analytical decisions, alternatives, limitations, and AI use |
| `data/hour.csv` | Hourly bike-sharing records used as the primary modelling dataset |
| `data/day.csv` | Daily records used only to reconcile hourly totals and weather aggregates | Natural-language assistant and Streamlit interface described in the project report |
| `app.py` | Streamlit application entry point |
| `nl_assistant.py` / `llm_parser.py` | Question parsing, validation, routing, and answer generation |
| `model_utils.py` | Demand model and fleet calculations |
| `tests/test_nl.py` | Automated tests for the assistant |

The notebook expects `hour.csv` and `day.csv` to be available in its working directory (or at the paths configured in the notebook). The prototype files are part of the described application structure; check that they are present in your local project before running the app.

## Dataset

The project uses the UCI Bike Sharing dataset for Capital Bikeshare, Washington, D.C., covering 2011–2012.

The hourly data contains completed rental counts and calendar and weather attributes. The analysis checks row completeness, duplicate records, date-derived fields, missing hours, unusual values, and consistency against the daily file.

Important data boundaries:

- The target is completed rentals, not total attempted demand. A rider who finds no bike is absent from the rental records.
- The data is city-wide and does not contain station locations, station inventory, fleet size, or rebalancing activity.
- Trip duration is not supplied in the dataset.
- The dataset covers two years, during which demand increased substantially.

Consequently, the analysis can describe *when* rental activity is concentrated and estimate a scenario-based lower bound for bikes in use. It cannot identify where bikes are unavailable or prove the cause of failed rentals.

## Analysis workflow

1. **Data quality:** inspect shape, nulls, duplicate timestamps, missing hours, unusual values, and consistency between hourly and daily records.
2. **Feature engineering:** derive time, weekday, working-day, holiday, month, year, rush-hour, and weather features.
3. **Exploratory analysis:** compare hourly profiles across working days and non-working days, rider types, seasons, and weather conditions.
4. **Hypothesis design:** define a field experiment to test whether proactive rebalancing reduces failed rentals during commuter peaks.
5. **Demand modelling:** compare a same-hour-previous-week baseline, Poisson regression, and gradient boosting with a Poisson loss.
6. **Fleet scenarios:** convert hourly rentals into estimated bikes in use using an assumed trip duration and selected demand percentile.
7. **Operational assistant:** parse plain-English questions, validate supported inputs, calculate model-backed answers, and refuse unsupported requests.

## Modelling approach

The modelling target is hourly total rentals (`cnt`). The data is split chronologically:

- **Training period:** January 2011 – June 2012
- **Test period:** July 2012 – December 2012

A chronological split is used so that future observations do not leak into training. `casual` and `registered` are excluded as predictors because they sum to the target. Identifier and redundant columns are also excluded.

The models compared are:

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Same hour last week | 64.3 | 110.6 | 0.75 |
| Poisson regression | 52.1 | 83.8 | 0.85 |
| Gradient boosting (Poisson loss) | **43.4** | **68.4** | **0.90** |

Errors are measured in rentals per hour. Gradient boosting is the selected model based on these test results. A validation check within the training period was also used to review the year-growth feature choice. The decision log documents that the feature choice was initially informed by test results, so the reported test performance may be somewhat optimistic.

The model still under-predicts working-day rush-hour demand by roughly 60 rentals per hour. Fleet planning therefore uses observed demand percentiles as its reference rather than relying on the forecast alone.

## Fleet estimation

Estimated bikes in use are calculated as:

`bikes in use = hourly rentals × assumed trip duration in minutes ÷ 60`

The main scenario uses a 20-minute average trip and the 90th percentile (P90) of hourly demand. These are planning assumptions, not values measured in the dataset.

For July–December 2012, the report gives:

| Measure | Estimated bikes in use |
|---|---:|
| Average hour (observed) | 85 |
| P90 hour (observed) | 192 |
| P90 hour (model forecast) | 175 |
| Busiest single hour (observed) | 326 |

The P90 observed requirement is about 2.3 times the average-hour usage. A fleet sized to that P90 level would be used about 44% in an average hour.

Trip duration materially changes the estimate. The notebook includes sensitivity scenarios for 15-, 20-, 30-, and 45-minute trips and additional maintenance and spatial buffers. These scenarios should not be interpreted as a purchase recommendation: the true demand and current fleet size are not available in the dataset.

## Hypothesis and proposed operational pilot

**H1:** On working days during 7–9 AM and 5–7 PM, proactively moving bikes to high-demand zones before the peak reduces the failed-rental rate compared with the current rebalancing process.

**H0:** Proactive rebalancing does not change the failed-rental rate.

The historical dataset cannot test this hypothesis because it contains no intervention, station locations, or failed-rental outcome. The proposed test is a 4–6 week randomized field experiment:

- Randomly assign zones (clusters of nearby stations) to proactive rebalancing or business as usual, stratifying by zone size and demand.
- Measure peak-period failed-rental rate: app opens that find no bike divided by all app opens.
- Track guardrails including rebalancing cost, truck kilometres, off-peak idle bikes, and overflow at destination stations.
- Compare treated and control zone-days while adjusting for weather and weekday, and report an effect size with a confidence interval.

The recommendation should be revisited if shortages occur across the system simultaneously even after rebalancing.

## Natural-language assistant

The prototype is intended to help an operations manager ask supported questions in plain English. Its flow is:

`Question → Parse → Validate → Model/calculation → Templated answer`

The documented question categories are:

- Bikes required for a specified scenario
- Expected rental demand
- Peak-demand hours
- Whether the company should buy more bikes

The assistant is designed to keep numerical answers grounded in the model and to show assumptions such as day type, month, weather, trip duration, and service level. Unsupported topics (for example, station-specific availability) and invalid inputs should be refused rather than answered with guesses.

The decision log describes a deterministic, rule-based parser as the default, with an optional Gemini step used only to extract structured fields. Extracted values are validated again; failures fall back to rule-based parsing. The LLM is not responsible for writing numerical answers.

## Running the notebook

1. Obtain the UCI Bike Sharing dataset and place `hour.csv` and `day.csv` in the notebook's expected data location.
2. Install the Python packages imported by the notebook. These include NumPy, pandas, Matplotlib, Seaborn, SciPy, and scikit-learn.
3. Open `bike_sharing_analysis.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the notebook from top to bottom to reproduce the analysis and figures.

The notebook defines key scenario settings near the beginning, including the test-period start date, service percentile, and assumed trip duration. Review these values before changing the analysis.

## Running the prototype

The report describes a Streamlit application under `prototype/`. If that folder is present in your checkout:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The exact dependency file and environment-variable names should be checked against the implementation in your project before running. The documented optional language-model integration uses Gemini; without a configured live API connection, the rule-based path is intended to remain available.

## Tests and verification

The project report records **17 automated tests passing**, covering parsing, input validation, refusals, the Gemini parsing path with a mocked response, fallback behaviour, peak-hour logic, fleet-planning caps, and the cautious purchase recommendation. It also records that the Streamlit page loads and displays answers and refusals.

The live Gemini API call was not verified in the build environment because Google's service was unreachable there. A real-key integration check is still required before relying on the live LLM path.

Run the test suite from the project root if the documented test file is present:

```bash
pytest test_nl.py
```

## Limitations

- Completed rentals understate demand because unsuccessful rental attempts are not recorded.
- City-wide data cannot locate station-level shortages.
- Trip duration, service percentile, maintenance share, and spatial buffer are assumptions.
- Recorded weather is used as though it were known in advance; real forecast errors are not included.
- Only two years of data are available, with strong year-on-year growth.
- The selected forecasting model under-predicts rush-hour demand.
- The proposed rebalancing hypothesis requires a field experiment and station-level data.

## Suggested data to collect next

To distinguish a distribution problem from a fleet-capacity problem, collect:

- Station-level bike availability by time
- App searches or rental attempts that find no available bike
- Trip start/end stations and trip durations
- Bike serviceability and offline/broken status
- Rebalancing movements, costs, and vehicle kilometres

These records would support a measured rebalancing pilot and a better-informed decision about whether more bikes are needed.

## Source

Fanaee-T, H., & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence. https://doi.org/10.1007/s13748-013-0040-3
