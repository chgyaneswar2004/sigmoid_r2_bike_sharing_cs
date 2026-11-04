# Assumptions, Decisions and AI Use

## Hypothesis I chose (testable)

**H1:** On working days during 7-9am and 5-7pm, proactively moving bikes to high-demand zones before the peak **reduces the failed-rental rate** compared with the current rebalancing process. **H0:** no difference.

**How I would test it:** a randomized field experiment. Randomly assign zones (clusters of nearby stations, stratified by size and demand) to proactive rebalancing or business as usual for 4-6 weeks. Primary metric: failed-rental rate at peak (app opens that find no bike / app opens). Guardrails: rebalancing cost, truck-km, idle bikes off-peak, overflow at drop zones. Analyse treated vs control zone-days, adjusting for weather and weekday, and report the effect with a confidence interval. Sample size comes from a business-chosen minimum effect (e.g. a 20% relative cut in failed rentals) and the measured baseline failed rate.

**Why I chose it:** the CEO's two complaints ("can't find a bike" and "bikes sit unused") can both be true if the problem is timing and location, not fleet size. The data supports the motivation (commuter-hour peaks, mostly registered riders, near-empty off-peak hours) but cannot run the test: it has no intervention, no failed-rental outcome and no station location. Full design: notebook section 4.

**What would change my mind:** station data showing shortages everywhere at once, which would mean the fleet really is too small.

---

One file for everything: what I assumed, what I decided and why, how AI was used, and what is still limited.
Numbers below come from `bike_sharing_analysis.ipynb` (re-run it to reproduce).

## 1. Assumptions (not facts in the data)

| # | Assumption | Value | Why / effect |
|---|---|---|---|
| A1 | Average trip length | **20 min** | Not in the data. Bikes in use = rentals/hr x trip minutes / 60. Fleet need scales almost linearly with it (see sensitivity table) |
| A2 | Service level | **P90** of hourly demand | A business choice: "enough bikes in 90% of hours". P80 to P99 shown in the sensitivity table |
| A3 | Rentals = demand | Treated as a floor | Riders who found no bike are never recorded, so true demand is higher |
| A4 | Missing hours = zero rentals | Zero-filled for averages | Supported by `day.csv` totals matching the hourly sums exactly |
| A5 | Weather known in advance | Recorded weather used as input | Real forecast error is not modelled, so real errors would be larger |
| A6 | Maintenance and spatial buffers | 5-15% and 1.0-2.0x | Invented scenarios, shown as a table, not claimed as measured |

## 2. Decisions

| # | Decision | Alternative | Reason |
|---|---|---|---|
| D1 | Time-based split: train to 2012-06-30, test Jul-Dec 2012 | Random split | A random split leaks the future; this test also covers summer, fall and winter |
| D2 | `casual` / `registered` not used as inputs | Use them | They add up to the target (leakage) |
| D3 | Keep outliers | Remove them | High rush-hour values are real busy hours and drive fleet need |
| D4 | Keep missing hours, flag short days, no imputation | Impute / drop | Missing rows are mostly 2-5am with ~0 rentals; some short days are real events (Hurricane Sandy, 29-30 Oct 2012) |
| D5 | Poisson-type models; main model is gradient boosting | Plain linear regression | Counts, non-negative, variance grows with the mean |
| D6 | Boosting uses `yr` (0/1), not a day counter | `t_days` | A day counter made rush-hour under-prediction much worse (trees cannot extrapolate) |
| D7 | Do not tune further | More features / weights | Keep scope explainable; limits are documented instead |

## 3. Inputs added after review, with context and reason

**I1. Trip-duration formula (correction).**
- *Context:* an earlier version of my reasoning treated "rentals per hour" as roughly "bikes in use", which is only true if every trip lasts an hour.
- *Decision:* concurrent bikes ~ rentals/hr x trip minutes / 60, default 20 min.
- *Reason:* the 1-hour version overstates fleet need about 3x versus 20 min. With 20 min the P90 hour needs about 192 bikes in use (observed, test period); with 60 min it would be about 577. Trip length is the biggest assumption, so it is a visible variable plus a sensitivity table (15/20/30/45 min).

**I2. Use `day.csv` beyond a single check.**
- *Context:* `day.csv` is `hour.csv` aggregated, so it adds no new features.
- *Decision:* (a) reconcile totals and weather (all 731 days match exactly), and (b) score the hourly model against it at daily level. It is never a model input.
- *Reason:* (a) proves the zero-fill assumption A4; (b) gives an independent view for planning: daily MAE is 775 rentals/day on a mean of 6,043 (median error 11.7%), though the model is biased low by about 532 rentals/day.

**I3. Validation check.**
- *Context:* I chose the growth feature (D6) after seeing test scores, which makes test results slightly optimistic.
- *Decision:* repeat the comparison on a validation slice inside training (fit before 2012-04-01, validate Apr-Jun 2012) without touching the test set.
- *Reason:* `yr + mnth` also wins there (validation MAE 36.2 vs 39.8 for `t_days`, 81.1 with no growth feature), so the choice holds up. It is still a rough check, since only 3 months of 2012 are available to learn the `yr` effect.


## 3b. Prototype decisions (natural-language assistant, folder `prototype/`)

| # | Decision | Alternative | Reason |
|---|---|---|---|
| P1 | Rule-based parser by default; optional Gemini parser that only extracts fields as JSON. Answers always come from templates filled by the model | Let the LLM write the answer | No invented numbers, works offline, testable. Gemini output is re-validated and any failure falls back to rules. Unsupported topics are caught by a rule guard before the LLM is called |
| P2 | Four question types: bikes needed, demand, peak hours, "should we buy" | Open-ended chat | Covers the operations manager's real questions; anything else gets a help message |
| P3 | Refuse questions about stations, locations, prices, future dates | Give a best guess | The data is city-wide and ends 2012; a guess would look like a fact |
| P4 | Show assumptions (day, month, weather, trip minutes, service level) in every answer | Show only the number | The fleet number depends on them; the manager can see and change them |
| P5 | Safety multiplier = P90 of actual/forecast on Jul-Dec 2012 (rush and non-rush separately), planning level capped at the highest recorded hour | Use the forecast alone | The model under-predicts peaks. First version gave a planning level above the all-time record, so I added the cap |
| P6 | Unmentioned weather defaults to clear, with typical temperature for that month and weather type | Ask a follow-up | Keeps the prototype one-step; the default is stated in the answer |

Tested: 17 automated tests (parsing, input validation, refusals, Gemini path with a mocked response, fallback on failure and on bad output, peak hours are commute hours on working days and midday on weekends, the cap, and the cautious "buy" answer) and the Streamlit page loads and shows answers and refusals. The live Gemini call is untested (no access to Google's API from the build environment); test it with my own key before presenting.

## 4. Results in one place

- Test (Jul-Dec 2012) MAE / R2: last-week baseline 64.3 / 0.75; Poisson GLM 52.1 / 0.85; boosting 43.4 / 0.90.
- Rush hours remain under-predicted by about 60 rentals/hr; daily totals about 9% low.
- Fleet (observed demand, test period, 20 min): mean 85 bikes in use, P90 192 (2.3x average), busiest hour 326. A P90-sized fleet would be about 44% used in an average hour.
- Recommendation: do not buy bikes yet; pilot forecast-guided rebalancing before commuter peaks, collect station-level availability and failed-search data, check bike health. This is consistent with a distribution problem but does not prove it.

## 5. AI use

- **AI helped with:** brainstorming the approach, explaining concepts, drafting and debugging the notebook code, and a first draft of this log.
- **I decided and can explain:** the split, the assumptions in section 1, the decisions in section 2, the hypothesis and the recommendation.
- **Verified:** the notebook was executed end to end and the figures above were read from its outputs. A reference solution I found was used to learn structure, not copied.

## 6. Limitations

| Limitation | Effect |
|---|---|
| No stations, inventory or unmet demand | Cannot show where bikes are missing; fleet number is a floor |
| Two years, strong 2011 to 2012 growth | Seasonality and growth partly confounded |
| Boosting under-predicts peaks | Fleet sizing uses observed demand as the reference |
| Weather taken as recorded | Real forecasts add error |
| Hypothesis H1 is design only | Needs a field experiment with station data |
