# Edge search: trends in how Polymarket bettors price NBA games

Exploratory research to decide where the paper should focus. It is **not** trading advice. Every number comes from `python edge_search.py` (output in `results/edge_stdout.txt`, `results/edge_numbers.json`, `results/edge_*.csv`). All intervals are 95% bootstrap intervals that resample **whole games** (1,000 draws). In-game analyses use the 607 games that have play-by-play (Oct 21 2025 to Jan 16 2026, regular season only). Pregame analyses and one in-game replication use all 1,513 priced games.

**Terms.** *Pass-through* PT(k) = Σ market log-odds change / Σ benchmark log-odds change toward the running team. Both are measured from 1 min before the run to k min after it ends. PT = 1 means the market moved as much as the model. *Eda alignment* is the minute grid in `eda.py`, which takes the last price print at or before each minute. *Next-print alignment* takes the first print within 59 s after the minute.

**Multiple testing.** The script reports about 500 interval estimates in roughly 40 families: run curves, 7 context splits × 4 metrics, the outcome test, distributed lags, ACFs, jumps, calibration buckets, and pregame models. About 25 intervals would exclude zero by chance alone. Only results that hold across specifications, or that are large relative to their CI, are ranked high below.

---

## 1. The "1-2 minute lag" is a timestamp-alignment artifact. The market prices runs within about 10-20 s of the NBA log. (Robust)

| measure | estimate [95% CI] | n |
|---|---|---|
| PT(+1 min), eda alignment | 0.66 [0.64, 0.68] | 1,940 runs / 588 games |
| PT(+2 min), eda alignment | 1.00 [0.98, 1.02] | same |
| PT(0), next-print alignment | 0.975 [0.956, 0.993] | 1,913 / 587 |
| Minutes to close 90% of the gap | 2 (eda) vs 0 (next print), in every draw | |
| Trades: PT at 0 s / +10 s / +20 s after last basket | 0.81 [0.71, 0.91] / 0.91 [0.81, 1.00] / 0.98 [0.87, 1.08] | 60 runs / 60 games |
| Trades: median trades in the 60 s after a run | 18 | |
| All minutes, distributed lag: peak coefficient | lag 1 = 0.56 (eda) vs lag 0 = 0.555 [0.535, 0.574] (next print) | 80,089 minutes / 607 games |

**Interpretation.** CLOB price prints land about 5-15 s past each minute (median spacing 60 s). `build_game_minutes` takes the last print at or before hh:mm:00, so it pairs the game state at hh:mm:00 with a price from about 50 s earlier. That gap alone accounts for the jump from PT≈0.66 to PT≈1.0 between +1 and +2 min. When the price is aligned to the next print, almost all of the move is already in place at k = 0. Second-level trades for a random sample of 60 runs (Polymarket data API v2) show the market reaching about 98% of the model's move within 20 s of the last basket's `timeActual`. The market is also already about 80% of the way there *at* the logged time, which is roughly the share of the run scored before its last basket. Bettors trade as fast as the official log. The event-study "lag" in `eda.md` should not be read as slow updating. See `figures/edge_passthrough.png` and `figures/edge_distributed_lag.png`.

**Caveats.** `timeActual` is NBA log time, not broadcast time, so "within 20 s" is relative to the log. The trade sample is small (60 runs, one per game) and each price is the median of the last 3 trades. The lead coefficients are small but nonzero (lag −1 = 0.068 [0.058, 0.077] under next print), which partly reflects the 8 s look-ahead in that alignment. The truth lies between the two alignments.

**How this could change the paper.** Reframe Q1 around **speed and completeness of updating**, with "no measurable lag at minute resolution" as a headline. Fix the alignment in the main pipeline, either with nearest print or by using trades. Avoid presenting the lag as a behavioral finding. Trade-level data for a larger sample of runs is the natural upgrade.

## 2. The 0.77 slope is not evidence of underreaction. Average reaction ≈ model, and post-run prices are close to calibrated. (Robust)

| check (+2 min unless noted) | estimate [95% CI] |
|---|---|
| OLS slope, eda alignment | 0.77 [0.73, 0.82] |
| OLS slope, next-print alignment | 0.88 [0.85, 0.92] |
| Reverse-regression upper bound (1/b) | 1.11 [1.05, 1.19] |
| IV slope, instrument = run_pts / √(sec_left+1) | 0.95 [0.87, 1.06] |
| Ratio of sums (Σ market / Σ model) | 0.99 [0.96, 1.01] |
| OLS slope at +15 min (eda) | 0.85 [0.82, 0.89] |
| Outcome test: runner win rate − market price at +2 min | +1.1 pp [−0.1, +2.3] (2,194 runs / 597 games) |
| same at +10 min | +1.3 pp [+0.2, +2.6] |

**Interpretation.** The slope below 1 comes from two things. First, the alignment artifact attenuates it (0.77 rises to 0.88). Second, errors-in-variables: the benchmark is a noisy proxy for the true fair change. The IV estimate and the OLS/reverse-regression bounds both bracket 1. A test that does not use the model at all looks directly at outcomes after runs. If anything, running teams win *slightly more* often than the post-run price implies, which is the opposite of overreaction, though the effect is small and borderline. The benchmark does the same (+1.0 pp), so this probably reflects outcome noise in this sample.

**How this could change the paper.** Report β with attenuation-robust versions (IV, reverse bounds, ratio of sums) and add the outcome-based calibration test as the primary check of "rational vs. over/under". The honest headline is "no detectable average mispricing of runs."

## 3. Relative to the benchmark, market reactions are *compressed*. The outcome test does not confirm this as mispricing, which points to benchmark misspecification. (Strong pattern; meaning uncertain)

| split (PT at +10 min, eda) | estimate [95% CI] | runs / games |
|---|---|---|
| Runner was pregame longshot (<0.30) | 1.25 [1.17, 1.43] | 363 / 205 |
| Runner was heavy favorite (>0.70) | 0.92 [0.89, 0.95] | 543 / 223 |
| Difference | +0.33 [+0.23, +0.46] | |
| Runner trailing by 10+ before run | 1.21 [1.09, 1.37] | 493 / 358 |
| Runner leading by 10+ | 1.01 [0.95, 1.06] | 276 / 175 |
| Difference | +0.21 [+0.08, +0.39] | |
| Q1-Q2 | 1.07 [1.05, 1.10] | 1,044 / 513 |
| Final 5 min / OT | 0.90 [0.82, 1.08] | 228 / 209 |
| Difference | −0.17 [−0.29, −0.04] | |
| Drift (+2 to +10), longshot runner | +0.109 [+0.078, +0.150] log-odds | |
| Distributed-lag cumulative pass-through, all minutes before final 5 min | 0.99 [0.97, 1.01] | 70,634 min |
| same, final 5 min / OT | 0.88 [0.86, 0.92] | 9,561 min |

**Interpretation.** Compared with the benchmark, the market moves *more* when the benchmark moves little (early game, comeback runs, longshot runners) and *less* when it moves a lot (late game, heavy favorites). That is the same pattern that produces the slope below 1. The outcome check (`figures/edge_context.png`, panel b) does not show longshot or comeback runners being overpriced after their runs. Longshot runner win rate − price is +5.5 pp [−0.3, +11.7]. Late-game Brier scores on held-out games are also no worse for the market (market − model in the final 5 min: −0.0036 [−0.010, +0.002]). So these gaps more likely show where the simple logistic benchmark is wrong than where bettors are wrong. The benchmark has a fixed pregame weight, is additive in logit, and has no interactions. The runner-pregame outcome pattern (dogs +4.5 pp, favorites −4.0 pp) mirrors the sample-period calibration in trend 5, not anything specific to runs.

**How this could change the paper.** Q3 (context) is only as good as the benchmark. Before interpreting context effects, improve it: interact pregame log-odds with time remaining, add possession and free-throw state, or compare against a published win-probability model. Then judge context effects with the outcome test as well as β.

## 4. Timeouts after runs: the market prices about 5-8% less of the run for 1-3 minutes, then the difference disappears. (Moderate; one context of seven)

| next-print alignment, timeout within 90 s of run end (629 runs / 392 games) minus no timeout (1,573 / 564) | estimate [95% CI] |
|---|---|
| ΔPT at +1 min | −0.048 [−0.078, −0.014] |
| ΔPT at +2 min | −0.075 [−0.107, −0.043] |
| ΔPT at +3 min | −0.059 [−0.094, −0.024] |
| ΔPT at +10 min | −0.016 [−0.055, +0.027] |
| Drift +2 to +10 (eda), timeout vs none | +0.090 [+0.059, +0.117] vs −0.017 [−0.034, +0.001] |
| Outcome test at +10, timeout runs | +2.8 pp [+0.3, +5.3] |

**Interpretation.** When a timeout follows a run, the benchmark's +2 min reaction is larger (0.97 vs 0.81 log-odds) and then partly reverts as the other team answers. The market holds back part of the move, consistent with a belief that "timeouts stop runs". By +10 min the two paths agree. The outcome test leans slightly toward the market being *too* cautious after timeouts. It survives both alignments, but it is one of seven splits and the timeout flag is crude: any team, within 90 s of real time.

**How this could change the paper.** This is a candidate behavioral Q: do bettors price a "momentum-stopping" effect of timeouts, and is it justified by subsequent scoring? It needs attribution of the timeout to a team and a placebo (timeouts with no preceding run).

## 5. Favorite-longshot bias appears only at the extremes, and calibration swings with the season period. (Suggestive; fragile)

| measure | estimate [95% CI] | n |
|---|---|---|
| Pregame: favorites priced 0.90-0.95 at tip, win rate − price | +6.5 pp [+2.9, +8.2] (68 of 69 won) | 69 games |
| Pregame recalibration slope, all games (1 = calibrated, >1 = favorites underpriced) | 1.06 [0.94, 1.19] | 1,513 |
| Pregame home bias (intercept with logit offset) | −0.02 [−0.14, +0.09] | 1,513 |
| Slope, 2025-26 Dec-Jan | 0.66 [0.46, 0.88] | 425 |
| Slope, 2025-26 Feb-Apr | 1.34 [1.12, 1.55] | 498 |
| Difference | +0.68 [+0.37, +1.03] | |
| Slope, 2026-27 Oct-Nov | 1.39 [1.06, 1.97] | 205 |
| In-game, price ≥ 0.98, pbp games: win rate − price | +0.39 pp [+0.20, +0.54] | 10,284 min / 568 games |
| In-game, price 0.90-0.95, price-only games (0-120 min after tip) | +3.4 pp [+1.7, +5.0] | 10,928 min / 661 games |
| In-game, price 0.95-0.98, price-only games | +2.6 pp [+1.0, +3.4] | 7,254 min / 529 games |
| In-game, price 0.50-0.60, pbp games (favorites *over*priced) | −2.7 pp [−4.7, −0.6] | 15,394 min / 453 games |

**Interpretation.** Across the pregame and in-game samples, heavy favorites (≥0.90) win a bit more often than priced, which is the classic favorite-longshot bias. In the middle of the price range the sign flips between samples. The Oct-Jan play-by-play games show favorites overpriced, and the later price-only games show them underpriced. The pregame slope also differs sharply by period: Dec-Jan is below 1, while Feb-Apr and the new season are above 1. Late-season tanking and resting could explain Feb-Apr, but with about 400 games per period this could also be outcome luck. Each game contributes one outcome, so many correlated minutes do not add independent information. See `figures/edge_calibration.png`.

**Caveats.** Very few heavy-favorite losses (one at the pregame 0.90-0.95 level). Price ticks and the minimum price near 0/1. Five season periods were compared, with no correction. The price-only sample uses scheduled start + 11.1 min as tip-off and caps at 120 min to avoid post-game prints.

**How this could change the paper.** A secondary research question on favorite-longshot bias, pregame and in-game, using all 1,513 games. It is the one analysis that does not depend on play-by-play coverage. Frame it as calibration by price level and season period, not as an edge.

## 6. Big jumps keep going when there is score news. "No-news" jumps partly revert. (Weak; small n for no-news)

Jumps of |Δ log-odds| ≥ 0.5 per minute with a pre-jump price in [0.05, 0.95], next-print alignment:

| | n jumps / games | market change, next 5 min | market − model, next 5 min |
|---|---|---|---|
| News jumps (benchmark also moved ≥0.25 the same way) | 2,232 / 509 | +0.24 [+0.19, +0.30] | +0.15 [+0.12, +0.18] |
| No-news jumps (no score change ±1 min) | 64 / 62 | −0.35 [−0.78, +0.06] | −0.32 [−0.60, −0.07] |

Minute-change autocorrelation is small and positive (lag 1 = 0.044 [0.030, 0.058]). The market-minus-model change autocorrelation at lag 1 is −0.29 because the two series are misaligned at sub-minute scale, so it is not behavioral. Continuation after news jumps fits the small positive post-run drift and runs with PT slightly above 1 at +5 to +10 min (1.05 [1.03, 1.07], next print). It may also be benchmark error. No-news reversals are interesting but rest on only 64 jumps.

---

## What did NOT pan out

- **Slow updating / 1-2 min lag**: an alignment artifact (trend 1).
- **Volume**: no difference in run pass-through between high- and low-volume terciles (ΔPT(+1) = +0.02 [−0.03, +0.07]; ΔPT(+10) = −0.02 [−0.09, +0.05]).
- **Run size**: PT(+2) is about 1 for 8, 9-10 and 11+ point runs (0.98, 1.00, 0.99).
- **Halftime drift**: an early version suggested drift toward the leader. It vanished once the first 2 break minutes were dropped (stale last-basket prints), giving toward leader −0.003 [−0.012, +0.006] (552 games). Quarter breaks are too short on the minute grid to test.
- **Pregame drift (tip −60 min to tip)**: mean |drift| is 0.058 log-odds. It adds nothing beyond the tip price (coefficient −0.27 [−1.32, +1.08]), and the Brier score of the −60 min price equals the tip price's (difference +0.0002 [−0.0006, +0.0011]).
- **Team form / rest beyond the price**: season-to-date point differential +0.018 per point [−0.006, +0.039]; rest differential and back-to-back flags all include 0.
- **Home bias**: none detectable.
- **Market vs. benchmark accuracy by phase**: no phase where either is clearly better (held-out, 116 games).
- **Fade-the-run**: not revisited; the eda result (−1.6% before spread) and the near-calibrated post-run prices both say there is nothing there.

## General caveats

- Only 607 of 1,515 games have play-by-play, all from Oct 2025 to mid-Jan 2026 regular season. Every in-game run result comes from that window, which also had unusual calibration (trend 5).
- The prices are 1-minute prints, and a print appears every minute even without a trade. Stale prints damp moves in thin markets.
- `dp.logit` clips at 0.01/0.99, which truncates late-game moves. Results restricted to pre-run prices in [0.1, 0.9] are similar (slope 0.78).
- The benchmark is a 3-feature logistic model trained on 80% of the same games, so it is in-sample for most runs.
- The Polymarket trades v1 endpoint named in the plan has been retired. The script uses `data-api.polymarket.com/v2/trades?condition=...` with cursor paging, 0.35 s sleeps and a stop after 3 errors. The data is cached in `data/raw/trades/` (60 games).

## Recommended next analyses

1. Re-run the event study and β with corrected (nearest-print or trade-based) alignment. Pull trades for all runs, not just 60, to measure reaction speed in seconds.
2. Build a stronger benchmark (pregame × time-left interaction, possession, free-throw state) and repeat trend 3. Add the outcome test as a standard column in every context table.
3. Favorite-longshot bias on all 1,513 games: pregame and time-since-tip calibration, by season period, with a pre-registered set of buckets.
4. Timeouts: attribute each timeout to a team, add a no-run placebo, and check whether the market's "timeout discount" is matched by subsequent scoring.
