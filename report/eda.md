---
title: "Overreaction on the Court? A Behavioral Model of How Polymarket Bettors Price In-Game NBA Momentum"
subtitle: "CSE 163 Final Project: Exploratory Data Analysis"
author: "Anand Nachnani · <https://github.com/anandnachnani-byte/NBA>"
date: "November 2026"
fontsize: 11pt
linestretch: 1.15
geometry: margin=0.7in
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{htbp}
  - \usepackage{caption}
  - \captionsetup{skip=1pt}
  - \setlength{\intextsep}{3pt}
  - \usepackage{titling}
  - \setlength{\droptitle}{-6em}
  - \usepackage{titlesec}
  - \titlespacing*{\section}{0pt}{5pt}{1pt}
  - \setlength{\parskip}{2pt}
  - \usepackage{enumitem}
  - \setlist{nosep,leftmargin=*}
---

# 1. Title and Author(s)

Anand Nachnani. Repository: <https://github.com/anandnachnani-byte/NBA> (code, tests, sample data, and figures).

# 2. Summary of Research Questions

I compare how much the Polymarket price moves after an in-game NBA event with how much a statistical win-probability model says it should move. The gap measures bettor behavior.

1. **Immediate reaction.** After a scoring run (at least 8-0 within 3 game-minutes), does the price move more or less than the model? I estimate a reaction coefficient $\beta$ and test $\beta = 1$ (rational), $\beta > 1$ (overreaction), or $\beta < 1$ (underreaction).
2. **Reversion vs. momentum.** In the 10 minutes after a run, does the price drift back toward the model or keep moving, compared with random moments?
3. **Context.** Does the reaction depend on the quarter, whether the favorite or the underdog is on the run, and trading volume?
4. **Economic significance.** Would buying the team that got run on and selling 10 minutes later make money after the spread?

# 3. Motivation

Prediction markets are more and more often quoted as real-time forecasts, but it is still an open question whether their prices are accurate during a game. Behavioral finance shows that people overreact to vivid information, like a belief in the "hot hand." A scoring run is a near-perfect natural experiment: it is loud and emotional, but a 10-0 run in the first quarter changes the true win probability much less than fans feel it does. The answers test whether a crowd processes news rationally, which matters to fans and to me as someone working toward quantitative finance, where how fast prices absorb news is the same question. Here, though, the "news" is clean, timestamped, and public.

# 4. Dataset

I use three public sources that need no API key; `fetch_data.py` caches them as Parquet in `data/raw/`, and a 10-game sample is committed in `data/sample/`. The **Polymarket Gamma API** (`gamma-api.polymarket.com/events`) is the catalog: I page through every closed NBA event and keep each game's moneyline market (1,682 markets). The **CLOB price history** (`clob.polymarket.com/prices-history`) gives the home team's price every minute from one hour before tip-off to four hours after, with a fallback to `data-api.polymarket.com/trades` that only one game needed. **`nba_api`** gives 1,525 games from `LeagueGameFinder` and the live play-by-play (`period`, `clock`, `timeActual`, `scoreHome`, `scoreAway`, `teamTricode`). The scope is the 2025-26 season plus 2026-27 through November 17, 2026.

# 5. Method

My EDA follows the pipeline in Figure 1, which I drew in Figma before writing code. I parse the team tokens from the nested JSON and match each market to its game when both teams and the date agree. I try the exact date first, because teams sometimes play back-to-back nights, and I log failures instead of fixing them by hand. For each game I build a one-minute grid on real-world time and use `merge_asof` to attach the latest score and the price for that minute. Polymarket's minute prices arrive about 6 seconds after each minute, so I assign each price to the minute it falls in. The play-by-play needed more cleaning than I expected: the NBA feed logs some plays late with a stale score, so I sort plays by game time and never let a score go down. Without this fix, my code found impossible "40-0 runs."

![Data flow from the raw APIs to Questions 1–4 (drawn in Figma).](figures/fig0_dataflow.png){width=100%}

The benchmark is a logistic regression fit on 80% of games. It predicts a home win from the margin, the margin divided by $\sqrt{\text{seconds left}+1}$, and the pregame log-odds. A run is 8 or more unanswered points within 180 game-seconds. The market reaction is the change in the price's log-odds from 1 minute before the run to 2 minutes after it, signed toward the running team. The fair reaction is the model's change over the same window, and the drift is the market's change minus the model's from +2 to +10 minutes. As a placebo, I compute the same drift at three random minutes per game that are at least 15 minutes from any run.

# 6. EDA Results

**Size.** {{table_sizes_text}} A row is one moneyline market in A, one minute of one game's price in B, and one play in C. In D, a row is one matched game with its result and per-game measures. In E, a row is one game-minute with the score, price, and fair probability.

**Missing data.** Most of my missing data has one cause. The NBA's content delivery network blocked my computer partway through the download, so play-by-play exists for only 607 of 1,515 matched games, and 59.9% of D's per-game columns are missing. These are the first 607 games in date order (October 21, 2025 to January 16, 2026), so the results do not yet cover the playoffs or late season. I am re-downloading the rest slowly and will re-run every number afterward. The other gaps are small. 9.6% of markets lack `market_volume`, so I use the event volume. 1.6% of plays lack `teamTricode` because they are period breaks. 1.6% of game-minutes have no price fresher than three minutes, and I leave these missing. 2.6% of runs have no reaction (5.1% no drift) because they end too close to the final buzzer. Finally, 167 markets did not match (154 preseason, 6 duplicates, 5 ambiguous, 2 using "PHO"), and all are logged.

**Variables of interest.** The market belief is `price_home` and its log-odds. The game state is `margin`, `sec_remaining`, and `period`, which feed the benchmark with `pregame_price`; `period` is also the Q3 quarter split. The context variables are `volume` (by tercile), `runner_role` (favorite or underdog), and `quarter`. The outcomes are `market_reaction` and `fair_reaction` (Q1), `drift` (Q2), and `fade_return` (Q4).

{{table_quant}}
Table: Seven-number summaries (letter = source table).

Three summaries stand out. The median minute change in log-odds is zero, yet within games the price moves in 73% of minutes. The market and model reactions to runs have nearly the same mean (0.866 vs. 0.858), which argues against a large average overreaction. Finally, `fade_return` reaches 10 when a price is near zero, so I will cap entry prices at 0.05–0.95.

The categorical variables take these values. Runs by team on the run: home 1,138, away 1,064. By runner role: favorite 1,206, underdog 996. By quarter: Q1 490, Q2 554, Q3 579, Q4 569, OT 10. By volume tercile: low 791, mid 877, high 534. The games in D are 1,424 regular-season, 85 playoff, and 6 play-in games, and the home team won 830 of 1,515. The Appendix lists every value.

![Dataset A: most games trade \$1–3 million (a); markets follow the NBA calendar (b).](figures/fig1_markets.png){width=88%}

**Figure 2** shows each market's volume on a log scale, with my tercile cut-offs, because volume ranges from \$22 to \$28 million. Almost every game is heavily traded (median \$1.9 million), so even my "low volume" tercile is a liquid market.

![Dataset B: the market tracks the benchmark during runs (a); price changes have fat tails (b).](figures/fig2_prices.png){width=88%}

**Figure 3** follows Dallas against San Antonio on October 22, 2025, with each Spurs run shaded. For example, the Spurs' 13-0 second-quarter run cut Dallas from about 63% to 35%, and the market and model fell together. The histogram compares minute changes in log-odds with a normal curve. Their kurtosis is 122, so my tests should not assume normal errors.

![Dataset C: runs are spread evenly across quarters (a); margins widen as the game goes on (b).](figures/fig3_pbp.png){width=88%}

**Figure 4** counts runs per quarter and shows the margin at the end of each quarter. Runs are about equally common in every quarter (0.8 to 1.0 per game), so the Q1 vs. Q4 comparison will be balanced.

![Runs: market reaction slope just below one (a); the market moves with the model, then stays flat (b).](figures/fig6_validity.png){width=88%}

**Figure 5** plots each run's market reaction against the model's, with the $y = x$ line a rational market would follow, and the average path around a run. The slope is 0.88, a little below one. In panel (b), the market moves with the model minute by minute and then stays flat. The mean drift is 0.025 against a placebo of $-0.006$, and fading the run averages $-2.1\%$ before the spread. My first look shows a market that prices runs about right, with no overreaction or reversion. This is the opposite of what I expected in my proposal. The benchmark is a fair yardstick: on held-out games, its Brier score is 0.152 against the market's 0.149, and both are well calibrated (`figures/fig5_minutes.png`).

**Result Validity.**

- **Q1:** $H_0: \beta = 1$, tested with OLS and standard errors clustered by game. The scatter is linear, and there are 595 clusters with 3.6 runs each, well above 50. The residuals are heavy-tailed (kurtosis 18), so I will also trim 1%. The assumption that fails is that the fair reaction is measured without error, since benchmark noise pulls $\beta$ toward zero. A first instrumental-variables check gives about 0.95, so I will report both.
- **Q2:** $H_0$: the mean drift after runs equals the mean placebo drift, tested with a bootstrap that resamples whole games. Games are independent, the 1,813 placebo windows never overlap runs, and the kurtosis of 14 rules out a t-test.
- **Q3:** $H_0$: every interaction coefficient is zero, tested with the same clustered OLS. Every group has at least 490 runs except overtime (10), which I will merge into Q4.
- **Q4:** $H_0$: the mean net return is at most zero, tested with a one-sided game bootstrap. Prices-history has no bid or ask, so I will estimate the spread from trades and also show 1¢ and 2¢ spreads.

# 7. Challenge Goals Update

**Messy Data** has expanded. Beyond paging and nested JSON, I handled problems the documentation did not show: late-logged plays with stale scores, back-to-back games, missing volume, minute prices that arrive seconds after the minute (which first created a fake one-minute "lag"), and an NBA CDN that blocks fast clients (the fetch now waits 1.5 seconds per game and stops after five failures). **Result Validity** is the same in scope: the clustered errors, bootstrap, placebo, and calibration check are built, and I added the instrument and spread bands.

# 8. Work Plan Evaluation

My estimates were close for setup (3 hours) and the Polymarket pipeline (8 hours). The NBA pipeline and join (7 hours) took longer, because the late plays, back-to-back games, and CDN block were surprises I only found in the data. Run detection and a preliminary benchmark (task 4) are done, and the statistical testing belongs in the gap my plan left between tasks 4 and 6. What remains is finishing the download (1 hour), the final benchmark (3 hours), the tests for Questions 1–4 (6 hours), and the write-up (6 hours).

# 9. Testing

`test_eda.py` runs 13 assert-based tests on small fixtures in `tests/data/` whose answers I worked out by hand, because a quiet bug in parsing, matching, or run detection would change every number here. For example, a fake game with a 9-0 run, a 7-0 burst, a slow 8-0 streak over 270 seconds, and a fourth-quarter 8-0 run must give exactly two runs. After the 40-point bug, I added late and stale-score rows that must not create a run. Other tests check the clock (`PT11M32.00S` is 692 seconds), matching (a back-to-back, "PHO", a preseason game, and a duplicate market), and that a price printed seconds after a minute belongs to that minute and is carried forward for at most three minutes. A final test runs `eda.py` on the 10-game sample and checks invariants: prices in [0, 1], runs of at least 8 points, and every game in the minute table. I also checked the figures by eye (prices should fall during away runs), and all code passes `flake8`.

# 10. Collaboration

Claude (Anthropic's AI assistant) helped write and debug the code and draft this report; I checked its output against the data and tests. I also used Figma and the Polymarket, `nba_api`, pandas, and statsmodels documentation.

\newpage

# Appendix: Categorical counts

Every categorical variable, with the count of each value (generated by `eda.py`).

{{table_cats_all}}
