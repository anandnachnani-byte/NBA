# Overreaction on the Court? A Behavioral Model of How Polymarket Bettors Price In-Game NBA Momentum

**Anand Nachnani**

> Baseline research proposal for this project. Treat it as the source of truth for the research questions, data sources, and methodology.

---

## 1. Summary of Research Questions

This project will build a behavioral model of how Polymarket bettors react to in-game NBA events. I will compare how much the market price moves after an event against how much a statistical win-probability model says it should move. The gap between the two measures bettor behavior.

1. **Immediate reaction.** When a team goes on a scoring run (at least 8-0 within 3 game-minutes), does the price move more or less than the win-probability model says it should? I will estimate a reaction coefficient, Market Reaction (β), and test whether β = 1 (rational), β > 1 (overreaction), or β < 1 (underreaction).
2. **Reversion vs. momentum.** In the 10 minutes after a run ends, does the price drift back toward the model (overreaction) or keep moving in the same direction (slow updating)? Is this drift significantly different from zero?
3. **Context.** Does the size of the reaction depend on the quarter (Q1 vs. Q4), whether the pregame favorite or the underdog is on the run, and how heavily the market is traded?
4. **Economic significance.** If prices revert, would buying the team that just got run on and selling 10 minutes later have made money after the bid-ask spread?

## 2. Motivation

Prediction markets like Polymarket are more and more often quoted as real-time forecasts, and live sports markets are some of the most heavily traded. However, it is still an open question whether these prices are accurate while the game is being played. Behavioral finance has shown that people tend to overreact to recent and vivid information, and belief in the "hot hand" is one example of this. An NBA scoring run is a near-perfect natural experiment. It is loud and emotional, but a 10-0 run in the first quarter changes the true win probability much less than most fans feel it does.

The answers matter because they test whether a crowd of bettors processes new information rationally, which is a core question of market efficiency. They also tell fans how much to trust live odds. Finally, I am working toward a career in quantitative finance, and how fast and how accurately prices take in new information is the same question asked in stock markets. The difference is that in a basketball game, the "news" is clean, timestamped, and public.

## 3. Dataset

This project uses three real, public sources that need no account or API key. Raw responses will be cached as Parquet files, and I will include a sample of this data with my final submission in case an endpoint changes. None of these datasets have been used in CSE 163, and I have not done this analysis in any other class.

- **Polymarket Gamma API** (`https://gamma-api.polymarket.com/events`): the catalog of markets, filtered down to NBA games. I will look at outcomes, volume, and the game start time.
- **Polymarket CLOB price history** (`https://clob.polymarket.com/prices-history?market=<token_id>&fidelity=1`): a timestamp `t` and price `p` (the implied probability) for each minute. If minute-level history is missing for older markets, I will rebuild prices from trades at `https://data-api.polymarket.com/trades`.
- **NBA play-by-play** through the `nba_api` package (https://github.com/swar/nba_api), which pulls from `https://cdn.nba.com/static/json/liveData/playbyplay/`. I will use `period`, `clock`, `timeActual` (the real-world time of each play), `scoreHome`, `scoreAway`, and `teamTricode`.

**Scope:** every 2025-26 NBA game with a Polymarket moneyline market (around 1,200 games), plus games from the start of the 2026-27 season.

Anyone can download the data by running `python fetch_data.py` from my repository, which saves all three sources to `data/raw/`. The Polymarket API documentation is at https://docs.polymarket.com.

## 4. Challenge Goals

**Messy Data.** None of this data comes as a CSV. It comes from three REST APIs with pagination, rate limits, and nested JSON (for example, `clobTokenIds` is a JSON string inside another JSON field). To build a usable table, I need to find NBA markets among thousands of events, match each team to its token, match each market to its NBA game ID using the teams and date in the slug, parse the game clock, and line up two separate time series on real-world time down to the minute.

**Result Validity.** Market Reaction (β) will be estimated with standard errors clustered by game, since events in the same game are not independent, and tested against β = 1. Confidence intervals for drift will come from a bootstrap that resamples whole games. A placebo test will repeat every calculation at random moments with no run, to show whether results around runs differ from normal price movement. The win-probability model will also be checked for calibration (Brier score and a reliability curve) on held-out games before I use it as the benchmark.

## 5. Method

**Building the table.** I will pull closed NBA markets from the Gamma API, parse the team tokens, and use `LeagueGameFinder` in `nba_api` to join each market to its game on date, home team, and away team. Unmatched markets will be logged and checked in code, not fixed by hand. The final table has one row per game-minute with the market price, score margin, period, and seconds remaining.

**Benchmark model and events.** Using held-out games, I will fit a logistic regression that predicts a home win from the score margin, the margin divided by √(seconds remaining + 1), and the pregame log-odds from the Polymarket price at tip-off. This gives a "fair" probability for every minute. I will then find runs (≥ 8-0 within 3 game-minutes) and record the period, the team on the run, and whether it was the pregame favorite.

**Question 1.** For each run, the market reaction is the change in the price's log-odds from 1 minute before the run to 2 minutes after it. The fair reaction is the model's change over the same window. Regressing the market reaction on the fair reaction gives Market Reaction (β). If β is significantly above 1, bettors overreact, and if it is significantly below 1, they underreact.

**Question 2.** The drift is the market's change from 2 to 10 minutes after the run, minus the model's change. I will compare the average drift with the placebo results. A negative drift means the market is correcting an overreaction.

**Question 3.** I will add interaction terms for the fourth quarter, the underdog being on the run, and volume tercile, and plot Market Reaction for each group with seaborn.

**Question 4.** I will simulate buying the team that got run on 2 minutes after the run and selling 10 minutes after, minus the spread, and report the average return with a bootstrap confidence interval.
