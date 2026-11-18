# Overreaction on the Court?

A behavioral model of how Polymarket bettors price in-game NBA momentum.
I compare how much the Polymarket moneyline price moves after an NBA
scoring run (8-0 or better within 3 game-minutes) against how much a
win-probability model says it should move.

* Research plan: [docs/baseline_research_paper.md](docs/baseline_research_paper.md)
* EDA report: [eda.pdf](eda.pdf) (source in [report/eda.md](report/eda.md))

## Running it

```bash
pip install -r requirements.txt
python fetch_data.py          # downloads and caches everything to data/raw/ (~1 hour)
python eda.py                 # tables -> results/, figures -> figures/
python test_eda.py            # tests on hand-built fixtures + the sample
flake8 .
./build_report.sh             # eda.py + pandoc -> eda.pdf
```

The full cache is not committed. `data/sample/` holds 10 real games in
the same layout, so `python eda.py --data data/sample` runs without any
download.

## Files

| File | What it does |
| --- | --- |
| `fetch_data.py` | Pages the Polymarket Gamma catalog, pulls minute prices from the CLOB (trade fallback), pulls NBA games and play-by-play with `nba_api`, joins markets to games, caches everything as Parquet |
| `data_processing.py` | Pure functions: nested-JSON parsing, slug parsing, market-to-game matching, clock parsing, play-by-play cleaning, run detection, aligning prices and game state on real time |
| `eda.py` | Builds the analysis tables, missingness and summaries, a preliminary win-probability benchmark, run reactions, placebo drifts and all figures |
| `test_eda.py` | Assert-based tests on small fixtures in `tests/data/` |

## Data sources

* Polymarket Gamma API: `https://gamma-api.polymarket.com/events`
* Polymarket CLOB price history: `https://clob.polymarket.com/prices-history`
* Polymarket trades: `https://data-api.polymarket.com/trades`
* NBA games and play-by-play via [nba_api](https://github.com/swar/nba_api)
