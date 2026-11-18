"""
Anand Nachnani
CSE 163 Final Project: Overreaction on the Court?

Exploratory data analysis. Loads the cached raw data, builds the five
tables used in the report, prints their sizes, missingness and
summaries, saves summary tables to results/ and figures to figures/.

Tables:
  markets  (A) Polymarket NBA moneyline markets (Gamma API)
  prices   (B) minute prices of the home-team token (CLOB)
  pbp      (C) cleaned NBA play-by-play
  joined   (D) markets joined to NBA games, one row per game
  minutes  (E) game-minute table: price and game state on real time
  runs     scoring runs with market and model reactions

Usage:
  python eda.py                    # full data in data/raw
  python eda.py --data data/sample # the small committed sample
"""
import argparse
import json
import os
import zlib

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

import data_processing as dp

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = 'figures'
RES_DIR = 'results'
MIN_PRICE_POINTS = 60
PRE_MIN = 1
POST_MIN = 2
LATE_MIN = 10
EVENT_WINDOW = range(-3, LATE_MIN + 1)
WP_FEATURES = ['margin', 'margin_scaled', 'pregame_logit']
SIDE_COLORS = {'home': '#2a6fdb', 'away': '#e07b39'}
FAV_COLORS = {'favorite': '#2a6fdb', 'underdog': '#e07b39'}


def load_data(data_dir):
    """
    Reads every cached file in data_dir into a dict with keys games,
    markets, matched, unmatched, prices and pbp. prices and pbp map a
    game_id to that game's DataFrame and only include matched games
    whose file exists.
    """
    data = {}
    for name, file in (('games', 'nba_games'),
                       ('markets', 'gamma_markets'),
                       ('matched', 'matched'),
                       ('unmatched', 'unmatched')):
        data[name] = pd.read_parquet(os.path.join(data_dir,
                                                  file + '.parquet'))
    for sub in ('prices', 'pbp'):
        data[sub] = {}
        for game_id in data['matched']['game_id']:
            path = os.path.join(data_dir, sub, game_id + '.parquet')
            if os.path.exists(path):
                data[sub][game_id] = pd.read_parquet(path)
    return data


def build_tables(data):
    """
    Cleans and joins the raw data into the tables described in the
    module docstring and returns them in a dict.
    """
    prices = []
    pbps = []
    minutes = []
    runs = []
    joined = []
    for game in data['matched'].to_dict('records'):
        game_id = game['game_id']
        price = data['prices'].get(game_id)
        raw_pbp = data['pbp'].get(game_id)
        info = {'game_id': game_id, 'has_prices': price is not None,
                'has_pbp': raw_pbp is not None,
                'n_price_points': 0 if price is None else len(price)}
        if price is not None and len(price):
            price = price.copy()
            price['game_id'] = game_id
            price['d_logit'] = np.diff(dp.logit(price['price']),
                                       prepend=np.nan)
            prices.append(price)
        if raw_pbp is None or price is None or len(price) == 0:
            joined.append(info)
            continue
        pbp = dp.clean_pbp(raw_pbp)
        pbp['game_id'] = game_id
        pbps.append(pbp)
        tip = pbp['ts'].min()
        pre = dp.pregame_price(price, tip)
        game_min = dp.build_game_minutes(pbp, price)
        game_min['game_id'] = game_id
        game_min['pregame_price'] = pre
        game_min['home_win'] = game['home_win']
        minutes.append(game_min)
        game_runs = dp.detect_runs(pbp)
        game_runs['game_id'] = game_id
        game_runs['pregame_price'] = pre
        if len(game_runs):
            runs.append(game_runs)
        moved = game_min['price_home'].diff().fillna(0) != 0
        info.update({'tip_ts': tip, 'pregame_price': pre,
                     'n_minutes': len(game_min),
                     'moved_share': float(moved.mean()),
                     'n_minutes_priced': int(
                         game_min['price_home'].notna().sum()),
                     'n_runs': len(game_runs)})
        joined.append(info)
    joined = data['matched'].merge(pd.DataFrame(joined), on='game_id')
    joined = joined.merge(data['games'][['game_id', 'season_type']],
                          on='game_id', how='left')
    # About 10% of Gamma markets lack a market-level volume; the event
    # volume (all markets on the game) is the closest stand-in.
    joined['volume'] = joined['market_volume'].fillna(
        joined['event_volume'])
    joined['priced_share'] = joined['n_minutes_priced'] / \
        joined['n_minutes']
    joined['volume_tercile'] = pd.qcut(joined['volume'], 3,
                                       labels=['low', 'mid', 'high'])
    minutes = pd.concat(minutes, ignore_index=True)
    minutes['pregame_logit'] = dp.logit(minutes['pregame_price'])
    minutes['margin_scaled'] = minutes['margin'] / np.sqrt(
        minutes['sec_remaining'] + 1)
    runs = pd.concat(runs, ignore_index=True)
    runs = label_runs(runs, joined)
    return {'markets': data['markets'],
            'prices': pd.concat(prices, ignore_index=True),
            'pbp': pd.concat(pbps, ignore_index=True),
            'joined': joined, 'minutes': minutes, 'runs': runs,
            'unmatched': data['unmatched'], 'games': data['games']}


def label_runs(runs, joined):
    """
    Adds context columns to runs: whether the running team was the
    pregame favorite, the quarter label, and the market volume tercile.
    """
    runs = runs.copy()
    home_fav = runs['pregame_price'] >= 0.5
    runs['runner_role'] = np.where(
        (runs['team'] == 'home') == home_fav, 'favorite', 'underdog')
    runs.loc[runs['pregame_price'].isna(), 'runner_role'] = np.nan
    runs['quarter'] = np.where(runs['period'] <= 4,
                               'Q' + runs['period'].astype(str), 'OT')
    tercile = joined.set_index('game_id')['volume_tercile']
    runs['volume_tercile'] = runs['game_id'].map(tercile)
    return runs


def is_train_game(game_id):
    """Deterministic 80/20 split of games for the benchmark model."""
    return zlib.crc32(game_id.encode()) % 5 != 0


def fit_wp_model(minutes):
    """
    Fits the preliminary logistic win-probability benchmark on training
    games (rows with a pregame price) and adds fair_prob / fair_logit to
    every minute. Returns (minutes, fitted model, held-out Brier scores
    for the model and the market).
    """
    minutes = minutes.copy()
    usable = minutes.dropna(subset=WP_FEATURES)
    train = usable[usable['game_id'].map(is_train_game)]
    model = sm.Logit(train['home_win'],
                     sm.add_constant(train[WP_FEATURES])).fit(disp=0)
    x = sm.add_constant(minutes[WP_FEATURES].fillna(0), has_constant='add')
    minutes['fair_prob'] = model.predict(x)
    minutes.loc[minutes['pregame_logit'].isna(), 'fair_prob'] = np.nan
    minutes['fair_logit'] = dp.logit(minutes['fair_prob'])
    test = minutes[~minutes['game_id'].map(is_train_game)].dropna(
        subset=['fair_prob', 'price_home'])
    brier = {
        'model': float(((test['fair_prob'] - test['home_win']) ** 2)
                       .mean()),
        'market': float(((test['price_home'] - test['home_win']) ** 2)
                        .mean()),
        'n_test_minutes': int(len(test)),
        'n_test_games': int(test['game_id'].nunique()),
    }
    return minutes, model, brier


def run_reactions(runs, minutes):
    """
    Adds, for each run, the market and model log-odds changes signed so
    that positive means toward the running team:
      market_reaction / fair_reaction: 1 min before the run starts to
        2 min after it ends (Question 1)
      drift: market change minus model change from 2 to 10 min after
        the run ends (Question 2)
      fade_return: return from buying the team that got run on at
        +2 min and selling at +10 min, before the spread (Question 4)
    """
    by_game = {g: df.set_index('minute_ts')
               for g, df in minutes.groupby('game_id')}
    rows = []
    for run in runs.to_dict('records'):
        game = by_game[run['game_id']]
        sign = 1 if run['team'] == 'home' else -1
        t_pre = run['start_ts'].floor('min') - pd.Timedelta(
            minutes=PRE_MIN)
        t_end = run['end_ts'].floor('min')
        t_post = t_end + pd.Timedelta(minutes=POST_MIN)
        t_late = t_end + pd.Timedelta(minutes=LATE_MIN)
        vals = {}
        for name, t in (('pre', t_pre), ('post', t_post),
                        ('late', t_late)):
            row = game.loc[t] if t in game.index else None
            vals['m_' + name] = np.nan if row is None else \
                row['logit_price']
            vals['f_' + name] = np.nan if row is None else row['fair_logit']
            vals['p_' + name] = np.nan if row is None else row['price_home']
        loser_post = 1 - vals['p_post'] if sign == 1 else vals['p_post']
        loser_late = 1 - vals['p_late'] if sign == 1 else vals['p_late']
        rows.append({
            'market_reaction': sign * (vals['m_post'] - vals['m_pre']),
            'fair_reaction': sign * (vals['f_post'] - vals['f_pre']),
            'drift': sign * ((vals['m_late'] - vals['m_post'])
                             - (vals['f_late'] - vals['f_post'])),
            'fade_return': (loser_late - loser_post) / loser_post,
        })
    return pd.concat([runs.reset_index(drop=True), pd.DataFrame(rows)],
                     axis=1)


def placebo_drifts(minutes, runs, n_per_game=3, seed=163):
    """
    Computes the same post-event drift at random minutes that are at
    least 15 minutes away from any run, with a random sign, so run
    drifts can be compared with ordinary price movement.
    """
    rng = np.random.default_rng(seed)
    run_times = runs.groupby('game_id')['end_ts'].apply(list).to_dict()
    out = []
    for game_id, game in minutes.groupby('game_id'):
        game = game.dropna(subset=['logit_price', 'fair_logit'])
        game = game.set_index('minute_ts')
        times = game.index[:-LATE_MIN]
        ends = run_times.get(game_id, [])
        far = [t for t in times
               if all(abs((t - e).total_seconds()) > 900 for e in ends)]
        if not far:
            continue
        picks = rng.choice(len(far), size=min(n_per_game, len(far)),
                           replace=False)
        for i in picks:
            t_post = far[i] + pd.Timedelta(minutes=POST_MIN)
            t_late = far[i] + pd.Timedelta(minutes=LATE_MIN)
            if t_post not in game.index or t_late not in game.index:
                continue
            sign = rng.choice([-1, 1])
            m = game.loc[t_late, 'logit_price'] - \
                game.loc[t_post, 'logit_price']
            f = game.loc[t_late, 'fair_logit'] - \
                game.loc[t_post, 'fair_logit']
            out.append({'game_id': game_id, 'drift': sign * (m - f)})
    return pd.DataFrame(out)


def event_study(runs, minutes):
    """
    Returns the average signed log-odds path of the market and the
    model around runs, relative to 1 minute before the run starts, for
    k minutes after the run ends (k in EVENT_WINDOW).
    """
    by_game = {g: df.set_index('minute_ts')
               for g, df in minutes.groupby('game_id')}
    rows = []
    for run in runs.dropna(subset=['market_reaction']).to_dict('records'):
        game = by_game[run['game_id']]
        sign = 1 if run['team'] == 'home' else -1
        base_t = run['start_ts'].floor('min') - pd.Timedelta(minutes=1)
        if base_t not in game.index:
            continue
        base = game.loc[base_t]
        for k in EVENT_WINDOW:
            t = run['end_ts'].floor('min') + pd.Timedelta(minutes=k)
            if t not in game.index:
                continue
            row = game.loc[t]
            rows.append({
                'k': k, 'game_id': run['game_id'],
                'market': sign * (row['logit_price']
                                  - base['logit_price']),
                'model': sign * (row['fair_logit'] - base['fair_logit']),
            })
    return pd.DataFrame(rows).dropna()


def missingness(df, name):
    """Returns a table of missing counts and percentages per column."""
    miss = pd.DataFrame({'missing': df.isna().sum(),
                         'pct_missing': 100 * df.isna().mean()})
    miss['table'] = name
    miss['n_rows'] = len(df)
    return miss.rename_axis('column').reset_index()


def summaries(tables):
    """
    Returns (quantitative, categorical) summary tables for the
    variables of interest.
    """
    joined = tables['joined']
    minutes = tables['minutes']
    runs = tables['runs']
    prices = tables['prices']
    quant = {
        'A market_volume (USD)': tables['markets']['market_volume'],
        'B price (home win prob)': prices['price'],
        'B d_logit per minute': prices['d_logit'],
        'C margin (home - away)': tables['pbp']['margin'],
        'C sec_remaining': tables['pbp']['sec_remaining'],
        'D pregame_price (home)': joined['pregame_price'],
        'D price points per game': joined['n_price_points'],
        'D priced_share of minutes': joined['priced_share'],
        'D moved_share of minutes': joined['moved_share'],
        'D runs per game': joined['n_runs'],
        'E logit_price': minutes['logit_price'],
        'E margin_scaled': minutes['margin_scaled'],
        'runs run_pts': runs['run_pts'],
        'runs market_reaction': runs['market_reaction'],
        'runs fair_reaction': runs['fair_reaction'],
        'runs drift': runs['drift'],
        'runs fade_return': runs['fade_return'],
    }
    counts = {k: int(pd.Series(v, dtype=float).replace(
        [np.inf, -np.inf], np.nan).notna().sum()) for k, v in quant.items()}
    quant = pd.DataFrame({k: dp.seven_number_summary(
        pd.Series(v, dtype=float).replace([np.inf, -np.inf], np.nan))
        for k, v in quant.items()}).T
    quant.insert(0, 'n', pd.Series(counts))
    cats = []
    for table, col, df in (
            ('A', 'season (slug year-month)',
             tables['markets']['slug_date'].str[:7]),
            ('B', 'source', prices['source']),
            ('C', 'period', tables['pbp']['period']),
            ('C', 'actionType', tables['pbp']['actionType']),
            ('D', 'season_type', joined['season_type']),
            ('D', 'volume_tercile', joined['volume_tercile']),
            ('D', 'home_win', joined['home_win']),
            ('unmatched', 'reason', tables['unmatched']['reason']),
            ('runs', 'team', runs['team']),
            ('runs', 'runner_role', runs['runner_role']),
            ('runs', 'quarter', runs['quarter']),
            ('runs', 'volume_tercile', runs['volume_tercile'])):
        counts = df.value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            cats.append({'table': table, 'variable': col,
                         'value': str(value), 'count': int(count)})
    return quant, pd.DataFrame(cats)


def save_fig(fig, name):
    """Saves a figure to figures/<name>.png and closes it."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + '.png'), dpi=200)
    plt.close(fig)


def plot_markets(tables):
    """Figure 1: market volume distribution and markets per month."""
    markets = tables['markets']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.0))
    vol = np.log10(markets['market_volume'].clip(lower=1))
    ax1.hist(vol, bins=40, color='#2a6fdb', alpha=0.85)
    cuts = np.log10(tables['joined']['volume'].quantile([1 / 3, 2 / 3]))
    for i, cut in enumerate(cuts):
        ax1.axvline(cut, color='black', ls='--', lw=1,
                    label='Volume tercile cut-offs' if i == 0 else None)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set(title='(a) Moneyline volume per game',
            xlabel='log10(volume in USD)', ylabel='Number of markets')
    month = markets['slug_date'].str[:7].value_counts().sort_index()
    ax2.bar(month.index, month.values, color='#2a6fdb')
    ax2.set(title='(b) Moneyline markets per month', xlabel='Month',
            ylabel='Number of markets')
    ax2.tick_params(axis='x', rotation=70, labelsize=7)
    save_fig(fig, 'fig1_markets')


def plot_prices(tables, example_game):
    """Figure 2: one game's price path and minute log-odds changes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.0))
    game = tables['minutes'][tables['minutes']['game_id'] == example_game]
    info = tables['joined'].set_index('game_id').loc[example_game]
    ax1.plot(game['minute_ts'], game['price_home'], color='black', lw=1.2,
             label='Market (home win prob)')
    if 'fair_prob' in game:
        ax1.plot(game['minute_ts'], game['fair_prob'], color='#888888',
                 lw=1, ls='--', label='Benchmark model')
    game_runs = tables['runs'][tables['runs']['game_id'] == example_game]
    for run in game_runs.to_dict('records'):
        ax1.axvspan(run['start_ts'], run['end_ts'],
                    color=SIDE_COLORS[run['team']], alpha=0.3, lw=0)
    for side, color in SIDE_COLORS.items():
        ax1.fill_between([], [], color=color, alpha=0.3,
                         label=f'{side} run')
    ax1.set(title=f"(a) {info['away']} at {info['home']}, "
            f"{info['game_date']}", xlabel='Time (UTC)',
            ylabel='Home win probability', ylim=(0, 1))
    ax1.legend(fontsize=7, loc='best')
    ax1.tick_params(axis='x', rotation=30, labelsize=7)
    d = tables['prices']['d_logit'].replace([np.inf, -np.inf], np.nan)
    d = d.dropna()
    d = d[d != 0]
    bins = np.linspace(-1.5, 1.5, 81)
    ax2.hist(d.clip(-1.5, 1.5), bins=bins, density=True, color='#2a6fdb',
             alpha=0.8, label='Observed (non-zero moves)')
    x = np.linspace(-1.5, 1.5, 300)
    sd = d.std()
    ax2.plot(x, np.exp(-x ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)),
             color='black', lw=1.2, label='Normal, same s.d.')
    ax2.set_yscale('log')
    ax2.set(title='(b) Minute-to-minute change in log-odds',
            xlabel='Change in log-odds (clipped at +/-1.5)',
            ylabel='Density (log scale)')
    ax2.legend(fontsize=7)
    save_fig(fig, 'fig2_prices')


def plot_pbp(tables):
    """Figure 3: runs per period by side and margin by period."""
    runs = tables['runs']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.0))
    counts = runs.groupby(['quarter', 'team']).size().unstack(fill_value=0)
    counts = counts.reindex([q for q in ['Q1', 'Q2', 'Q3', 'Q4', 'OT']
                             if q in counts.index])
    counts.plot.bar(ax=ax1, color=[SIDE_COLORS[c] for c in counts.columns],
                    rot=0)
    ax1.set(title='(a) Scoring runs (8-0 within 3 min) by quarter',
            xlabel='Quarter', ylabel='Number of runs')
    ax1.legend(title='Team on run', fontsize=8)
    pbp = tables['pbp']
    last = pbp.sort_values('actionNumber').groupby(
        ['game_id', 'period']).tail(1)
    last = last[last['period'] <= 4]
    sns.boxplot(data=last, x='period', y='margin', ax=ax2,
                color='#9ec1f5', fliersize=2)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set(title='(b) Score margin at the end of each quarter',
            xlabel='Quarter', ylabel='Home minus away points')
    save_fig(fig, 'fig3_pbp')


def plot_joined(tables):
    """Figure 4: pregame price by result and price coverage per game."""
    joined = tables['joined'].dropna(subset=['pregame_price'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.0))
    bins = np.linspace(0, 1, 21)
    for win, color, label in ((1, '#2a6fdb', 'Home team won'),
                              (0, '#e07b39', 'Home team lost')):
        ax1.hist(joined.loc[joined['home_win'] == win, 'pregame_price'],
                 bins=bins, alpha=0.6, color=color, label=label)
    ax1.set(title='(a) Pregame home win probability by result',
            xlabel='Polymarket price at tip-off', ylabel='Number of games')
    ax1.legend(fontsize=8)
    moved = tables['joined'].dropna(subset=['moved_share',
                                            'volume_tercile']).copy()
    moved['pct_moved'] = 100 * moved['moved_share']
    sns.boxplot(data=moved, x='volume_tercile', y='pct_moved', ax=ax2,
                color='#9ec1f5', fliersize=2,
                order=['low', 'mid', 'high'])
    ax2.set(title='(b) How often the price moves, by volume',
            xlabel='Market volume tercile',
            ylabel='Percent of game minutes with a price change')
    save_fig(fig, 'fig4_joined')


def plot_minutes(tables):
    """Figure 5: price vs margin by quarter and calibration curves."""
    minutes = tables['minutes'].dropna(subset=['price_home'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.0))
    reg = minutes[minutes['period'] <= 4]
    reg = reg[reg['margin'].between(-25, 25)]
    means = reg.groupby(['period', 'margin'])['price_home'].mean()
    palette = sns.color_palette('viridis', 4)
    for period, color in zip(range(1, 5), palette):
        line = means.loc[period]
        ax1.plot(line.index, line.values, color=color, label=f'Q{period}')
    ax1.set(title='(a) Average market price by score margin',
            xlabel='Home minus away points',
            ylabel='Average home win probability')
    ax1.legend(title='Quarter', fontsize=8)
    test = minutes[~minutes['game_id'].map(is_train_game)]
    test = test.dropna(subset=['fair_prob'])
    bins = np.linspace(0, 1, 11)
    for col, color, label in (('price_home', 'black', 'Market'),
                              ('fair_prob', '#e07b39', 'Benchmark model')):
        groups = pd.cut(test[col], bins, include_lowest=True)
        curve = test.groupby(groups, observed=True).agg(
            pred=(col, 'mean'), actual=('home_win', 'mean'))
        ax2.plot(curve['pred'], curve['actual'], marker='o', color=color,
                 label=label)
    ax2.plot([0, 1], [0, 1], color='#888888', ls='--', lw=1,
             label='Perfect calibration')
    ax2.set(title='(b) Reliability on held-out games',
            xlabel='Predicted home win probability',
            ylabel='Observed home win rate')
    ax2.legend(fontsize=8)
    save_fig(fig, 'fig5_minutes')


def plot_validity(tables, study, placebo):
    """
    Figure 6: market vs. model reaction to runs and the average
    event-time path of both.
    """
    runs = tables['runs'].dropna(subset=['market_reaction',
                                         'fair_reaction'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.1))
    sns.scatterplot(data=runs, x='fair_reaction', y='market_reaction',
                    hue='runner_role', palette=FAV_COLORS, s=10,
                    alpha=0.5, ax=ax1, edgecolor=None)
    lim = np.nanpercentile(np.abs(runs[['fair_reaction',
                                        'market_reaction']]), 99)
    x = np.linspace(-lim, lim, 10)
    ax1.plot(x, x, color='black', ls='--', lw=1, label='y = x (rational)')
    fit = np.polyfit(runs['fair_reaction'], runs['market_reaction'], 1)
    ax1.plot(x, np.polyval(fit, x), color='#c0392b', lw=1.5,
             label=f'OLS slope = {fit[0]:.2f}')
    ax1.set(xlim=(-lim, lim), ylim=(-lim, lim),
            title='(a) Market vs. model reaction to runs',
            xlabel='Model change in log-odds (toward runner)',
            ylabel='Market change in log-odds')
    ax1.legend(fontsize=7, title=None)
    path = study.groupby('k')[['market', 'model']].agg(['mean', 'sem'])
    for col, color, label in (('market', 'black', 'Market'),
                              ('model', '#e07b39', 'Benchmark model')):
        mean = path[(col, 'mean')]
        sem = path[(col, 'sem')]
        ax2.plot(mean.index, mean.values, color=color, marker='o', ms=3,
                 label=label)
        ax2.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                         color=color, alpha=0.15)
    ax2.axvline(0, color='#888888', lw=0.8)
    ax2.axvspan(POST_MIN, LATE_MIN, color='#2a6fdb', alpha=0.07,
                label='Drift window (+2 to +10)')
    ax2.set(title='(b) Average path around runs (95% CI)',
            xlabel='Minutes after the run ends',
            ylabel='Log-odds change toward runner')
    ax2.legend(fontsize=7)
    save_fig(fig, 'fig6_validity')


def context_table(runs):
    """
    Mean market and model reactions and drift by quarter, runner role
    and volume tercile, with the slope of market on model reaction.
    """
    rows = []
    for col in ('quarter', 'runner_role', 'volume_tercile'):
        for value, group in runs.dropna(
                subset=['market_reaction', 'fair_reaction']).groupby(
                    col, observed=True):
            slope = np.polyfit(group['fair_reaction'],
                               group['market_reaction'], 1)[0] \
                if len(group) > 2 else np.nan
            rows.append({'split': col, 'group': str(value),
                         'n_runs': len(group),
                         'n_games': group['game_id'].nunique(),
                         'mean_market': group['market_reaction'].mean(),
                         'mean_model': group['fair_reaction'].mean(),
                         'slope': slope,
                         'mean_drift': group['drift'].mean()})
    return pd.DataFrame(rows)


def write_markdown_tables(tables, quant, cats, miss):
    """
    Writes the summary tables as Markdown files in results/ so the
    report (built by build_report.sh) always shows the current numbers.
    """
    sizes = pd.DataFrame([
        {'Table': name, 'Rows': f'{tables[name].shape[0]:,}',
         'Columns': tables[name].shape[1]}
        for name in ('markets', 'prices', 'pbp', 'joined', 'minutes',
                     'runs')])
    labels = {'markets': 'A (markets)', 'prices': 'B (prices)',
              'pbp': 'C (play-by-play)', 'joined': 'D (markets x games)',
              'minutes': 'E (game-minutes)', 'runs': 'runs'}
    sizes_text = '; '.join(
        f'{labels[name]} {tables[name].shape[0]:,} rows x '
        f'{tables[name].shape[1]} columns' for name in labels)
    with open(os.path.join(RES_DIR, 'table_sizes_text.md'), 'w') as f:
        f.write(sizes_text + '.\n')
    short = {'A market_volume (USD)': 'volume, USD (A)',
             'B d_logit per minute': 'd_logit/min (B)',
             'C margin (home - away)': 'margin (C)',
             'C sec_remaining': 'sec_remaining (C)',
             'D pregame_price (home)': 'pregame_price (D)',
             'D moved_share of minutes': 'moved_share (D)',
             'D runs per game': 'runs per game (D)',
             'runs run_pts': 'run_pts',
             'runs market_reaction': 'market_reaction',
             'runs fair_reaction': 'fair_reaction',
             'runs drift': 'drift', 'runs fade_return': 'fade_return'}
    q = quant.loc[list(short)].rename(index=short)
    for col in q.columns[1:]:
        q[col] = q[col].map(lambda v: f'{v:,.3g}' if abs(v) < 1000
                            else f'{v:,.0f}')
    q['n'] = q['n'].map(lambda v: f'{v:,}')
    gaps = miss[miss['missing'] > 0].copy()
    gaps['pct_missing'] = gaps['pct_missing'].round(1)
    gaps = gaps[['table', 'column', 'missing', 'pct_missing']]
    cat_short = cats[cats['table'].isin(['runs', 'unmatched', 'D'])]
    for name, df, index in (('sizes', sizes, False), ('quant', q, True),
                            ('missing', gaps, False),
                            ('cats_short', cat_short, False),
                            ('cats_all', cats, False)):
        with open(os.path.join(RES_DIR, f'table_{name}.md'), 'w') as f:
            f.write(df.to_markdown(index=index))
            f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join('data', 'raw'))
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    pd.set_option('display.width', 120)

    tables = build_tables(load_data(args.data))
    minutes, model, brier = fit_wp_model(tables['minutes'])
    tables['minutes'] = minutes
    tables['runs'] = run_reactions(tables['runs'], minutes)
    placebo = placebo_drifts(minutes, tables['runs'])
    study = event_study(tables['runs'], minutes)

    print('=== Sizes (rows, columns) ===')
    for name in ('markets', 'prices', 'pbp', 'joined', 'minutes', 'runs'):
        print(f'{name:8s} {tables[name].shape}')
    miss = pd.concat([missingness(tables[n], n) for n in
                      ('markets', 'prices', 'pbp', 'joined', 'minutes',
                       'runs')])
    miss.to_csv(os.path.join(RES_DIR, 'missingness.csv'), index=False)
    print('\n=== Columns with missing values ===')
    print(miss[miss['missing'] > 0].to_string(index=False))
    quant, cats = summaries(tables)
    quant.to_csv(os.path.join(RES_DIR, 'quantitative_summary.csv'))
    cats.to_csv(os.path.join(RES_DIR, 'categorical_counts.csv'),
                index=False)
    write_markdown_tables(tables, quant, cats, miss)
    print('\n=== Seven-number summaries ===')
    print(quant.round(3).to_string())
    context = context_table(tables['runs'])
    context.to_csv(os.path.join(RES_DIR, 'run_context.csv'), index=False)
    print('\n=== Reactions by context ===')
    print(context.round(3).to_string(index=False))
    print('\n=== Benchmark model ===')
    print(model.params.round(4).to_string())
    print('Held-out Brier:', brier)

    joined = tables['joined']
    numbers = {
        'n_gamma_markets': int(len(tables['markets'])),
        'n_nba_games': int(len(tables['games'])),
        'n_matched': int(len(joined)),
        'n_unmatched': int(len(tables['unmatched'])),
        'n_games_with_prices': int(joined['has_prices'].sum()),
        'n_games_with_pbp': int(joined['has_pbp'].sum()),
        'n_games_sparse_prices': int(
            (joined['n_price_points'] < MIN_PRICE_POINTS).sum()),
        'n_games_used': int(minutes['game_id'].nunique()),
        'n_minutes': int(len(minutes)),
        'pct_minutes_priced': float(100 * minutes['price_home']
                                    .notna().mean()),
        'n_runs': int(len(tables['runs'])),
        'n_runs_with_reaction': int(tables['runs']['market_reaction']
                                    .notna().sum()),
        'n_games_with_runs': int(tables['runs']['game_id'].nunique()),
        'ols_slope_all': float(np.polyfit(
            *tables['runs'].dropna(subset=['market_reaction',
                                           'fair_reaction'])[
                ['fair_reaction', 'market_reaction']].T.values, 1)[0]),
        'mean_drift_runs': float(tables['runs']['drift'].mean()),
        'mean_drift_placebo': float(placebo['drift'].mean()),
        'n_placebo': int(len(placebo)),
        'mean_fade_return': float(tables['runs']['fade_return'].mean()),
        'kurtosis_d_logit': float(tables['prices']['d_logit'].replace(
            [np.inf, -np.inf], np.nan).kurt()),
        'brier': brier,
        'wp_params': model.params.round(4).to_dict(),
    }
    with open(os.path.join(RES_DIR, 'eda_numbers.json'), 'w') as f:
        json.dump(numbers, f, indent=2, default=str)
    print('\n=== Key numbers ===')
    print(json.dumps(numbers, indent=2, default=str))

    counts = joined.set_index('game_id')['n_runs'].dropna()
    example = counts[counts >= 3].index[0] if (counts >= 3).any() \
        else counts.index[0]
    plot_markets(tables)
    plot_prices(tables, example)
    plot_pbp(tables)
    plot_joined(tables)
    plot_minutes(tables)
    plot_validity(tables, study, placebo)
    print('\nSaved figures to', FIG_DIR)


if __name__ == '__main__':
    main()
