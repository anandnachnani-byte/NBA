"""
Anand Nachnani
CSE 163 Final Project: Overreaction on the Court?

Exploratory search for patterns in how Polymarket NBA moneyline prices
move, to decide where the paper should focus. This is descriptive
research about bettor behavior, not a trading tool. It reuses the
tables built in eda.py and looks at three areas:

  1. Scoring runs: how fast and how fully the market passes through the
     benchmark model's change, by context, at minute and (for a small
     sample) trade resolution, plus attenuation checks on the 0.77 slope.
  2. Other in-game patterns: distributed-lag pass-through of all model
     moves, autocorrelation, big-jump reversal, in-game calibration
     (favorite-longshot bias), and drift during breaks.
  3. Pregame pricing: calibration of the tip-off price, the last-hour
     drift, team form / rest information, and season period.

Every interval is a 95% bootstrap interval that resamples whole games.
Writes results/edge_numbers.json, results/edge_*.csv and
figures/edge_*.png; results/edge_report.md is written by hand from them.

Usage:
  python edge_search.py                 # data/raw, cached trades only
  python edge_search.py --fetch-trades  # also download the trade sample
  python edge_search.py --data data/raw --n-boot 1000
"""
import argparse
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

import data_processing as dp
import eda

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

RES_DIR = 'results'
FIG_DIR = 'figures'
SEED = 163
KS = list(range(-3, 16))
HORIZONS = (0, 1, 2, 3, 5, 10, 15)
LAGS = list(range(-2, 7))
TRADES_URL = 'https://data-api.polymarket.com/v2/trades'
TRADE_OFFSETS = [-60, -30, -10, -5, 0, 5, 10, 20, 30, 45, 60, 90, 120,
                 150, 180, 240, 300, 420, 600]
N_TRADE_RUNS = 60
FAV_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0001]
COLORS = {'market': '#1f1f1f', 'model': '#e07b39', 'alt': '#2a6fdb',
          'trades': '#2a9d5c'}


# ---------------------------------------------------------------------
# Game-level bootstrap
# ---------------------------------------------------------------------

class GameBoot:
    """
    Bootstrap that resamples whole games. Every statistic is computed
    from per-game column sums, and the same resampling weights are used
    for every statistic, so differences between groups also get valid
    intervals.
    """

    def __init__(self, game_ids, n_boot, seed=SEED):
        self.games = pd.Index(sorted(set(game_ids)))
        rng = np.random.default_rng(seed)
        g = len(self.games)
        self.weights = rng.multinomial(g, np.full(g, 1 / g),
                                       size=n_boot).astype(float)

    def totals(self, df, cols):
        """
        Returns (full-sample totals, bootstrap totals) of cols in df,
        shapes (C,) and (n_boot, C). df needs a game_id column.
        """
        sums = df.groupby('game_id')[cols].sum()
        sums = sums.reindex(self.games, fill_value=0)
        sums = sums.to_numpy(dtype=float)
        return sums.sum(axis=0), self.weights @ sums


def interval(est, draws):
    """Returns {'est', 'lo', 'hi'} from an estimate and its draws."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {'est': float(est), 'lo': np.nan, 'hi': np.nan}
    return {'est': float(est), 'lo': float(np.percentile(draws, 2.5)),
            'hi': float(np.percentile(draws, 97.5))}


def boot_fn(boot, df, cols, fn, draws=False):
    """
    Applies fn to the column totals of df (last axis indexes cols) for
    the full sample and every bootstrap draw. Returns an interval dict,
    or (estimate, draws) if draws is True.
    """
    df = df.dropna(subset=cols)
    est_t, boot_t = boot.totals(df, cols)
    with np.errstate(divide='ignore', invalid='ignore'):
        est, reps = fn(est_t), fn(boot_t)
    if draws:
        return est, reps
    return interval(est, reps)


def with_products(df, pairs):
    """Adds a column 'a*b' for each (a, b) in pairs and a column 'one'."""
    df = df.copy()
    for a, b in pairs:
        df[a + '*' + b] = df[a] * df[b]
    df['one'] = 1.0
    return df


def boot_mean(boot, df, col):
    """Mean of col with a game bootstrap interval."""
    d = with_products(df[['game_id', col]], [])
    return boot_fn(boot, d, [col, 'one'], lambda t: t[..., 0] / t[..., 1])


def boot_ratio(boot, df, num, den):
    """Ratio of sums (num / den) with a game bootstrap interval."""
    return boot_fn(boot, df[['game_id', num, den]], [num, den],
                   lambda t: t[..., 0] / t[..., 1])


def slope_from(t):
    """OLS slope from totals [x, y, x*x, x*y, n]."""
    x, y, xx, xy, n = (t[..., i] for i in range(5))
    return (n * xy - x * y) / (n * xx - x * x)


def boot_slope(boot, df, x, y):
    """OLS slope of y on x with a game bootstrap interval."""
    d = with_products(df[['game_id', x, y]], [(x, x), (x, y)])
    return boot_fn(boot, d, [x, y, x + '*' + x, x + '*' + y, 'one'],
                   slope_from)


def boot_iv(boot, df, z, x, y):
    """Instrumental-variable slope cov(z, y) / cov(z, x) with interval."""
    d = with_products(df[['game_id', z, x, y]], [(z, x), (z, y)])

    def fn(t):
        zs, xs, ys, zx, zy, n = (t[..., i] for i in range(6))
        return (n * zy - zs * ys) / (n * zx - zs * xs)
    return boot_fn(boot, d, [z, x, y, z + '*' + x, z + '*' + y, 'one'], fn)


def flat(name, ci, **extra):
    """Flattens an interval dict into one results row."""
    return {'stat': name, 'est': ci['est'], 'lo': ci['lo'], 'hi': ci['hi'],
            **extra}


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_inputs(data_dir):
    """
    Builds the eda.py tables, fits the benchmark and adds run
    reactions. Returns the tables dict with 'minutes' (with fair_prob,
    fair_logit) and 'runs' (with reactions) replaced, plus 'brier'.
    """
    tables = eda.build_tables(eda.load_data(data_dir))
    minutes, _, brier = eda.fit_wp_model(tables['minutes'])
    tables['minutes'] = minutes
    tables['runs'] = eda.run_reactions(tables['runs'], minutes)
    tables['brier'] = brier
    return tables


def realign_minutes(minutes, prices):
    """
    Returns a copy of minutes whose price is the next 1-minute price
    print within 59 s after each grid minute instead of the last print
    before it. CLOB prints land about 5-15 s past each minute, so the
    eda.py alignment pairs the game state at hh:mm:00 with a price from
    about 50 s earlier; this version pairs it with one about 8 s later.
    Together they bracket the true alignment.
    """
    out = []
    by_game = {g: df[['ts', 'price']].sort_values('ts')
               for g, df in prices.groupby('game_id')}
    for game_id, game in minutes.groupby('game_id', sort=False):
        game = game.sort_values('minute_ts').drop(
            columns=['price_home', 'logit_price'])
        price = by_game[game_id].rename(columns={'price': 'price_home'})
        game = pd.merge_asof(game, price, left_on='minute_ts',
                             right_on='ts', direction='forward',
                             tolerance=pd.Timedelta('59s')).drop(
                                 columns='ts')
        out.append(game)
    out = pd.concat(out, ignore_index=True)
    out['logit_price'] = dp.logit(out['price_home'])
    out.loc[out['price_home'].isna(), 'logit_price'] = np.nan
    return out


# ---------------------------------------------------------------------
# 1. Scoring runs
# ---------------------------------------------------------------------

def run_paths(runs, minutes, ks=KS):
    """
    Returns a long table (run_id, game_id, k, market, model, market_p,
    model_p) with the signed change toward the running team from 1 min
    before the run starts to k min after the run ends, in log-odds and
    in probability points. Same convention as eda.event_study, but kept
    per run so paths can be split by context.
    """
    cols = ['logit_price', 'fair_logit', 'price_home', 'fair_prob']
    by_game = {g: df.set_index('minute_ts')[cols]
               for g, df in minutes.groupby('game_id')}
    offsets = pd.to_timedelta(ks, unit='min')
    out = []
    for run_id, run in runs.iterrows():
        game = by_game[run['game_id']]
        sign = 1 if run['team'] == 'home' else -1
        base_t = run['start_ts'].floor('min') - pd.Timedelta(minutes=1)
        if base_t not in game.index:
            continue
        base = game.loc[base_t]
        vals = game.reindex(run['end_ts'].floor('min') + offsets)
        out.append(pd.DataFrame({
            'run_id': run_id, 'game_id': run['game_id'], 'k': ks,
            'market': sign * (vals['logit_price'].to_numpy()
                              - base['logit_price']),
            'model': sign * (vals['fair_logit'].to_numpy()
                             - base['fair_logit']),
            'market_p': sign * (vals['price_home'].to_numpy()
                                - base['price_home']),
            'model_p': sign * (vals['fair_prob'].to_numpy()
                               - base['fair_prob']),
        }))
    return pd.concat(out, ignore_index=True)


def pre_run_price(runs, minutes):
    """Home price 1 minute before each run starts (NaN if missing)."""
    by_game = {g: df.set_index('minute_ts')['price_home']
               for g, df in minutes.groupby('game_id')}
    return pd.Series({
        run_id: by_game[run['game_id']].get(
            run['start_ts'].floor('min') - pd.Timedelta(minutes=1), np.nan)
        for run_id, run in runs.iterrows()})


def label_contexts(runs, pbp):
    """
    Adds context labels to runs: size, phase (time left), runner score
    state before the run, runner pregame probability, side, volume
    tercile, whether a timeout came within 90 s after the run, and
    whether prices were away from the 0/1 clip zone.
    """
    runs = runs.copy()
    sign = np.where(runs['team'] == 'home', 1, -1)
    runs['size'] = pd.cut(runs['run_pts'], [7, 8, 10, 99],
                          labels=['8', '9-10', '11+'])
    sec = runs['sec_remaining']
    runs['phase'] = np.select(
        [runs['period'] <= 2, runs['period'] == 3,
         (runs['period'] == 4) & (sec > 300)],
        ['Q1-Q2', 'Q3', 'Q4 >5 min left'], 'Final 5 min / OT')
    before = sign * runs['margin_before']
    runs['runner_margin_before'] = before
    runs['score_state'] = pd.cut(
        before, [-99, -10, -1, 9, 99],
        labels=['trailing 10+', 'trailing 1-9', 'tied/leading 0-9',
                'leading 10+'])
    runner_pre = np.where(sign == 1, runs['pregame_price'],
                          1 - runs['pregame_price'])
    runs['runner_pregame'] = pd.cut(
        runner_pre, [0, 0.3, 0.5, 0.7, 1],
        labels=['longshot <0.30', 'dog 0.30-0.50', 'fav 0.50-0.70',
                'heavy fav >0.70'])
    runs['side'] = runs['team']
    timeouts = pbp[pbp['actionType'] == 'timeout'].groupby('game_id')['ts']
    timeouts = {g: np.sort(s.astype('int64').to_numpy())
                for g, s in timeouts}
    flags = []
    for game_id, end in zip(runs['game_id'], runs['end_ts']):
        t = timeouts.get(game_id, np.array([], dtype='int64'))
        lo = np.searchsorted(t, end.value)
        hi = np.searchsorted(t, end.value + 90 * 10 ** 9)
        flags.append('timeout <=90 s' if hi > lo else 'no timeout')
    runs['timeout_after'] = flags
    return runs


CONTEXTS = ['size', 'phase', 'score_state', 'runner_pregame',
            'volume_tercile', 'side', 'timeout_after']
OUTCOME_CONTEXTS = ['size', 'phase', 'score_state', 'runner_pregame',
                    'timeout_after']


def wide_paths(paths, value):
    """Pivots run_paths output to one row per run, one column per k."""
    return paths.pivot(index='run_id', columns='k', values=value)


def passthrough_curve(boot, paths, ks=KS, market='market', model='model'):
    """
    Pass-through PT(k) = sum of market change / sum of model change at
    k minutes after the run ends, using runs observed at every k.
    Returns (table of PT(k) with intervals, bootstrap draws (B, K)).
    """
    m = wide_paths(paths, market)[ks]
    f = wide_paths(paths, model)[ks]
    keep = m.notna().all(axis=1) & f.notna().all(axis=1)
    games = paths.drop_duplicates('run_id').set_index('run_id')['game_id']
    rows = []
    reps = []
    for k in ks:
        d = pd.DataFrame({'game_id': games[keep[keep].index],
                          'm': m.loc[keep, k], 'f': f.loc[keep, k]})
        est, draw = boot_fn(boot, d, ['m', 'f'],
                            lambda t: t[..., 0] / t[..., 1], draws=True)
        ci = interval(est, draw)
        rows.append({'k': k, 'pt': ci['est'], 'lo': ci['lo'],
                     'hi': ci['hi'], 'mean_market': d['m'].mean(),
                     'mean_model': d['f'].mean(), 'n_runs': len(d),
                     'n_games': d['game_id'].nunique()})
        reps.append(draw)
    return pd.DataFrame(rows), np.column_stack(reps)


def minutes_to_close(curve, draws, ks=KS, levels=(0.5, 0.75, 0.9)):
    """
    First k >= 0 at which PT(k) reaches each level, for the estimate
    and as a bootstrap interval over draws.
    """
    ks = np.asarray(ks)
    pos = ks >= 0

    def first(values):
        hit = np.nonzero(values[pos] >= level)[0]
        return ks[pos][hit[0]] if len(hit) else np.nan
    out = {}
    for level in levels:
        est = first(curve['pt'].to_numpy())
        reps = np.array([first(d) for d in draws], dtype=float)
        never = float(np.mean(np.isnan(reps)))
        reps = reps[~np.isnan(reps)]
        out[f'{int(level * 100)}%'] = {
            'est': est,
            'lo': float(np.percentile(reps, 2.5)) if len(reps) else None,
            'hi': float(np.percentile(reps, 97.5)) if len(reps) else None,
            'share_draws_never': never}
    return out


def slope_by_horizon(boot, paths, horizons=HORIZONS):
    """OLS slope of market on model change toward the runner at each k."""
    rows = []
    for k in horizons:
        d = paths[paths['k'] == k].dropna(subset=['market', 'model'])
        rows.append(flat(f'slope_k{k}', boot_slope(boot, d, 'model',
                                                   'market'),
                         k=k, n_runs=len(d)))
    return pd.DataFrame(rows)


def attenuation(boot, runs, paths):
    """
    Checks whether the 0.77 slope is an errors-in-variables artifact:
    OLS, reverse-regression upper bound, IV with run size scaled by
    time left as the instrument, and OLS on runs away from the 0/1 clip
    zone, at +2 and +10 minutes.
    """
    rows = []
    info = runs[['run_pts', 'sec_remaining', 'game_id', 'p_pre']].copy()
    info['z'] = info['run_pts'] / np.sqrt(info['sec_remaining'] + 1)
    for k in (2, 10):
        d = paths[paths['k'] == k].set_index('run_id').join(
            info[['z', 'p_pre']]).dropna(subset=['market', 'model', 'z'])
        d = d.reset_index()
        rows.append(flat('OLS', boot_slope(boot, d, 'model', 'market'),
                         k=k, n_runs=len(d)))
        rev = boot_slope(boot, d, 'market', 'model')
        rows.append({'stat': 'reverse-regression bound (1/b)',
                     'est': 1 / rev['est'], 'lo': 1 / rev['hi'],
                     'hi': 1 / rev['lo'], 'k': k, 'n_runs': len(d)})
        rows.append(flat('IV: z = run_pts / sqrt(sec_left + 1)',
                         boot_iv(boot, d, 'z', 'model', 'market'),
                         k=k, n_runs=len(d)))
        mid = d[d['p_pre'].between(0.1, 0.9)]
        rows.append(flat('OLS, pre-run price in [0.1, 0.9]',
                         boot_slope(boot, mid, 'model', 'market'),
                         k=k, n_runs=len(mid)))
        rows.append(flat('ratio of means (sum m / sum f)',
                         boot_ratio(boot, d, 'market', 'model'),
                         k=k, n_runs=len(d)))
    return pd.DataFrame(rows)


def context_table(boot, runs, paths):
    """
    For every context split and level: n runs and games, mean model
    reaction at +2, pass-through at +1, +2 and +10, and post-run drift
    (+2 to +10, market minus model), each with game bootstrap intervals.
    """
    rows = []
    drift = runs[['game_id', 'drift']]
    for col in CONTEXTS:
        for level, group in runs.groupby(col, observed=True):
            ids = group.index
            row = {'split': col, 'level': str(level),
                   'n_runs': len(group),
                   'n_games': group['game_id'].nunique()}
            for k in (1, 2, 10):
                d = paths[(paths['k'] == k) & paths['run_id'].isin(ids)]
                ci = boot_ratio(boot, d.dropna(subset=['market', 'model']),
                                'market', 'model')
                row[f'pt{k}'] = ci['est']
                row[f'pt{k}_lo'] = ci['lo']
                row[f'pt{k}_hi'] = ci['hi']
                if k == 2:
                    row['mean_model_k2'] = d['model'].mean()
            ci = boot_mean(boot, drift.loc[ids], 'drift')
            row.update({'drift': ci['est'], 'drift_lo': ci['lo'],
                        'drift_hi': ci['hi']})
            rows.append(row)
    return pd.DataFrame(rows)


def contrast(boot, paths, ids_a, ids_b, k):
    """
    Bootstrap interval for PT(k) of runs ids_a minus PT(k) of runs
    ids_b, using the same game draws for both.
    """
    reps = []
    ests = []
    for ids in (ids_a, ids_b):
        d = paths[(paths['k'] == k) & paths['run_id'].isin(ids)]
        est, draw = boot_fn(boot, d, ['market', 'model'],
                            lambda t: t[..., 0] / t[..., 1], draws=True)
        ests.append(est)
        reps.append(draw)
    return interval(ests[0] - ests[1], reps[0] - reps[1])


def run_outcome_test(boot, runs, minutes, ks=(2, 10)):
    """
    Model-free check of run pricing: for each run, the running team's
    market price k minutes after the run ends and whether it won.
    Win rate minus price, overall and by context, with game bootstrap
    intervals. Negative means the market priced the runner too high
    after the run (overreaction); positive means too low. The same is
    reported for the benchmark model's probability.
    """
    by_game = {g: df.set_index('minute_ts')[['price_home', 'fair_prob']]
               for g, df in minutes.groupby('game_id')}
    rows = []
    for run_id, run in runs.iterrows():
        game = by_game[run['game_id']]
        home = run['team'] == 'home'
        won = run['home_win'] if home else 1 - run['home_win']
        for k in ks:
            t = run['end_ts'].floor('min') + pd.Timedelta(minutes=k)
            if t not in game.index:
                continue
            p, f = game.loc[t, 'price_home'], game.loc[t, 'fair_prob']
            rows.append({'run_id': run_id, 'game_id': run['game_id'],
                         'k': k, 'won': won,
                         'mkt_excess': won - (p if home else 1 - p),
                         'model_excess': won - (f if home else 1 - f)})
    res = pd.DataFrame(rows).merge(runs[OUTCOME_CONTEXTS], left_on='run_id',
                                   right_index=True)
    out = []
    for k in ks:
        d = res[res['k'] == k]
        groups = [('all', 'all', d)] + [
            (col, str(level), part) for col in OUTCOME_CONTEXTS
            for level, part in d.groupby(col, observed=True)]
        for col, level, part in groups:
            m = boot_mean(boot, part, 'mkt_excess')
            f = boot_mean(boot, part, 'model_excess')
            out.append({'k': k, 'split': col, 'level': level,
                        'n_runs': int(part['mkt_excess'].notna().sum()),
                        'n_games': part['game_id'].nunique(),
                        'mkt': m['est'], 'mkt_lo': m['lo'],
                        'mkt_hi': m['hi'], 'model': f['est'],
                        'model_lo': f['lo'], 'model_hi': f['hi']})
    return pd.DataFrame(out)


def phase_brier(boot, minutes):
    """
    Held-out Brier score of the market and the benchmark by game phase,
    and market minus model (negative = market more accurate).
    """
    test = minutes[~minutes['game_id'].map(eda.is_train_game)].dropna(
        subset=['price_home', 'fair_prob']).copy()
    test['phase'] = ingame_phase(test)
    test['bm'] = (test['price_home'] - test['home_win']) ** 2
    test['bf'] = (test['fair_prob'] - test['home_win']) ** 2
    test['diff'] = test['bm'] - test['bf']
    rows = []
    for phase, part in test.groupby('phase'):
        for col, name in (('bm', 'market'), ('bf', 'model'),
                          ('diff', 'market - model')):
            rows.append(flat(name, boot_mean(boot, part, col),
                             phase=phase, n_minutes=len(part),
                             n_games=part['game_id'].nunique()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. Other in-game patterns
# ---------------------------------------------------------------------

def minute_changes(minutes, lags=LAGS):
    """
    Adds within-game minute changes: dm (market log-odds), dfair (model
    log-odds), dgap (dm - dfair), and dfair_l{j} = model change j
    minutes earlier (negative j = later, a lead).
    """
    df = minutes.sort_values(['game_id', 'minute_ts']).copy()
    g = df.groupby('game_id')
    df['dm'] = g['logit_price'].diff()
    df['dfair'] = g['fair_logit'].diff()
    df['dgap'] = df['dm'] - df['dfair']
    g = df.groupby('game_id')
    for j in lags:
        df[f'dfair_l{j}'] = g['dfair'].shift(j)
    return df


def distributed_lag(boot, df, lags=LAGS):
    """
    Regresses the market minute change on model changes at leads and
    lags (dm_t on dfair_{t-j}) with no intercept. Returns per-lag and
    cumulative (j >= 0) coefficients with game bootstrap intervals.
    """
    xcols = [f'dfair_l{j}' for j in lags]
    d = df.dropna(subset=['dm'] + xcols)[['game_id', 'dm'] + xcols].copy()
    names = []
    for a in xcols:
        for b in xcols:
            d[a + '*' + b] = d[a] * d[b]
            names.append(a + '*' + b)
        d[a + '*dm'] = d[a] * d['dm']
    cols = names + [a + '*dm' for a in xcols]
    est_t, boot_t = boot.totals(d, cols)
    p = len(xcols)

    def solve(t):
        xtx = t[:p * p].reshape(p, p)
        return np.linalg.solve(xtx, t[p * p:])
    est = solve(est_t)
    reps = np.array([solve(t) for t in boot_t])
    rows = []
    pos = [i for i, j in enumerate(lags) if j >= 0]
    for i, j in enumerate(lags):
        ci = interval(est[i], reps[:, i])
        cum = [q for q in pos if lags[q] <= j]
        cci = interval(est[cum].sum(), reps[:, cum].sum(axis=1)) \
            if j >= 0 else {'est': np.nan, 'lo': np.nan, 'hi': np.nan}
        rows.append({'lag': j, 'coef': ci['est'], 'lo': ci['lo'],
                     'hi': ci['hi'], 'cum': cci['est'], 'cum_lo': cci['lo'],
                     'cum_hi': cci['hi'], 'n_minutes': len(d),
                     'n_games': d['game_id'].nunique()})
    return pd.DataFrame(rows)


def acf_table(boot, df, cols=('dm', 'dgap'), max_lag=10):
    """
    Pooled within-game autocorrelation sum(x_t x_{t-k}) / sum(x_t^2) of
    each column at lags 1..max_lag, with game bootstrap intervals.
    """
    rows = []
    for col in cols:
        for k in range(1, max_lag + 1):
            d = df[['game_id', col]].copy()
            d['lag'] = d.groupby('game_id')[col].shift(k)
            d = d.dropna()
            d['xy'] = d[col] * d['lag']
            d['xx'] = d[col] ** 2
            ci = boot_ratio(boot, d, 'xy', 'xx')
            rows.append(flat(f'acf_{col}', ci, lag=k, n=len(d)))
    return pd.DataFrame(rows)


def jump_reversal(boot, df, threshold=0.5, horizons=(1, 2, 5, 10)):
    """
    For minutes where the market log-odds jumps by at least threshold
    (pre-jump price in [0.05, 0.95]), the signed market change and
    market-minus-model change over the next k minutes. Splits jumps
    into 'news' (model moved >= 0.25 the same way) and 'no news' (score
    margin unchanged from one minute before to one minute after and
    |model change| < 0.05). Use the next-print alignment so the price
    minute and the score minute line up.
    """
    d = df.copy()
    g = d.groupby('game_id')
    d['p_prev'] = g['price_home'].shift(1)
    d['dmargin'] = (g['margin'].shift(-1) - g['margin'].shift(2)).abs()
    for k in horizons:
        d[f'm_f{k}'] = g['logit_price'].shift(-k) - d['logit_price']
        d[f'f_f{k}'] = g['fair_logit'].shift(-k) - d['fair_logit']
    jumps = d[(d['dm'].abs() >= threshold)
              & d['p_prev'].between(0.05, 0.95)].copy()
    sign = np.sign(jumps['dm'])
    jumps['kind'] = np.where(
        sign * jumps['dfair'] >= 0.25, 'news',
        np.where((jumps['dmargin'] == 0) & (jumps['dfair'].abs() < 0.05),
                 'no news', 'other'))
    rows = []
    for kind in ('all', 'news', 'no news'):
        sub = jumps if kind == 'all' else jumps[jumps['kind'] == kind]
        for k in horizons:
            s = pd.DataFrame({
                'game_id': sub['game_id'],
                'mk': np.sign(sub['dm']) * sub[f'm_f{k}'],
                'gap': np.sign(sub['dm']) * (sub[f'm_f{k}']
                                             - sub[f'f_f{k}'])})
            rows.append(flat('market_after', boot_mean(boot, s, 'mk'),
                             kind=kind, k=k, n_jumps=s['mk'].notna().sum(),
                             n_games=sub['game_id'].nunique(),
                             mean_jump=float(sub['dm'].abs().mean())))
            rows.append(flat('gap_after', boot_mean(boot, s, 'gap'),
                             kind=kind, k=k, n_jumps=s['gap'].notna().sum(),
                             n_games=sub['game_id'].nunique(),
                             mean_jump=float(sub['dm'].abs().mean())))
    return pd.DataFrame(rows)


def favorite_frame(prob, won, game_id):
    """
    Re-expresses home probabilities from the favorite's side: returns a
    DataFrame with game_id, fav_p = max(p, 1 - p) and fav_won.
    """
    prob = np.asarray(prob, dtype=float)
    won = np.asarray(won, dtype=float)
    home_fav = prob >= 0.5
    return pd.DataFrame({
        'game_id': np.asarray(game_id),
        'fav_p': np.where(home_fav, prob, 1 - prob),
        'fav_won': np.where(home_fav, won, 1 - won)})


def calibration_table(boot, fav, label, by=None):
    """
    Calibration of favorite prices: for each price bucket (and level
    of by), n rows and games, mean price, win rate, and win rate minus
    price with a game bootstrap interval. Positive excess means
    favorites won more often than priced (a favorite-longshot bias).
    """
    fav = fav.dropna(subset=['fav_p', 'fav_won']).copy()
    fav['bucket'] = pd.cut(fav['fav_p'], FAV_BINS, right=False)
    fav['excess'] = fav['fav_won'] - fav['fav_p']
    groups = [(None, fav)] if by is None else list(
        fav.groupby(by, observed=True))
    rows = []
    for level, part in groups:
        for bucket, sub in part.groupby('bucket', observed=True):
            if sub['game_id'].nunique() < 10:
                continue
            ci = boot_mean(boot, sub, 'excess')
            rows.append(flat('win_minus_price', ci, sample=label,
                             group=level, bucket=str(bucket), n=len(sub),
                             n_games=sub['game_id'].nunique(),
                             mean_price=sub['fav_p'].mean(),
                             win_rate=sub['fav_won'].mean()))
    return pd.DataFrame(rows)


def ingame_phase(minutes):
    """Labels each game-minute with the same phases used for runs."""
    sec = minutes['sec_remaining']
    return np.select(
        [minutes['period'] <= 2, minutes['period'] == 3,
         (minutes['period'] == 4) & (sec > 300)],
        ['Q1-Q2', 'Q3', 'Q4 >5 min left'], 'Final 5 min / OT')


def price_only_minutes(prices, joined, pbp_games, tip_offset):
    """
    In-game prices for games without play-by-play: price prints from
    tip-off (scheduled start + the median observed offset) to 120 min
    after, which ends before the final buzzer in almost all games.
    """
    info = joined[~joined['game_id'].isin(pbp_games)].copy()
    info['tip'] = pd.to_datetime(info['scheduled_start'], utc=True) + \
        tip_offset
    p = prices.merge(info[['game_id', 'tip', 'home_win']], on='game_id')
    p['t_min'] = (p['ts'] - p['tip']).dt.total_seconds() / 60
    p = p[p['t_min'].between(0, 120)].copy()
    p['half'] = np.where(p['t_min'] < 60, '0-60 min after tip',
                         '60-120 min after tip')
    return p


def break_drift(boot, minutes, skip=2):
    """
    Price changes during halftime, when the game
    state (and so the model) is frozen. The first skip minutes of each
    break are dropped so a late basket that the price has not printed
    yet is not counted. Signed toward the pregame price ('toward
    prior') and toward the team leading at the break. Under a
    martingale both means are zero.
    """
    # Quarter breaks are too short (about 2 minutes on the grid) to
    # survive dropping the first minutes, so only halftime is kept.
    breaks = {'halftime': (2, 1440.0)}
    rows = []
    for name, (period, sec) in breaks.items():
        sel = minutes[(minutes['period'] == period)
                      & (minutes['sec_remaining'] == sec)]
        sel = sel.dropna(subset=['logit_price']).sort_values('minute_ts')
        sel = sel[sel.groupby('game_id').cumcount() >= skip]
        agg = sel.groupby('game_id').agg(
            first=('logit_price', 'first'), last=('logit_price', 'last'),
            n=('logit_price', 'size'), margin=('margin', 'first'),
            pre=('pregame_logit', 'first'))
        agg = agg[agg['n'] >= 2].reset_index()
        change = agg['last'] - agg['first']
        agg['toward_prior'] = np.sign(agg['pre'] - agg['first']) * change
        agg['toward_leader'] = np.sign(agg['margin']) * change
        agg['abs_change'] = change.abs()
        for col in ('toward_prior', 'toward_leader', 'abs_change'):
            rows.append(flat(col, boot_mean(boot, agg, col), brk=name,
                             n_games=len(agg),
                             median_minutes=float(agg['n'].median())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. Pregame pricing
# ---------------------------------------------------------------------

def team_form(games):
    """
    Season-to-date form for each team before each game: mean point
    differential in prior games of the same season (NaN before 5
    games) and days of rest since the team's previous game. Returns one
    row per game with home/away form and rest.
    """
    g = games.copy()
    g['date'] = pd.to_datetime(g['game_date'])
    home = pd.DataFrame({'game_id': g['game_id'], 'date': g['date'],
                         'season': g['season_id'], 'team': g['home'],
                         'diff': g['home_pts'] - g['away_pts'],
                         'side': 'home'})
    away = pd.DataFrame({'game_id': g['game_id'], 'date': g['date'],
                         'season': g['season_id'], 'team': g['away'],
                         'diff': g['away_pts'] - g['home_pts'],
                         'side': 'away'})
    long = pd.concat([home, away]).sort_values(['team', 'date'])
    grp = long.groupby(['team', 'season'])['diff']
    long['form'] = grp.transform(lambda s: s.shift(1).expanding().mean())
    long['n_prior'] = grp.cumcount()
    long.loc[long['n_prior'] < 5, 'form'] = np.nan
    long['rest'] = long.groupby('team')['date'].diff().dt.days.clip(
        upper=7)
    wide = long.pivot(index='game_id', columns='side',
                      values=['form', 'rest'])
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide['form_diff'] = wide['form_home'] - wide['form_away']
    wide['rest_diff'] = wide['rest_home'] - wide['rest_away']
    wide['home_b2b'] = (wide['rest_home'] == 1).astype(float)
    wide['away_b2b'] = (wide['rest_away'] == 1).astype(float)
    return wide.reset_index()


def season_period(row):
    """Labels a game's part of the season."""
    if row['season_type'] != 'Regular Season':
        return 'Playoffs / Play-In'
    date = pd.Timestamp(row['game_date'])
    if date >= pd.Timestamp('2026-09-01'):
        return '2026-27 Oct-Nov'
    if date.month in (10, 11):
        return '2025-26 Oct-Nov'
    if date.month in (12, 1):
        return '2025-26 Dec-Jan'
    return '2025-26 Feb-Apr'


def pregame_table(tables, tip_offset):
    """
    One row per priced game: tip time (play-by-play start, else the
    scheduled start plus the median offset), price at tip and 60 min
    before, outcome, volume, season period and team form.
    """
    joined = tables['joined'].copy()
    sched = pd.to_datetime(joined['scheduled_start'], utc=True)
    joined['tip'] = joined['tip_ts'].fillna(sched + tip_offset)
    by_game = dict(list(tables['prices'].groupby('game_id')))
    rows = []
    for game in joined.to_dict('records'):
        price = by_game.get(game['game_id'])
        if price is None or pd.isna(game['tip']):
            continue
        tip = game['tip']
        early = price[price['ts'] <= tip - pd.Timedelta(minutes=60)]
        rows.append({'game_id': game['game_id'],
                     'p_tip': dp.pregame_price(price, tip),
                     'p_open': early['price'].iloc[-1] if len(early)
                     else price['price'].iloc[0],
                     'open_is_first_print': len(early) == 0})
    pre = joined.merge(pd.DataFrame(rows), on='game_id')
    pre = pre.merge(team_form(tables['games']), on='game_id', how='left')
    pre['period_label'] = pre.apply(season_period, axis=1)
    pre['logit_tip'] = dp.logit(pre['p_tip'])
    pre['logit_open'] = dp.logit(pre['p_open'])
    pre['drift'] = pre['logit_tip'] - pre['logit_open']
    return pre.dropna(subset=['p_tip', 'home_win'])


def boot_logit(boot, df, y, xcols, offset=None, draws=False):
    """
    Logistic regression of y on xcols (plus a constant) refit on every
    game bootstrap draw via frequency weights. Returns ({name:
    interval}, n rows), plus the (B, P) draws if draws is True.
    """
    d = df.dropna(subset=[y] + xcols).set_index('game_id')
    w_all = pd.DataFrame(boot.weights.T, index=boot.games)
    w_all = w_all.reindex(d.index).fillna(0).to_numpy()
    x = sm.add_constant(d[xcols], has_constant='add')
    off = None if offset is None else d[offset].to_numpy()
    fit = sm.GLM(d[y], x, family=sm.families.Binomial(), offset=off).fit()
    reps = []
    for b in range(w_all.shape[1]):
        reps.append(sm.GLM(d[y], x, family=sm.families.Binomial(),
                           offset=off,
                           freq_weights=w_all[:, b]).fit().params.to_numpy())
    reps = np.array(reps)
    out = {name: interval(fit.params[name], reps[:, i])
           for i, name in enumerate(x.columns)}
    if draws:
        return out, len(d), reps
    return out, len(d)


def brier_compare(boot, df, a, b, y='home_win'):
    """Brier of a, of b, and a minus b, with game bootstrap intervals."""
    d = df.dropna(subset=[a, b, y]).copy()
    d['ba'] = (d[a] - d[y]) ** 2
    d['bb'] = (d[b] - d[y]) ** 2
    d['diff'] = d['ba'] - d['bb']
    return {a: boot_mean(boot, d, 'ba'), b: boot_mean(boot, d, 'bb'),
            'diff': boot_mean(boot, d, 'diff'), 'n_games': len(d)}


def pregame_analysis(boot, pre):
    """
    Pregame tests: calibration buckets, logistic recalibration
    (constant = home bias, slope > 1 = favorites underpriced), the
    last-hour drift, team form and rest beyond the price, Brier of the
    open vs tip price, and recalibration by season period.
    """
    out = {}
    fav = favorite_frame(pre['p_tip'], pre['home_win'], pre['game_id'])
    out['calibration'] = calibration_table(boot, fav, 'pregame tip price')
    out['recal'], out['n_recal'] = boot_logit(boot, pre, 'home_win',
                                              ['logit_tip'])
    out['home_bias'], _ = boot_logit(boot, pre, 'home_win', [],
                                     offset='logit_tip')
    out['drift_model'], out['n_drift'] = boot_logit(
        boot, pre[~pre['open_is_first_print']], 'home_win',
        ['logit_tip', 'drift'])
    out['form_model'], out['n_form'] = boot_logit(
        boot, pre, 'home_win', ['logit_tip', 'form_diff'])
    out['rest_model'], out['n_rest'] = boot_logit(
        boot, pre, 'home_win', ['logit_tip', 'rest_diff', 'home_b2b',
                                'away_b2b'])
    out['brier_open_vs_tip'] = brier_compare(boot, pre, 'p_open', 'p_tip')
    drift = pre[['game_id', 'drift']].copy()
    drift['abs_drift'] = drift['drift'].abs()
    out['mean_abs_drift'] = boot_mean(boot, drift, 'abs_drift')
    periods = []
    slope_draws = {}
    for label, part in pre.groupby('period_label'):
        fit, n, reps = boot_logit(boot, part, 'home_win', ['logit_tip'],
                                  draws=True)
        slope_draws[label] = (fit['logit_tip']['est'], reps[:, 1])
        d = part.assign(b=(part['p_tip'] - part['home_win']) ** 2)
        periods.append({'period': label, 'n_games': n,
                        'brier': boot_mean(boot, d, 'b'),
                        'slope': fit['logit_tip'], 'const': fit['const']})
    out['by_period'] = periods
    a, b = slope_draws['2025-26 Feb-Apr'], slope_draws['2025-26 Dec-Jan']
    out['slope_feb_apr_minus_dec_jan'] = interval(a[0] - b[0], a[1] - b[1])
    return out


# ---------------------------------------------------------------------
# 4. Trade-level check of the lag (small sample)
# ---------------------------------------------------------------------

def fetch_game_trades(session, condition_id, tip, max_pages=40):
    """
    Downloads trades for one market from the Polymarket data API (v2,
    newest first, cursor paging) until trades are older than 10 min
    before tip-off. Returns a DataFrame or raises on an API error.
    """
    rows = []
    cursor = None
    stop = (tip - pd.Timedelta(minutes=10)).timestamp()
    for _ in range(max_pages):
        params = {'condition': condition_id, 'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        resp = session.get(TRADES_URL, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        data = body.get('data', [])
        rows.extend({k: t.get(k) for k in ('timestamp', 'token_id',
                                           'price', 'size', 'side')}
                    for t in data)
        page = body.get('pagination', {})
        cursor = page.get('next_cursor')
        time.sleep(0.35)
        if not data or data[-1]['timestamp'] < stop or \
                not page.get('has_more'):
            break
    df = pd.DataFrame(rows, columns=['timestamp', 'token_id', 'price',
                                     'size', 'side'])
    df['ts'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    end = tip + pd.Timedelta(hours=4)
    return df[(df['ts'] >= tip - pd.Timedelta(minutes=10))
              & (df['ts'] <= end)].reset_index(drop=True)


def trade_sample(runs, joined, data_dir, fetch, n_runs=N_TRADE_RUNS):
    """
    Picks a fixed random sample of runs and returns {game_id: trades}
    from data/raw/trades/, downloading missing games when fetch is
    True. Stops downloading after 3 errors in a row.
    """
    pool = runs.dropna(subset=['market_reaction', 'fair_reaction'])
    sample = pool.sample(n=min(n_runs, len(pool)), random_state=SEED)
    cache = os.path.join(data_dir, 'trades')
    os.makedirs(cache, exist_ok=True)
    info = joined.set_index('game_id')
    session = requests.Session()
    session.headers['User-Agent'] = 'cse163-research-script'
    trades = {}
    errors = 0
    for game_id in sample['game_id'].unique():
        path = os.path.join(cache, game_id + '.parquet')
        if os.path.exists(path):
            trades[game_id] = pd.read_parquet(path)
            continue
        if not fetch or errors >= 3:
            continue
        try:
            df = fetch_game_trades(session, info.loc[game_id,
                                                     'condition_id'],
                                   info.loc[game_id, 'tip_ts'])
            df.to_parquet(path, index=False)
            trades[game_id] = df
            errors = 0
        except (requests.RequestException, ValueError) as err:
            errors += 1
            print('trade fetch failed for', game_id, err)
    return sample[sample['game_id'].isin(trades)], trades


def trade_paths(sample, trades, joined, minutes):
    """
    Second-level path of the market around each sampled run from
    trades: home-team price = median of the last 3 trade prices (home
    token, or 1 - away token price) at or before each offset after the
    last basket, signed toward the runner, relative to 60 s before the
    run starts. Also returns the model's full +2 min reaction and the
    minute-snapshot market change for the same runs.
    """
    home_token = joined.set_index('game_id')['home_token']
    by_game = {g: df.set_index('minute_ts')
               for g, df in minutes.groupby('game_id')}
    rows = []
    for run_id, run in sample.iterrows():
        t = trades[run['game_id']].copy()
        t['p_home'] = np.where(t['token_id'] == home_token[run['game_id']],
                               t['price'], 1 - t['price'])
        t = t.sort_values('ts')
        sign = 1 if run['team'] == 'home' else -1

        def price_at(ts):
            before = t[(t['ts'] <= ts)
                       & (t['ts'] > ts - pd.Timedelta(minutes=5))]
            return before['p_home'].tail(3).median() if len(before) \
                else np.nan
        base = price_at(run['start_ts'] - pd.Timedelta(seconds=60))
        game = by_game[run['game_id']]
        t_base = run['start_ts'].floor('min') - pd.Timedelta(minutes=1)
        t_post = run['end_ts'].floor('min') + pd.Timedelta(minutes=2)
        if t_base not in game.index or t_post not in game.index:
            continue
        model = sign * (game.loc[t_post, 'fair_logit']
                        - game.loc[t_base, 'fair_logit'])
        lag_s = (run['end_ts'] - run['end_ts'].floor('min')).total_seconds()
        for s in TRADE_OFFSETS:
            p = price_at(run['end_ts'] + pd.Timedelta(seconds=s))
            rows.append({'run_id': run_id, 'game_id': run['game_id'],
                         'offset_s': s, 'model_full': model,
                         'end_sec_in_minute': lag_s,
                         'market': sign * (dp.logit(p) - dp.logit(base))
                         if pd.notna(p) and pd.notna(base) else np.nan})
        n_after = ((t['ts'] > run['end_ts'])
                   & (t['ts'] <= run['end_ts'] + pd.Timedelta(seconds=60)))
        rows[-1]['trades_first_60s'] = int(n_after.sum())
    return pd.DataFrame(rows)


def trade_passthrough(boot, tp, paths):
    """
    Pass-through of the model's +2 min reaction at each trade offset,
    and at minute-snapshot horizons for the same runs. Uses runs with
    a market value at every offset.
    """
    wide = tp.pivot(index='run_id', columns='offset_s', values='market')
    keep = wide.notna().all(axis=1)
    info = tp.drop_duplicates('run_id').set_index('run_id')
    rows = []
    for s in TRADE_OFFSETS:
        d = pd.DataFrame({'game_id': info.loc[keep[keep].index, 'game_id'],
                          'm': wide.loc[keep, s],
                          'f': info.loc[keep[keep].index, 'model_full']})
        ci = boot_ratio(boot, d, 'm', 'f')
        rows.append({'source': 'trades', 'seconds_after_end': s,
                     'pt': ci['est'], 'lo': ci['lo'], 'hi': ci['hi'],
                     'n_runs': len(d), 'n_games': d['game_id'].nunique()})
    ids = keep[keep].index
    mean_sec = info.loc[ids, 'end_sec_in_minute'].mean()
    for k in range(0, 11):
        d = paths[(paths['k'] == k) & paths['run_id'].isin(ids)].copy()
        d = d.merge(info[['model_full']], left_on='run_id',
                    right_index=True).dropna(subset=['market'])
        ci = boot_ratio(boot, d, 'market', 'model_full')
        rows.append({'source': 'minute snapshots',
                     'seconds_after_end': 60 * k - mean_sec,
                     'pt': ci['est'], 'lo': ci['lo'], 'hi': ci['hi'],
                     'n_runs': len(d), 'n_games': d['game_id'].nunique()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def save_fig(fig, name):
    """Saves a figure to figures/<name>.png and closes it."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + '.png'), dpi=200)
    plt.close(fig)


def plot_passthrough(curves, trade_pt):
    """
    Figure: pass-through of the model's run reaction by minutes after
    the run, for both price alignments, and the trade-level sample.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for label, (curve, color) in curves.items():
        ax1.plot(curve['k'], curve['pt'], marker='o', ms=3, color=color,
                 label=label)
        ax1.fill_between(curve['k'], curve['lo'], curve['hi'],
                         color=color, alpha=0.15)
    ax1.axhline(1, color='#888888', ls='--', lw=1, label='Full pass-through')
    ax1.axvline(0, color='#888888', lw=0.8)
    ax1.set(title='(a) Market move / model move after runs (all runs)',
            xlabel='Minutes after the run ends (minute grid)',
            ylabel='Pass-through (sum market / sum model)',
            ylim=(-0.2, 1.4))
    ax1.legend(fontsize=7, loc='lower right')
    if trade_pt is not None and len(trade_pt):
        for source, color in (('trades', COLORS['trades']),
                              ('minute snapshots', COLORS['market'])):
            d = trade_pt[trade_pt['source'] == source]
            ax2.plot(d['seconds_after_end'], d['pt'], marker='o', ms=3,
                     color=color, label=source)
            ax2.fill_between(d['seconds_after_end'], d['lo'], d['hi'],
                             color=color, alpha=0.15)
        n = int(trade_pt.loc[trade_pt['source'] == 'trades',
                             'n_runs'].max())
        ax2.axhline(1, color='#888888', ls='--', lw=1)
        ax2.axvline(0, color='#888888', lw=0.8)
        ax2.set(title=f'(b) Same runs, trade-level prices ({n} runs)',
                xlabel='Seconds after the last basket (NBA log time)',
                ylabel='Pass-through of the model +2 min move',
                ylim=(-0.4, 1.6))
        ax2.legend(fontsize=7, loc='lower right')
    else:
        ax2.text(0.5, 0.5, 'No cached trades\n(run with --fetch-trades)',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_axis_off()
    save_fig(fig, 'edge_passthrough')


def plot_distributed_lag(dl_main, dl_alt):
    """Figure: distributed-lag coefficients and cumulative pass-through."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    width = 0.38
    for shift, (dl, color, label) in zip(
            (-width / 2, width / 2),
            ((dl_main, COLORS['market'], 'Last print before minute'),
             (dl_alt, COLORS['alt'], 'Next print after minute'))):
        err = np.vstack([dl['coef'] - dl['lo'], dl['hi'] - dl['coef']])
        ax1.bar(dl['lag'] + shift, dl['coef'], width=width, color=color,
                yerr=err, capsize=2, label=label)
        pos = dl[dl['lag'] >= 0]
        ax2.plot(pos['lag'], pos['cum'], marker='o', color=color,
                 label=label)
        ax2.fill_between(pos['lag'], pos['cum_lo'], pos['cum_hi'],
                         color=color, alpha=0.15)
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set(title='(a) Market minute change on model change j min earlier',
            xlabel='Lag j in minutes (negative = model moves later)',
            ylabel='Coefficient (95% CI)')
    ax1.legend(fontsize=7)
    ax2.axhline(1, color='#888888', ls='--', lw=1, label='Full pass-through')
    ax2.set(title='(b) Cumulative pass-through of every model move',
            xlabel='Minutes since the model moved',
            ylabel='Sum of coefficients, lags 0..j', ylim=(0, 1.2))
    ax2.legend(fontsize=7, loc='lower right')
    save_fig(fig, 'edge_distributed_lag')


def plot_calibration(cal, pregame, periods):
    """
    Figure: favorite win rate minus price in-game (two samples and the
    benchmark), at tip-off, and the tip-off recalibration slope by
    season period.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    styles = (('in-game, pbp games', COLORS['market'],
               'Market, 607 pbp games (Oct-Jan)'),
              ('in-game, price-only games', COLORS['alt'],
               'Market, price-only games (0-120 min)'),
              ('benchmark model, held-out games', COLORS['model'],
               'Benchmark model, held-out games'))
    for i, (sample, color, label) in enumerate(styles):
        d = cal[(cal['sample'] == sample) & cal['group'].isna()]
        x = d['mean_price'] + (i - 1) * 0.004
        axes[0].errorbar(x, d['est'], yerr=[d['est'] - d['lo'],
                                            d['hi'] - d['est']],
                         marker='o', ms=4, capsize=2, color=color,
                         label=label)
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].set(title='(a) In-game: favorite win rate minus price',
                xlabel='Favorite price (bucket mean, every game-minute)',
                ylabel='Win rate - price (95% CI)')
    axes[0].legend(fontsize=7, loc='lower left')
    axes[1].errorbar(pregame['mean_price'], pregame['est'],
                     yerr=[pregame['est'] - pregame['lo'],
                           pregame['hi'] - pregame['est']],
                     marker='o', capsize=3, color=COLORS['market'],
                     label='Tip-off price, all priced games')
    for x, n in zip(pregame['mean_price'], pregame['n']):
        axes[1].annotate(f'n={n}', (x, pregame['lo'].min()),
                         textcoords='offset points', xytext=(0, -10),
                         ha='center', fontsize=6)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set(title='(b) Pregame: favorite win rate minus tip price',
                xlabel='Favorite tip-off price (bucket mean)',
                ylabel='Win rate - price (95% CI)')
    axes[1].legend(fontsize=7, loc='upper left')
    order = ['2025-26 Oct-Nov', '2025-26 Dec-Jan', '2025-26 Feb-Apr',
             'Playoffs / Play-In', '2026-27 Oct-Nov']
    per = {p['period']: p for p in periods}
    order = [o for o in order if o in per]
    est = np.array([per[o]['slope']['est'] for o in order])
    lo = np.array([per[o]['slope']['lo'] for o in order])
    hi = np.array([per[o]['slope']['hi'] for o in order])
    axes[2].errorbar(np.arange(len(order)), est, yerr=[est - lo, hi - est],
                     fmt='o', capsize=3, color=COLORS['market'],
                     label='Slope of win on logit(tip price)')
    axes[2].axhline(1, color='#888888', ls='--', lw=1,
                    label='Calibrated (slope 1)')
    axes[2].set_xticks(np.arange(len(order)),
                       [f"{o}\n(n={per[o]['n_games']})" for o in order],
                       fontsize=6)
    axes[2].set(title='(c) Tip-off recalibration slope by season period',
                xlabel='', ylabel='Slope (>1: favorites underpriced)')
    axes[2].legend(fontsize=7, loc='upper left')
    save_fig(fig, 'edge_calibration')


def plot_context(ctx, outcome):
    """
    Figure: run pass-through at +10 min relative to the benchmark by
    runner pregame role and score state, next to the outcome-based
    check (runner win rate minus market price 2 min after the run).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    rows = ctx[ctx['split'].isin(['runner_pregame', 'score_state',
                                  'timeout_after'])].reset_index(drop=True)
    y = np.arange(len(rows))
    err = np.vstack([rows['pt10'] - rows['pt10_lo'],
                     rows['pt10_hi'] - rows['pt10']])
    axes[0].errorbar(rows['pt10'], y, xerr=err, fmt='o', capsize=2,
                     color=COLORS['market'], label='Market / model, +10 min')
    axes[0].axvline(1, color='#888888', ls='--', lw=1,
                    label='Market moves as much as model')
    axes[0].set_yticks(y, [f'{lv} (n={n})' for lv, n in
                           zip(rows['level'], rows['n_runs'])], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set(title='(a) Pass-through vs. benchmark after runs',
                xlabel='Sum market change / sum model change',
                ylabel='')
    axes[0].legend(fontsize=7, loc='lower right')
    out = outcome[(outcome['k'] == 2) & outcome['split'].isin(
        ['runner_pregame', 'score_state', 'timeout_after'])]
    out = rows[['split', 'level']].merge(out, on=['split', 'level'])
    for shift, col, color, label in (
            (-0.15, 'mkt', COLORS['market'], 'Market price'),
            (0.15, 'model', COLORS['model'], 'Benchmark model')):
        axes[1].errorbar(out[col], y + shift,
                         xerr=[out[col] - out[col + '_lo'],
                               out[col + '_hi'] - out[col]],
                         fmt='o', ms=4, capsize=2, color=color, label=label)
    axes[1].axvline(0, color='#888888', ls='--', lw=1)
    axes[1].set_yticks(y, [''] * len(y))
    axes[1].invert_yaxis()
    axes[1].set(title='(b) Outcome check: runner win rate - price (+2 min)',
                xlabel='Win rate minus probability (<0: runner overpriced)',
                ylabel='')
    axes[1].legend(fontsize=7, loc='lower right')
    save_fig(fig, 'edge_context')


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def to_jsonable(obj):
    """Converts numpy / pandas objects so json.dump can write them."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join('data', 'raw'))
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--fetch-trades', action='store_true')
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 30)
    start = time.time()
    tables = load_inputs(args.data)
    minutes = tables['minutes']
    joined = tables['joined']
    runs = label_contexts(tables['runs'], tables['pbp'])
    runs['home_win'] = runs['game_id'].map(
        joined.set_index('game_id')['home_win'])
    boot = GameBoot(minutes['game_id'], args.n_boot)
    numbers = {'n_games_pbp': int(minutes['game_id'].nunique()),
               'n_runs': int(len(runs)), 'n_boot': args.n_boot}
    print(f'Loaded {numbers["n_games_pbp"]} games, {len(runs)} runs '
          f'({time.time() - start:.0f}s)')

    # 1. Runs ---------------------------------------------------------
    paths = run_paths(runs, minutes)
    runs['p_pre'] = pre_run_price(runs, minutes)
    alt_minutes = realign_minutes(minutes, tables['prices'])
    alt_paths = run_paths(runs, alt_minutes)
    curve, draws = passthrough_curve(boot, paths)
    alt_curve, alt_draws = passthrough_curve(boot, alt_paths)
    prob_curve, _ = passthrough_curve(boot, paths, market='market_p',
                                      model='model_p')
    numbers['passthrough_curve'] = curve
    numbers['passthrough_curve_alt_alignment'] = alt_curve
    numbers['passthrough_curve_prob_points'] = prob_curve
    numbers['minutes_to_close'] = minutes_to_close(curve, draws)
    numbers['minutes_to_close_alt'] = minutes_to_close(alt_curve, alt_draws)
    numbers['slope_by_horizon'] = slope_by_horizon(boot, paths)
    numbers['slope_by_horizon_alt'] = slope_by_horizon(boot, alt_paths)
    att = attenuation(boot, runs, paths)
    numbers['attenuation'] = att
    ctx = context_table(boot, runs, paths)
    ctx.to_csv(os.path.join(RES_DIR, 'edge_run_context.csv'), index=False)
    numbers['context'] = ctx
    early = runs.index[runs['phase'].isin(['Q1-Q2'])]
    late = runs.index[runs['phase'].isin(['Final 5 min / OT'])]
    hi_vol = runs.index[runs['volume_tercile'] == 'high']
    lo_vol = runs.index[runs['volume_tercile'] == 'low']
    numbers['contrasts'] = {
        'pt1 final5/OT minus Q1-Q2': contrast(boot, paths, late, early, 1),
        'pt10 final5/OT minus Q1-Q2': contrast(boot, paths, late, early,
                                               10),
        'pt1 high minus low volume': contrast(boot, paths, hi_vol, lo_vol,
                                              1),
        'pt10 high minus low volume': contrast(boot, paths, hi_vol, lo_vol,
                                               10)}
    longshot = runs.index[runs['runner_pregame'] == 'longshot <0.30']
    heavy = runs.index[runs['runner_pregame'] == 'heavy fav >0.70']
    numbers['contrasts']['pt10 longshot minus heavy-fav runner'] = \
        contrast(boot, paths, longshot, heavy, 10)
    trailing = runs.index[runs['score_state'] == 'trailing 10+']
    leading = runs.index[runs['score_state'] == 'leading 10+']
    numbers['contrasts']['pt10 trailing 10+ minus leading 10+'] = \
        contrast(boot, paths, trailing, leading, 10)
    timeout = runs.index[runs['timeout_after'] == 'timeout <=90 s']
    no_timeout = runs.index[runs['timeout_after'] == 'no timeout']
    for k in (1, 2, 3, 5, 10):
        numbers['contrasts'][f'pt{k} timeout minus none (next print)'] = \
            contrast(boot, alt_paths, timeout, no_timeout, k)
    print('\n=== Run pass-through PT(k) (eda alignment) ===')
    print(curve.round(3).to_string(index=False))
    print('\n=== Run pass-through PT(k) (next-print alignment) ===')
    print(alt_curve.round(3).to_string(index=False))
    print('\nMinutes to close:', numbers['minutes_to_close'])
    print('Minutes to close (alt):', numbers['minutes_to_close_alt'])
    print('\n=== Slope by horizon ===')
    print(numbers['slope_by_horizon'].round(3).to_string(index=False))
    print(numbers['slope_by_horizon_alt'].round(3).to_string(index=False))
    print('\n=== Attenuation ===')
    print(att.round(3).to_string(index=False))
    print('\n=== Context ===')
    print(ctx.round(3).to_string(index=False))
    print('\nContrasts:', json.dumps(numbers['contrasts'], indent=1))
    outcome = run_outcome_test(boot, runs, alt_minutes)
    outcome.to_csv(os.path.join(RES_DIR, 'edge_run_outcomes.csv'),
                   index=False)
    numbers['run_outcomes'] = outcome
    print('\n=== Run outcome test (win rate - price after run) ===')
    print(outcome.round(3).to_string(index=False))
    pb = phase_brier(boot, minutes)
    numbers['phase_brier'] = pb
    print('\n=== Held-out Brier by phase ===')
    print(pb.round(4).to_string(index=False))
    print(f'({time.time() - start:.0f}s)')

    # 2. In-game ------------------------------------------------------
    ch = minute_changes(minutes)
    alt_ch = minute_changes(alt_minutes)
    dl = distributed_lag(boot, ch)
    dl_alt = distributed_lag(boot, alt_ch)
    vol = joined.set_index('game_id')['volume_tercile']
    dl_split = {}
    for name, mask in (
            ('final 5 min / OT', (ch['period'] >= 4)
             & (ch['sec_remaining'] <= 300)),
            ('before final 5 min', ~((ch['period'] >= 4)
                                     & (ch['sec_remaining'] <= 300))),
            ('high volume', ch['game_id'].map(vol) == 'high'),
            ('low volume', ch['game_id'].map(vol) == 'low')):
        dl_split[name] = distributed_lag(boot, ch[mask])
    numbers['distributed_lag'] = dl
    numbers['distributed_lag_alt'] = dl_alt
    numbers['distributed_lag_splits'] = dl_split
    print('\n=== Distributed lag (eda alignment) ===')
    print(dl.round(3).to_string(index=False))
    print('\n=== Distributed lag (next-print alignment) ===')
    print(dl_alt.round(3).to_string(index=False))
    for name, d in dl_split.items():
        print('\n---', name)
        print(d.round(3).to_string(index=False))
    acf = pd.concat([acf_table(boot, ch).assign(alignment='eda'),
                     acf_table(boot, alt_ch).assign(alignment='next')],
                    ignore_index=True)
    numbers['acf'] = acf
    print('\n=== ACF ===')
    print(acf.round(3).to_string(index=False))
    jumps = jump_reversal(boot, alt_ch)
    numbers['jumps'] = jumps
    print('\n=== Jumps ===')
    print(jumps.round(3).to_string(index=False))
    fav_in = favorite_frame(minutes['price_home'], minutes['home_win'],
                            minutes['game_id'])
    fav_in['phase'] = ingame_phase(minutes)
    cal_in = calibration_table(boot, fav_in, 'in-game, pbp games',
                               by='phase')
    cal_in_all = calibration_table(boot, fav_in, 'in-game, pbp games')
    fav_model = favorite_frame(minutes['fair_prob'], minutes['home_win'],
                               minutes['game_id'])
    test = ~minutes['game_id'].map(eda.is_train_game)
    cal_model = calibration_table(boot, fav_model[test.to_numpy()],
                                  'benchmark model, held-out games')
    tip_offset = (joined['tip_ts'] - pd.to_datetime(
        joined['scheduled_start'], utc=True)).median()
    po = price_only_minutes(tables['prices'], joined,
                            set(minutes['game_id']), tip_offset)
    boot_all = GameBoot(joined['game_id'], args.n_boot)
    fav_po = favorite_frame(po['price'], po['home_win'], po['game_id'])
    fav_po['half'] = po['half'].to_numpy()
    cal_po = calibration_table(boot_all, fav_po,
                               'in-game, price-only games', by='half')
    cal_po_all = calibration_table(boot_all, fav_po,
                                   'in-game, price-only games')
    cal = pd.concat([cal_in, cal_in_all, cal_model, cal_po, cal_po_all],
                    ignore_index=True)
    cal.to_csv(os.path.join(RES_DIR, 'edge_calibration.csv'), index=False)
    numbers['ingame_calibration'] = cal
    numbers['n_price_only_games'] = int(po['game_id'].nunique())
    print('\n=== In-game calibration ===')
    print(cal.round(4).to_string(index=False))
    brk = break_drift(boot, alt_minutes)
    numbers['breaks'] = brk
    print('\n=== Breaks ===')
    print(brk.round(4).to_string(index=False))
    print(f'({time.time() - start:.0f}s)')

    # 3. Pregame ------------------------------------------------------
    pre = pregame_table(tables, tip_offset)
    boot_pre = GameBoot(pre['game_id'], args.n_boot)
    pg = pregame_analysis(boot_pre, pre)
    numbers['pregame'] = pg
    numbers['n_pregame_games'] = int(len(pre))
    numbers['tip_offset_min'] = tip_offset.total_seconds() / 60
    print('\n=== Pregame ===')
    print(pg['calibration'].round(4).to_string(index=False))
    for key in ('recal', 'home_bias', 'drift_model', 'form_model',
                'rest_model', 'brier_open_vs_tip', 'mean_abs_drift',
                'slope_feb_apr_minus_dec_jan'):
        print(key, json.dumps(to_jsonable(pg[key]), indent=None))
    for row in pg['by_period']:
        print(json.dumps(to_jsonable(row)))
    print(f'({time.time() - start:.0f}s)')

    # 4. Trades -------------------------------------------------------
    sample, trades = trade_sample(runs, joined, args.data,
                                  args.fetch_trades)
    trade_pt = None
    if len(sample):
        tp = trade_paths(sample, trades, joined, minutes)
        trade_pt = trade_passthrough(boot, tp, paths)
        numbers['trade_passthrough'] = trade_pt
        numbers['trades_first_60s_median'] = float(
            tp['trades_first_60s'].dropna().median())
        numbers['n_trade_runs_sampled'] = int(len(sample))
        print('\n=== Trade-level pass-through ===')
        print(trade_pt.round(3).to_string(index=False))
        print('median trades in first 60 s after run:',
              numbers['trades_first_60s_median'])
    else:
        print('\nNo cached trades; run with --fetch-trades for section 4.')

    # Figures and numbers ---------------------------------------------
    plot_passthrough({'Last print before minute (eda.py)':
                      (curve, COLORS['market']),
                      'Next print after minute': (alt_curve,
                                                  COLORS['alt'])},
                     trade_pt)
    plot_distributed_lag(dl, dl_alt)
    plot_calibration(cal, pg['calibration'], pg['by_period'])
    plot_context(ctx, outcome)
    with open(os.path.join(RES_DIR, 'edge_numbers.json'), 'w') as f:
        json.dump(to_jsonable(numbers), f, indent=1, default=str)
    print(f'\nDone in {time.time() - start:.0f}s. Wrote results/edge_*.csv,'
          ' results/edge_numbers.json and figures/edge_*.png')


if __name__ == '__main__':
    main()
