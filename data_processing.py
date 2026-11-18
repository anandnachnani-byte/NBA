"""
Anand Nachnani
CSE 163 Final Project: Overreaction on the Court?

Parsing, matching, alignment, and run-detection functions shared by
fetch_data.py and eda.py. Every function here is pure (no network or
file access) so it can be tested on small hand-built inputs in
test_eda.py.
"""
import json
import re

import numpy as np
import pandas as pd

# NBA tricode -> the team nickname Polymarket uses as the outcome name.
TEAM_NICKNAMES = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks',
    'DEN': 'Nuggets', 'DET': 'Pistons', 'GSW': 'Warriors',
    'HOU': 'Rockets', 'IND': 'Pacers', 'LAC': 'Clippers',
    'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat', 'MIL': 'Bucks',
    'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs',
    'TOR': 'Raptors', 'UTA': 'Jazz', 'WAS': 'Wizards',
}
NICKNAME_TO_TRICODE = {v: k for k, v in TEAM_NICKNAMES.items()}

GAME_SLUG = re.compile(r'^nba-([a-z]{3})-([a-z]{3})-(\d{4}-\d{2}-\d{2})$')
CLOCK = re.compile(r'^PT(\d+)M(\d+(?:\.\d+)?)S$')
REG_PERIOD_S = 720
OT_PERIOD_S = 300


def parse_json_field(value):
    """
    Returns the list stored in a Gamma field that may be a JSON-encoded
    string (e.g. clobTokenIds = '["123", "456"]') or already a list.
    Returns an empty list for missing values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return list(json.loads(value))


def parse_event_slug(slug):
    """
    Parses a Polymarket game slug such as 'nba-gsw-lal-2025-10-21' into
    (first_code, second_code, date_string) with upper-case codes.
    Returns None for slugs that are not single-game NBA events
    (props, futures, exact-margin events, ...).
    """
    match = GAME_SLUG.match(slug or '')
    if match is None:
        return None
    first, second, date = match.groups()
    return first.upper(), second.upper(), date


def extract_moneyline(event):
    """
    Takes one Gamma event dict and returns a flat dict describing its
    moneyline market, or None if the event has no moneyline market or
    its slug is not a single NBA game.
    """
    parsed = parse_event_slug(event.get('slug'))
    if parsed is None:
        return None
    for market in event.get('markets') or []:
        if market.get('sportsMarketType') != 'moneyline':
            continue
        outcomes = parse_json_field(market.get('outcomes'))
        tokens = parse_json_field(market.get('clobTokenIds'))
        if len(outcomes) != 2 or len(tokens) != 2:
            return None
        return {
            'slug': event['slug'],
            'event_id': str(event.get('id')),
            'slug_code_1': parsed[0],
            'slug_code_2': parsed[1],
            'slug_date': parsed[2],
            'condition_id': market.get('conditionId'),
            'question': market.get('question'),
            'outcome_0': outcomes[0],
            'outcome_1': outcomes[1],
            'token_0': str(tokens[0]),
            'token_1': str(tokens[1]),
            'market_volume': _to_float(market.get('volume')),
            'event_volume': _to_float(event.get('volume')),
            'scheduled_start': market.get('gameStartTime')
            or event.get('startTime'),
        }
    return None


def _to_float(value):
    """Converts a Gamma numeric field (number or string) to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def games_from_finder(finder):
    """
    Collapses a LeagueGameFinder table (one row per team per game) into
    one row per game with columns game_id, game_date, home, away,
    home_pts, away_pts, home_win, season_id. 'BOS vs. ORL' marks BOS
    as home and 'ORL @ BOS' marks ORL as away.
    """
    df = finder.copy()
    df['is_home'] = df['MATCHUP'].str.contains('vs.', regex=False)
    home = df[df['is_home']].set_index('GAME_ID')
    away = df[~df['is_home']].set_index('GAME_ID')
    ids = home.index.intersection(away.index)
    games = pd.DataFrame({
        'game_id': ids,
        'game_date': home.loc[ids, 'GAME_DATE'].values,
        'season_id': home.loc[ids, 'SEASON_ID'].values,
        'home': home.loc[ids, 'TEAM_ABBREVIATION'].values,
        'away': away.loc[ids, 'TEAM_ABBREVIATION'].values,
        'home_pts': home.loc[ids, 'PTS'].values,
        'away_pts': away.loc[ids, 'PTS'].values,
    })
    games['home_win'] = (games['home_pts'] > games['away_pts']).astype(int)
    return games.sort_values(['game_date', 'game_id']).reset_index(
        drop=True)


def match_markets_to_games(markets, games):
    """
    Joins moneyline markets to NBA games. Both outcome nicknames are
    mapped to tricodes and a market matches the game with the same two
    teams whose date is within one day of the slug date (the slug uses
    the US Eastern date); an exact date match wins over a one-day
    match so back-to-back games between the same teams stay apart.
    Returns (matched, unmatched); matched has the
    market columns plus game_id, home, away, home_win and home_token,
    and unmatched has a 'reason' column. Nothing is fixed by hand.
    """
    rows = []
    bad = []
    by_teams = {}
    for game in games.itertuples(index=False):
        key = frozenset([game.home, game.away])
        by_teams.setdefault(key, []).append(game)
    for market in markets.to_dict('records'):
        code_0 = NICKNAME_TO_TRICODE.get(market['outcome_0'])
        code_1 = NICKNAME_TO_TRICODE.get(market['outcome_1'])
        if code_0 is None or code_1 is None:
            bad.append({**market, 'reason': 'unknown team name'})
            continue
        slug_date = pd.Timestamp(market['slug_date'])
        same_teams = by_teams.get(frozenset([code_0, code_1]), [])
        candidates = [g for g in same_teams
                      if pd.Timestamp(g.game_date) == slug_date]
        if not candidates:
            candidates = [
                g for g in same_teams
                if abs((pd.Timestamp(g.game_date) - slug_date).days) <= 1
            ]
        if len(candidates) != 1:
            reason = 'no game found' if not candidates else 'ambiguous'
            bad.append({**market, 'reason': reason})
            continue
        game = candidates[0]
        home_is_0 = code_0 == game.home
        rows.append({
            **market,
            'game_id': game.game_id,
            'game_date': game.game_date,
            'home': game.home,
            'away': game.away,
            'home_win': game.home_win,
            'home_token': market['token_0'] if home_is_0
            else market['token_1'],
            'home_is_outcome_0': home_is_0,
        })
    matched = pd.DataFrame(rows)
    if not matched.empty:
        # Keep the busiest market when two markets map to one game.
        matched = matched.sort_values('market_volume', ascending=False)
        dup = matched['game_id'].duplicated()
        for market in matched[dup].to_dict('records'):
            bad.append({**market, 'reason': 'duplicate market for game'})
        matched = matched[~dup].sort_values('game_date')
        matched = matched.reset_index(drop=True)
    return matched, pd.DataFrame(bad)


def parse_clock(clock):
    """
    Converts an ISO-8601 game clock such as 'PT11M32.00S' to the
    number of seconds left in the period (692.0).
    """
    match = CLOCK.match(clock or '')
    if match is None:
        raise ValueError('bad clock: ' + str(clock))
    return int(match.group(1)) * 60 + float(match.group(2))


def seconds_remaining(period, clock_s):
    """
    Returns seconds left in regulation for periods 1-4. In overtime it
    returns the seconds left in that overtime period.
    """
    if period <= 4:
        return (4 - period) * REG_PERIOD_S + clock_s
    return clock_s


def seconds_elapsed(period, clock_s):
    """Returns total game seconds played, counting overtime periods."""
    if period <= 4:
        return (period - 1) * REG_PERIOD_S + (REG_PERIOD_S - clock_s)
    return (4 * REG_PERIOD_S + (period - 5) * OT_PERIOD_S
            + (OT_PERIOD_S - clock_s))


def clean_pbp(raw):
    """
    Cleans one game's raw play-by-play table. Keeps rows with a real
    timestamp, parses the clock, sorts rows into game-time order, makes
    scores integers that never decrease, and adds ts (UTC), clock_s,
    sec_remaining, elapsed and margin (home - away).
    """
    df = raw.copy()
    df['ts'] = pd.to_datetime(df['timeActual'], utc=True,
                              errors='coerce', format='ISO8601')
    df = df.dropna(subset=['ts'])
    df['period'] = df['period'].astype(int)
    df['clock_s'] = df['clock'].map(parse_clock)
    df['sec_remaining'] = [
        seconds_remaining(p, c) for p, c in zip(df['period'], df['clock_s'])
    ]
    df['elapsed'] = [
        seconds_elapsed(p, c) for p, c in zip(df['period'], df['clock_s'])
    ]
    # The feed appends late corrections with a high actionNumber, so
    # order by game time; actionNumber only breaks ties.
    df = df.sort_values(['elapsed', 'actionNumber']).reset_index(drop=True)
    for col in ('scoreHome', 'scoreAway'):
        score = pd.to_numeric(df[col], errors='coerce').ffill().fillna(0)
        # A score never goes down in game order; a dip is a stale row.
        df[col] = score.cummax().astype(int)
    df['margin'] = df['scoreHome'] - df['scoreAway']
    return df


def scoring_events(pbp):
    """
    Returns one row per scoring change in a cleaned play-by-play table,
    with columns ts, period, elapsed, sec_remaining, team ('home' or
    'away'), pts, and margin after the basket.
    """
    home_diff = pbp['scoreHome'].diff().fillna(pbp['scoreHome'])
    away_diff = pbp['scoreAway'].diff().fillna(pbp['scoreAway'])
    events = []
    for i in range(len(pbp)):
        for team, diff in (('home', home_diff.iloc[i]),
                           ('away', away_diff.iloc[i])):
            if diff > 0:
                row = pbp.iloc[i]
                events.append({
                    'ts': row['ts'], 'period': row['period'],
                    'elapsed': row['elapsed'],
                    'sec_remaining': row['sec_remaining'],
                    'team': team, 'pts': int(diff),
                    'margin': row['margin'],
                })
    return pd.DataFrame(events, columns=['ts', 'period', 'elapsed',
                                         'sec_remaining', 'team', 'pts',
                                         'margin'])


def detect_runs(pbp, min_pts=8, window_s=180):
    """
    Finds scoring runs in a cleaned play-by-play table: at least
    min_pts unanswered points by one team scored within window_s game
    seconds (first to last basket). Each unanswered streak gives at
    most one run, using the earliest window with the most points.
    Returns a DataFrame with team, run_pts, start_ts, end_ts,
    start_elapsed, end_elapsed, period, sec_remaining, and
    margin_before / margin_after (home - away).
    """
    events = scoring_events(pbp)
    runs = []
    streak_start = 0
    for i in range(1, len(events) + 1):
        if i < len(events) and events['team'].iloc[i] == \
                events['team'].iloc[streak_start]:
            continue
        streak = events.iloc[streak_start:i]
        best = _best_window(streak, window_s)
        if best is not None and best[2] >= min_pts:
            first = streak.iloc[best[0]]
            last = streak.iloc[best[1]]
            runs.append({
                'team': first['team'],
                'run_pts': best[2],
                'start_ts': first['ts'],
                'end_ts': last['ts'],
                'start_elapsed': first['elapsed'],
                'end_elapsed': last['elapsed'],
                'period': int(last['period']),
                'sec_remaining': last['sec_remaining'],
                'margin_before': first['margin'] - _signed(first),
                'margin_after': last['margin'],
            })
        streak_start = i
    return pd.DataFrame(runs, columns=[
        'team', 'run_pts', 'start_ts', 'end_ts', 'start_elapsed',
        'end_elapsed', 'period', 'sec_remaining', 'margin_before',
        'margin_after'])


def _signed(event):
    """Returns the basket's effect on the home margin."""
    return event['pts'] if event['team'] == 'home' else -event['pts']


def _best_window(streak, window_s):
    """
    Returns (start_index, end_index, points) for the earliest window of
    at most window_s game seconds with the most points inside one
    unanswered streak, or None for an empty streak.
    """
    if len(streak) == 0:
        return None
    elapsed = streak['elapsed'].to_numpy()
    pts = streak['pts'].to_numpy()
    best = None
    for start in range(len(streak)):
        total = 0
        for end in range(start, len(streak)):
            if elapsed[end] - elapsed[start] > window_s:
                break
            total += pts[end]
            if best is None or total > best[2]:
                best = (start, end, int(total))
    return best


def price_frame(history):
    """
    Converts a CLOB prices-history list [{'t': unix, 'p': price}, ...]
    to a DataFrame with ts (UTC) and price, sorted and de-duplicated.
    """
    df = pd.DataFrame(history, columns=['t', 'p'])
    df['ts'] = pd.to_datetime(df['t'], unit='s', utc=True)
    df = df.rename(columns={'p': 'price'})[['ts', 'price']]
    df = df.drop_duplicates('ts', keep='last')
    return df.sort_values('ts').reset_index(drop=True)


def logit(p, eps=0.01):
    """Log-odds of a probability, clipped to [eps, 1 - eps]."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_game_minutes(pbp, prices, max_gap='3min'):
    """
    Lines up one game's cleaned play-by-play and home-team price series
    on real-world time. Builds a one-minute grid from the first to the
    last play, takes the latest game state at or before each minute,
    and takes the latest price no older than max_gap (otherwise NaN).
    Returns columns minute_ts, period, sec_remaining, elapsed, margin,
    price_home and logit_price.
    """
    start = pbp['ts'].min().floor('min')
    end = pbp['ts'].max().ceil('min')
    grid = pd.DataFrame({'minute_ts': pd.date_range(start, end,
                                                    freq='min')})
    # timeActual is not always increasing in actionNumber order (the
    # feed logs corrections later), so sort by real time first.
    state = pbp[['ts', 'period', 'sec_remaining', 'elapsed', 'margin']]
    state = state.sort_values('ts', kind='stable')
    out = pd.merge_asof(grid, state, left_on='minute_ts', right_on='ts',
                        direction='backward').drop(columns='ts')
    out = out.dropna(subset=['period'])
    price = prices.rename(columns={'price': 'price_home'})
    out = pd.merge_asof(out, price, left_on='minute_ts', right_on='ts',
                        direction='backward',
                        tolerance=pd.Timedelta(max_gap)).drop(columns='ts')
    out['period'] = out['period'].astype(int)
    out['logit_price'] = logit(out['price_home'])
    out.loc[out['price_home'].isna(), 'logit_price'] = np.nan
    return out.reset_index(drop=True)


def pregame_price(prices, tip_ts):
    """
    Returns the last home price at or before tip-off, or NaN if the
    series has no price before tip-off.
    """
    before = prices[prices['ts'] <= tip_ts]
    if before.empty:
        return np.nan
    return float(before['price'].iloc[-1])


def value_at(series_df, ts, col, ts_col='minute_ts'):
    """
    Returns the value of col in the last row with ts_col at or before
    ts, or NaN if there is none.
    """
    before = series_df[series_df[ts_col] <= ts]
    if before.empty:
        return np.nan
    return before[col].iloc[-1]


def seven_number_summary(values):
    """
    Returns a Series with mean, std, min, 25%, 50%, 75% and max of the
    non-missing values.
    """
    s = pd.Series(values, dtype=float).dropna()
    return pd.Series({
        'mean': s.mean(), 'std': s.std(), 'min': s.min(),
        '25%': s.quantile(0.25), '50%': s.median(),
        '75%': s.quantile(0.75), 'max': s.max(),
    })
