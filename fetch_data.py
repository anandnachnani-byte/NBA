"""
Anand Nachnani
CSE 163 Final Project: Overreaction on the Court?

Downloads and caches every raw input for the project into data/raw/:
  nba_games.parquet       one row per NBA game (LeagueGameFinder)
  gamma_markets.parquet   one row per Polymarket NBA moneyline market
  matched.parquet         markets joined to NBA game ids
  unmatched.parquet       markets that could not be joined, with reason
  prices/<game_id>.parquet  minute prices of the home-team token
  pbp/<game_id>.parquet     play-by-play for the game
  fetch_log.csv           every failed request

Usage:
  python fetch_data.py               # everything (resumes from cache)
  python fetch_data.py --limit 5     # smoke test on 5 matched games
  python fetch_data.py --sample 10   # copy 10 games to data/sample/
"""
import argparse
import json
import os
import shutil
import time

import pandas as pd
import requests

import data_processing as dp

RAW = os.path.join('data', 'raw')
SAMPLE = os.path.join('data', 'sample')
GAMMA = 'https://gamma-api.polymarket.com/events'
CLOB = 'https://clob.polymarket.com/prices-history'
TRADES = 'https://data-api.polymarket.com/trades'
SEASONS = ['2025-26', '2026-27']
SEASON_TYPES = ['Regular Season', 'PlayIn', 'Playoffs']
FIRST_DATE = '2025-10-01'
LAST_DATE = '2026-11-17'
# One known game per season; their 'series' ids give the NBA catalogs.
SEED_SLUGS = ['nba-gsw-lal-2025-10-21', 'nba-nyk-tor-2026-11-17']
MIN_PRICE_POINTS = 60
PBP_COLUMNS = ['actionNumber', 'clock', 'timeActual', 'period',
               'actionType', 'subType', 'teamTricode', 'scoreHome',
               'scoreAway', 'description', 'orderNumber']

SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'cse163-research-client'


def log_failure(kind, key, error):
    """Appends one failed request to data/raw/fetch_log.csv."""
    path = os.path.join(RAW, 'fetch_log.csv')
    row = pd.DataFrame([{'time': pd.Timestamp.now(tz='UTC'),
                         'kind': kind, 'key': key, 'error': str(error)}])
    row.to_csv(path, mode='a', header=not os.path.exists(path),
               index=False)


def get_json(url, params, tries=4, pause=0.15):
    """
    GETs a JSON endpoint with retries and exponential backoff. Waits
    `pause` seconds after every successful call to stay polite.
    """
    for attempt in range(tries):
        try:
            resp = SESSION.get(url, params=params, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError('status ' + str(resp.status_code))
            resp.raise_for_status()
            time.sleep(pause)
            return json.loads(resp.text, strict=False)
        except (requests.RequestException, ValueError) as error:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
            last = error
    raise RuntimeError(last)


def fetch_games():
    """Fetches the NBA game list for every season and season type."""
    path = os.path.join(RAW, 'nba_games.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    from nba_api.stats.endpoints import leaguegamefinder
    frames = []
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, league_id_nullable='00',
                season_type_nullable=season_type, timeout=60)
            df = finder.get_data_frames()[0]
            print(f'  {season} {season_type}: {len(df)} team-games')
            df['season_type'] = season_type
            if not df.empty:
                frames.append(df)
            time.sleep(1)
    finder = pd.concat(frames, ignore_index=True)
    games = dp.games_from_finder(finder)
    types = finder.drop_duplicates('GAME_ID').set_index('GAME_ID')
    games['season_type'] = games['game_id'].map(types['season_type'])
    games = games[(games['game_date'] >= FIRST_DATE)
                  & (games['game_date'] <= LAST_DATE)]
    games.to_parquet(path, index=False)
    return games


def fetch_gamma():
    """
    Pages through the closed events of each NBA series on the Gamma API
    and keeps the moneyline market of every single-game event.
    """
    path = os.path.join(RAW, 'gamma_markets.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    series_ids = set()
    for slug in SEED_SLUGS:
        for event in get_json(GAMMA, {'slug': slug}):
            for series in event.get('series') or []:
                series_ids.add(series['id'])
    print('  series ids:', sorted(series_ids))
    rows = {}
    n_events = 0
    for series_id in sorted(series_ids):
        offset = 0
        while True:
            page = get_json(GAMMA, {'series_id': series_id,
                                    'closed': 'true', 'limit': 100,
                                    'offset': offset})
            if not isinstance(page, list) or not page:
                break
            n_events += len(page)
            for event in page:
                row = dp.extract_moneyline(event)
                if row is not None:
                    rows[row['slug']] = row
            offset += 100
        print(f'  series {series_id}: {n_events} events so far, '
              f'{len(rows)} moneyline games')
    markets = pd.DataFrame(list(rows.values()))
    markets = markets[(markets['slug_date'] >= FIRST_DATE)
                      & (markets['slug_date'] <= LAST_DATE)]
    markets.to_parquet(path, index=False)
    return markets


def match(markets, games):
    """Joins markets to games and saves matched and unmatched tables."""
    matched, unmatched = dp.match_markets_to_games(markets, games)
    matched.to_parquet(os.path.join(RAW, 'matched.parquet'), index=False)
    unmatched.to_parquet(os.path.join(RAW, 'unmatched.parquet'),
                         index=False)
    print(f'  matched {len(matched)} markets, {len(unmatched)} unmatched')
    return matched


def fetch_prices(row):
    """
    Saves minute prices for the home token from 1 hour before the
    scheduled start to 4 hours after. Falls back to rebuilding minute
    prices from trades if the history is too sparse.
    """
    path = os.path.join(RAW, 'prices', row['game_id'] + '.parquet')
    if os.path.exists(path):
        return
    tip = int(pd.Timestamp(row['scheduled_start']).timestamp())
    data = get_json(CLOB, {'market': row['home_token'],
                           'startTs': tip - 3600,
                           'endTs': tip + 4 * 3600, 'fidelity': 1})
    prices = dp.price_frame(data.get('history', []))
    prices['source'] = 'prices-history'
    if len(prices) < MIN_PRICE_POINTS:
        trades = fetch_trade_prices(row, tip)
        if len(trades) > len(prices):
            prices = trades
    prices.to_parquet(path, index=False)


def fetch_trade_prices(row, tip):
    """
    Rebuilds a minute price series for the home token from public
    trades: the last trade price in each minute, converted to the home
    side.
    """
    trades = []
    offset = 0
    while offset < 10000:
        page = get_json(TRADES, {'market': row['condition_id'],
                                 'limit': 500, 'offset': offset})
        if not page:
            break
        trades.extend(page)
        offset += 500
    if not trades:
        return pd.DataFrame(columns=['ts', 'price', 'source'])
    df = pd.DataFrame(trades)
    df = df[(df['timestamp'] >= tip - 3600)
            & (df['timestamp'] <= tip + 4 * 3600)]
    home = df['asset'].astype(str) == row['home_token']
    df['home_price'] = df['price'].where(home, 1 - df['price'])
    df['ts'] = pd.to_datetime(df['timestamp'], unit='s',
                              utc=True).dt.floor('min')
    df = df.sort_values('timestamp').groupby('ts', as_index=False).last()
    out = df[['ts', 'home_price']].rename(columns={'home_price': 'price'})
    out['source'] = 'trades'
    return out


def fetch_pbp(game_id):
    """Saves the play-by-play for one game from the NBA live data CDN."""
    path = os.path.join(RAW, 'pbp', game_id + '.parquet')
    if os.path.exists(path):
        return
    from nba_api.live.nba.endpoints import playbyplay
    actions = playbyplay.PlayByPlay(game_id).get_dict()['game']['actions']
    df = pd.DataFrame(actions)
    keep = [c for c in PBP_COLUMNS if c in df.columns]
    df = df[keep].astype({'scoreHome': str, 'scoreAway': str})
    df.to_parquet(path, index=False)
    time.sleep(0.3)


def make_sample(n_games):
    """Copies the first n_games fully fetched games into data/sample/."""
    matched = pd.read_parquet(os.path.join(RAW, 'matched.parquet'))
    have = matched[[
        os.path.exists(os.path.join(RAW, 'pbp', g + '.parquet'))
        and os.path.exists(os.path.join(RAW, 'prices', g + '.parquet'))
        for g in matched['game_id']]]
    sample = have.sort_values('game_date').iloc[::max(
        1, len(have) // n_games)].head(n_games)
    for sub in ('pbp', 'prices'):
        os.makedirs(os.path.join(SAMPLE, sub), exist_ok=True)
        for game_id in sample['game_id']:
            shutil.copy(os.path.join(RAW, sub, game_id + '.parquet'),
                        os.path.join(SAMPLE, sub, game_id + '.parquet'))
    sample.to_parquet(os.path.join(SAMPLE, 'matched.parquet'), index=False)
    for name in ('nba_games', 'gamma_markets', 'unmatched'):
        shutil.copy(os.path.join(RAW, name + '.parquet'),
                    os.path.join(SAMPLE, name + '.parquet'))
    print(f'  wrote {len(sample)} games to {SAMPLE}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='only fetch prices/pbp for N games')
    parser.add_argument('--sample', type=int, default=None,
                        help='copy N cached games into data/sample')
    args = parser.parse_args()
    for sub in ('prices', 'pbp'):
        os.makedirs(os.path.join(RAW, sub), exist_ok=True)
    if args.sample:
        make_sample(args.sample)
        return
    print('NBA games')
    games = fetch_games()
    print(f'  {len(games)} games')
    print('Polymarket catalog')
    markets = fetch_gamma()
    print(f'  {len(markets)} moneyline markets')
    matched = match(markets, games)
    todo = matched if args.limit is None else matched.head(args.limit)
    for i, row in enumerate(todo.to_dict('records')):
        for kind, func, arg in (('prices', fetch_prices, row),
                                ('pbp', fetch_pbp, row['game_id'])):
            try:
                func(arg)
            except Exception as error:  # keep going; logged for review
                log_failure(kind, row['game_id'], error)
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(todo)} games fetched', flush=True)
    print('done')


if __name__ == '__main__':
    main()
