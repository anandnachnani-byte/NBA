"""
Anand Nachnani
CSE 163 Final Project: Overreaction on the Court?

Tests for data_processing.py (and eda.py on the committed sample).
Each test uses a tiny hand-built fixture in tests/data/ whose correct
answer was worked out by hand, so a passing assert means the code
matches the hand calculation. Run with: python test_eda.py
"""
import json
import math
import os

import numpy as np
import pandas as pd

import data_processing as dp

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'tests', 'data')


def load_fixture(name):
    """Reads a fixture file from tests/data/."""
    path = os.path.join(FIXTURES, name)
    if name.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    return pd.read_csv(path, dtype={'GAME_ID': str})


def test_parse_json_field():
    """The doubly-encoded clobTokenIds string becomes a list."""
    assert dp.parse_json_field('["111", "222"]') == ['111', '222']
    assert dp.parse_json_field(['a', 'b']) == ['a', 'b']
    assert dp.parse_json_field(None) == []
    assert dp.parse_json_field(float('nan')) == []


def test_parse_event_slug():
    """Only single-game slugs parse; props and futures are rejected."""
    assert dp.parse_event_slug('nba-gsw-lal-2025-10-21') == \
        ('GSW', 'LAL', '2025-10-21')
    assert dp.parse_event_slug('nba-uta-lac-2026-11-17-exact-margin') \
        is None
    assert dp.parse_event_slug('nba-champion-2026') is None
    assert dp.parse_event_slug(None) is None


def test_extract_moneyline():
    """The moneyline market is picked out of an event with others."""
    row = dp.extract_moneyline(load_fixture('fake_event.json'))
    assert row['question'] == 'Warriors vs. Lakers'
    assert row['token_0'] == '111' and row['token_1'] == '222'
    assert row['outcome_1'] == 'Lakers'
    assert row['market_volume'] == 900.25
    assert row['event_volume'] == 1000.5
    no_ml = load_fixture('fake_event.json')
    no_ml['markets'] = no_ml['markets'][:1]
    assert dp.extract_moneyline(no_ml) is None


def test_games_from_finder():
    """Two team rows collapse into one game with home/away right."""
    games = dp.games_from_finder(load_fixture('fake_finder.csv'))
    assert len(games) == 3
    lal = games[games['game_id'] == '0022500002'].iloc[0]
    assert lal['home'] == 'LAL' and lal['away'] == 'GSW'
    assert lal['home_win'] == 0
    chi = games[games['game_id'] == '0022500100'].iloc[0]
    assert chi['home'] == 'CHI' and chi['home_win'] == 0


def make_market(slug, outcome_0, outcome_1, volume=1.0):
    """Builds one market row like extract_moneyline returns."""
    return {'slug': slug, 'slug_date': slug[-10:], 'outcome_0': outcome_0,
            'outcome_1': outcome_1, 'token_0': 't0-' + slug,
            'token_1': 't1-' + slug, 'market_volume': volume}


def test_match_markets_to_games():
    """
    Exact dates beat one-day matches (back-to-back games), unknown
    names and missing games are logged, and duplicates keep the
    busiest market.
    """
    games = dp.games_from_finder(load_fixture('fake_finder.csv'))
    markets = pd.DataFrame([
        make_market('nba-gsw-lal-2025-10-21', 'Warriors', 'Lakers', 5),
        make_market('nba-nyk-chi-2025-11-01', 'Knicks', 'Bulls'),
        make_market('nba-chi-nyk-2025-11-03', 'Bulls', 'Knicks'),
        make_market('nba-bkn-pho-2025-10-12', 'Nets', 'PHO'),
        make_market('nba-bos-mia-2025-10-12', 'Celtics', 'Heat'),
        make_market('nba-gsw-lal-2025-10-22', 'Warriors', 'Lakers', 1),
    ])
    matched, unmatched = dp.match_markets_to_games(markets, games)
    ids = dict(zip(matched['slug'], matched['game_id']))
    assert ids == {'nba-gsw-lal-2025-10-21': '0022500002',
                   'nba-nyk-chi-2025-11-01': '0022500100',
                   'nba-chi-nyk-2025-11-03': '0022500101'}
    lal = matched[matched['game_id'] == '0022500002'].iloc[0]
    assert lal['home_token'] == 't1-nba-gsw-lal-2025-10-21'
    reasons = dict(zip(unmatched['slug'], unmatched['reason']))
    assert reasons == {
        'nba-bkn-pho-2025-10-12': 'unknown team name',
        'nba-bos-mia-2025-10-12': 'no game found',
        'nba-gsw-lal-2025-10-22': 'duplicate market for game',
    }


def test_clock_and_time():
    """Clock parsing and regulation/overtime time arithmetic."""
    assert dp.parse_clock('PT11M32.00S') == 692.0
    assert dp.parse_clock('PT00M00.50S') == 0.5
    assert dp.seconds_remaining(1, 720) == 2880
    assert dp.seconds_remaining(4, 30) == 30
    assert dp.seconds_remaining(5, 120) == 120
    assert dp.seconds_elapsed(1, 720) == 0
    assert dp.seconds_elapsed(4, 0) == 2880
    assert dp.seconds_elapsed(6, 300) == 3180
    try:
        dp.parse_clock('11:32')
        assert False, 'bad clock should raise'
    except ValueError:
        pass


def test_clean_pbp():
    """
    Bad timestamps are dropped, the late correction row (actionNumber
    18, logged at 10:45 of Q1) is moved into game-time order, and
    margins are home minus away.
    """
    pbp = dp.clean_pbp(load_fixture('fake_pbp.csv'))
    assert len(pbp) == 17
    assert pbp['actionNumber'].tolist()[3] == 18
    assert pbp['margin'].iloc[-1] == 16
    assert pbp['sec_remaining'].iloc[1] == 2850
    assert (pbp['scoreHome'].diff().dropna() >= 0).all()
    assert str(pbp['ts'].dt.tz) == 'UTC'


def test_stale_score_rows():
    """A stale row showing an old score cannot create a fake run."""
    raw = load_fixture('fake_pbp.csv')
    stale = raw.iloc[[2]].copy()     # 5-0 at 11:00 ...
    stale['actionNumber'] = 99       # ... logged again at 07:00
    stale['clock'] = 'PT07M00.00S'
    raw = pd.concat([raw, stale], ignore_index=True)
    pbp = dp.clean_pbp(raw)
    assert (pbp['scoreHome'].diff().dropna() >= 0).all()
    assert len(dp.detect_runs(pbp)) == 2


def test_detect_runs():
    """
    The fixture has a 9-0 run in 120 game seconds (a run), a 7-0 burst
    (too small), an 8-0 streak spread over 270 seconds (too slow), and
    an 8-0 run in 60 seconds of the fourth quarter (a run).
    """
    runs = dp.detect_runs(dp.clean_pbp(load_fixture('fake_pbp.csv')))
    assert len(runs) == 2
    first, second = runs.iloc[0], runs.iloc[1]
    assert first['team'] == 'home' and first['run_pts'] == 9
    assert first['period'] == 1
    assert first['margin_before'] == 0 and first['margin_after'] == 9
    assert first['end_elapsed'] - first['start_elapsed'] == 90
    assert second['run_pts'] == 8 and second['period'] == 4
    assert second['margin_before'] == 8 and second['margin_after'] == 16
    loose = dp.detect_runs(dp.clean_pbp(load_fixture('fake_pbp.csv')),
                           min_pts=7)
    assert len(loose) == 3 and loose['team'].tolist()[1] == 'away'


def fake_prices():
    """Home prices with a gap between 00:02:10 and 00:10:00."""
    return dp.price_frame([
        {'t': 1767225630, 'p': 0.5},   # 00:00:30
        {'t': 1767225730, 'p': 0.7},   # 00:02:10
        {'t': 1767226200, 'p': 0.6},   # 00:10:00
    ])


def test_build_game_minutes():
    """
    Prices and scores line up minute by minute. A print a few seconds
    after a minute (00:02:10) belongs to that minute, not the next one,
    and prices older than three minutes drop.
    """
    pbp = dp.clean_pbp(load_fixture('fake_pbp.csv'))
    minutes = dp.build_game_minutes(pbp, fake_prices())
    by_min = minutes.set_index(minutes['minute_ts'].dt.strftime('%H:%M'))
    assert by_min.loc['00:00', 'price_home'] == 0.5
    assert by_min.loc['00:01', 'price_home'] == 0.5
    assert by_min.loc['00:02', 'price_home'] == 0.7
    assert by_min.loc['00:03', 'price_home'] == 0.7
    assert by_min.loc['00:03', 'margin'] == 9
    assert by_min.loc['00:05', 'price_home'] == 0.7
    assert np.isnan(by_min.loc['00:06', 'price_home'])
    assert by_min.loc['00:10', 'price_home'] == 0.6
    assert math.isclose(by_min.loc['00:03', 'logit_price'],
                        math.log(0.7 / 0.3))


def test_pregame_and_value_at():
    """The pregame price is the last price at or before tip-off."""
    prices = fake_prices()
    tip = pd.Timestamp('2026-01-01T00:01:00Z')
    assert dp.pregame_price(prices, tip) == 0.5
    assert np.isnan(dp.pregame_price(prices,
                                     pd.Timestamp('2025-12-31', tz='UTC')))
    assert dp.value_at(prices, tip, 'price', ts_col='ts') == 0.5


def test_logit_and_summary():
    """Log-odds clip extremes; the summary matches hand values."""
    assert dp.logit(0.5) == 0
    assert math.isclose(dp.logit(1.0), math.log(0.99 / 0.01))
    summary = dp.seven_number_summary([1, 2, 3, 4, np.nan])
    assert summary['mean'] == 2.5 and summary['50%'] == 2.5
    assert summary['min'] == 1 and summary['max'] == 4
    assert summary['25%'] == 1.75 and summary['75%'] == 3.25
    assert math.isclose(summary['std'], 1.2909944, rel_tol=1e-6)


def test_eda_on_sample():
    """eda.py builds consistent tables from the committed sample."""
    if not os.path.exists(os.path.join('data', 'sample',
                                       'matched.parquet')):
        print('  (skipped: no data/sample yet)')
        return
    import eda
    data = eda.load_data(os.path.join('data', 'sample'))
    tables = eda.build_tables(data)
    minutes = tables['minutes']
    assert minutes['game_id'].nunique() == len(data['matched'])
    assert minutes['price_home'].dropna().between(0, 1).all()
    assert (minutes['sec_remaining'] >= 0).all()
    runs = tables['runs']
    assert (runs['run_pts'] >= 8).all()
    assert set(runs['team']) <= {'home', 'away'}


def main():
    tests = [test_parse_json_field, test_parse_event_slug,
             test_extract_moneyline, test_games_from_finder,
             test_match_markets_to_games, test_clock_and_time,
             test_clean_pbp, test_stale_score_rows, test_detect_runs,
             test_build_game_minutes,
             test_pregame_and_value_at, test_logit_and_summary,
             test_eda_on_sample]
    for test in tests:
        test()
        print('passed', test.__name__)
    print('All tests passed!')


if __name__ == '__main__':
    main()
