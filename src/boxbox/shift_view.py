"""Tables for the 2026 Shift page (app/views/shift_study.py). Every number comes from the shift study code
(shift.py, the same calls as scripts/shift_study.py, with SeasonScaled frozen at config shift_study.season_adjust),
so the page matches notes/runs/2026-11-23_shift_study.txt when the same 2026 rounds are in the data. The wet call
section reads the crossover windows of the wet study (wet.crossover_study, pre 2026 seasons only).

Tables (data/models/shift_<name>.parquet, long format, one row per study, split and item):
  scores   measure 1: MAE, bias, gain over constant pace per model, overall ('all') and per compound
  rates    measure 1: wear slope per compound, training fit and test season alone, and the matched change
  rounds   measure 2: MAE per round and cumulative MAE, frozen, season scaled and rolling refit
  blocks   measure 2: MAE per round block, over all rounds, and bias
  other    measures 3, 4 and 6 as (Measure, Metric, Value): pit loss, predictor against grid, safety cars
  wet      crossover windows: gap trend per lap, crossover lap, cost of one lap off
  meta     one row: rounds scored, frozen SeasonScaled setting, build time
"""
import numpy as np
import pandas as pd

from boxbox import outcome, shift
from boxbox.degradation.task import RACE

TABLES = ['scores', 'rates', 'rounds', 'blocks', 'other', 'wet', 'meta']
VERSIONS = ['frozen', 'season scaled', 'rolling refit']


def span(years):
    years = list(years)
    return f'{years[0]} to {years[-1]}' if len(years) > 1 else str(years[0])


def scores_rows(preds, study, split, test):
    """shift.score_table overall and per compound, as long rows."""
    a = shift.score_table(preds).assign(Compound='all')
    b = shift.score_table(preds, 'Compound').reset_index('Compound')
    t = pd.concat([a, b]).reset_index().rename(columns={'index': 'Model'})
    return t.assign(Study=study, Split=split, Test=test)


def rounds_rows(rec, study, split, test):
    """shift.round_mae as long rows: Round, Version, MAE_s, CumMAE_s, and the round's circuit."""
    per, cum = shift.round_mae(rec)
    circuit = pd.concat(rec.values()).groupby('Round')['Circuit'].first()
    t = per.stack().rename('MAE_s').to_frame().join(cum.stack().rename('CumMAE_s'))
    t.index.names = ['Round', 'Version']
    t = t.reset_index()
    return t.assign(Circuit=t['Round'].map(circuit), Study=study, Split=split, Test=test)


def blocks_rows(rec, study, split, test):
    t = shift.block_table(rec)
    t.index.name = 'Version'
    return t.reset_index().melt('Version', var_name='Block', value_name='Value').assign(Study=study, Split=split, Test=test)


def _other(study, split, test, measure, values):
    return [{'Study': study, 'Split': split, 'Test': test, 'Measure': measure, 'Metric': k, 'Value': float(v)}
            for k, v in values.items()]


def build(t, cfg, log=print, wet_study=True):
    """All page tables from the processed tables (with the holdout season appended, shift.load_tables)."""
    ss = cfg['shift_study']
    scope, k = ss['season_adjust']['scope'], ss['season_adjust']['shrink']
    name_m = ss.get('outcome_model', 'ridge, alpha 1000')
    laps = t['laps']
    pre = laps[~laps['Year'].isin(cfg['holdout_seasons'])]
    out = {n: [] for n in ['scores', 'rates', 'rounds', 'blocks', 'other']}
    df = outcome.entries(t['results'], t['qualifying'], t['sprint_results'], t['events'], cfg, laps)
    ev = t['events'][RACE + ['Circuit']].drop_duplicates(RACE)
    stops = t['pit_loss_stops'].merge(ev, on=RACE, how='left')
    rounds = []
    for study in ['study', 'rehearsal']:
        sp = shift.splits(cfg, study)
        for split, (tr, te) in sp.items():
            log(f'{study} {split}: train {span(tr)}, test {te}')
            preds, fitted = shift.degradation(laps, cfg, tr, te)
            out['scores'].append(scores_rows(preds, study, split, te).assign(Train=span(tr)))
            r = shift.wear_rates(laps, cfg, tr, te, fitted['linear'])
            out['rates'].append(r.rename_axis('Compound').reset_index().assign(Study=study, Split=split, Test=te))
            rec = shift.recovery(laps, cfg, tr, te, scope, k, base=fitted['linear'])
            out['rounds'].append(rounds_rows(rec, study, split, te))
            out['blocks'].append(blocks_rows(rec, study, split, te))
            if study == 'study' and split == 'shift':
                rounds = sorted(preds['linear']['Round'].unique().tolist())
            d2 = shift.component_train(cfg, 'outcome', tr, te)
            tab, _ = shift.finishing(df, cfg, d2, te, name_m)
            out['other'] += _other(study, split, te, 'predictor', {
                'MAE': tab.loc[name_m, 'MAE'], 'GridMAE': tab.loc['grid', 'MAE'], 'Spearman': tab.loc[name_m, 'Spearman'],
                'GridSpearman': tab.loc['grid', 'Spearman'], 'Races': tab.loc[name_m, 'Races']})
            if study == 'study':
                from boxbox import reference
                ptr = shift.component_train(cfg, 'pit_loss', tr, te)
                ptab, _ = shift.pit_loss(stops, cfg, ptr, te, reference.load_circuits(cfg))
                circ = [i for i in ptab.index if i != 'global median'][0]
                out['other'] += _other(study, split, te, 'pit loss', {
                    'MAE': ptab.loc[circ, 'MAE_s'], 'Bias': ptab.loc[circ, 'Bias_s'], 'Races': ptab.loc[circ, 'Races'],
                    'GlobalMAE': ptab.loc['global median', 'MAE_s']})
                g = shift.compound_gaps(laps, cfg, tr, te, fitted['simulator wear'])
                out['other'] += _other(study, split, te, 'fresh gaps', {f'Change_{c}': v for c, v in g['Change'].items()})
                sc = shift.safety_cars(laps, cfg, shift.component_train(cfg, 'safety_car', tr, te), te)
                out['other'] += _other(study, split, te, 'safety cars', {
                    x: sc[x] for x in ['Races', 'LogLoss_SC', 'LogLoss_VSC', 'Rate_SC_pred', 'Rate_SC_obs',
                                       'Rate_VSC_pred', 'Rate_VSC_obs']})
    tables = {n: (pd.DataFrame(v) if n == 'other' else pd.concat(v, ignore_index=True)) for n, v in out.items()}
    if wet_study:
        from boxbox import wet
        log('wet crossover windows')
        win, _ = wet.crossover_study(pre, cfg)
        keep = ['Year', 'Round', 'Circuit', 'Start', 'End', 'GapTrend_s', 'Crossover', 'Found', 'Cross_lo', 'Cross_hi', 'Cost1Lap_s']
        tables['wet'] = win[[c for c in keep if c in win]].reset_index(drop=True)
    else:
        tables['wet'] = pd.DataFrame(columns=['Year', 'Round', 'Circuit', 'Start', 'End', 'GapTrend_s', 'Crossover', 'Found'])
    tables['meta'] = pd.DataFrame([{'Rounds': len(rounds), 'LastRound': max(rounds) if rounds else 0,
                                    'Scope': scope, 'Shrink': k, 'OutcomeModel': name_m,
                                    'Built': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}])
    return tables


def save(tables, cfg):
    d = cfg['paths']['models']
    d.mkdir(exist_ok=True)
    for k, v in tables.items():
        v.to_parquet(d / f'shift_{k}.parquet', index=False)
    return d


def load(cfg):
    d = cfg['paths']['models']
    return {k: pd.read_parquet(d / f'shift_{k}.parquet') for k in TABLES}


# ---------------------------------------------------------------- shaping for the page

def score(scores, study, split, model='linear', compound='all'):
    """One row of the scores table as a dict (MAE_s, Bias_s, Gain, N, Test, Train)."""
    s = scores[(scores['Study'] == study) & (scores['Split'] == split) & (scores['Model'] == model)
               & (scores['Compound'] == compound)]
    return s.iloc[0].to_dict()


def other(tab, study, split, measure):
    """{Metric: Value} of one measure in the other table."""
    o = tab[(tab['Study'] == study) & (tab['Split'] == split) & (tab['Measure'] == measure)]
    return dict(zip(o['Metric'], o['Value']))


def headline(t, study='study'):
    """Reference against shift numbers for the stat row and verdict tiles."""
    ref, sh = (score(t['scores'], study, s) for s in ['reference', 'shift'])
    out = {'ref': ref, 'shift': sh, 'mae_change': sh['MAE_s'] / ref['MAE_s'] - 1}
    for m in ['predictor', 'pit loss', 'safety cars']:
        out[m] = {s: other(t['other'], study, s, m) for s in ['reference', 'shift']}
    for s in ['reference', 'shift']:
        p = out['predictor'][s]
        if p:
            p['Gain'] = 1 - p['MAE'] / p['GridMAE']
    return out


def score_rows(scores, study, models=('constant pace', 'linear', 'simulator wear')):
    """Rows for score_table: per model, MAE, bias and gain on the reference and shift test seasons."""
    rows = []
    for m in models:
        row = {'Model': m}
        for split in ['reference', 'shift']:
            r = score(scores, study, split, m)
            y = int(r['Test'])
            row.update({f'MAE {y}': r['MAE_s'], f'Bias {y}': r['Bias_s'], f'Gain {y}': r['Gain']})
        rows.append(row)
    return rows


def round_series(rounds, study, split='shift'):
    """{version: frame of Round, Circuit, MAE_s, CumMAE_s} in VERSIONS order."""
    r = rounds[(rounds['Study'] == study) & (rounds['Split'] == split)].sort_values('Round')
    return {v: r[r['Version'] == v] for v in VERSIONS if (r['Version'] == v).any()}


def block_rows(blocks, study):
    """Rows for score_table: MAE per round block for each version, reference and shift."""
    b = blocks[blocks['Study'] == study]
    rows = []
    for split in ['reference', 'shift']:
        s = b[b['Split'] == split]
        if s.empty:
            continue
        y = int(s['Test'].iloc[0])
        for v in VERSIONS:
            x = s[s['Version'] == v].set_index('Block')['Value']
            rows.append({'Model': f'{y}, {v}', **x.to_dict()})
    return rows


def rates_wide(rates, study):
    """Wear slope per compound: training fit (shift split), reference season alone, shift season alone, and the
    matched cell change of each test season. Index Compound."""
    r = rates[rates['Study'] == study]
    ref, sh = r[r['Split'] == 'reference'].set_index('Compound'), r[r['Split'] == 'shift'].set_index('Compound')
    yr, ys = int(ref['Test'].iloc[0]), int(sh['Test'].iloc[0])
    return pd.DataFrame({'Training fit': sh['Train'], f'{yr} alone': ref['TestAlone'], f'{ys} alone': sh['TestAlone'],
                         f'Matched change {yr}': ref['MatchedChange'], f'Matched change {ys}': sh['MatchedChange']})


def wet_costs(wet, laps_off):
    """Cost of switching laps_off laps from the crossover in every window where a crossover was found (|b| k^2 / 2,
    wet.lateness_cost), and the cost at the median trend. Returns (frame Race, GapTrend_s, Cost_s by cost, median)."""
    from boxbox.wet import lateness_cost
    w = wet[wet['Found'].astype(bool) & np.isfinite(wet['GapTrend_s'].astype(float))].copy()
    w['Race'] = [f'{c} {int(y)}, laps {int(a)} to {int(b)}' for c, y, a, b in zip(w['Circuit'], w['Year'], w['Start'], w['End'])]
    w['Cost_s'] = lateness_cost(w['GapTrend_s'].to_numpy(), laps_off)
    med = float(lateness_cost(np.median(np.abs(w['GapTrend_s'])), laps_off)) if len(w) else float('nan')
    return w.sort_values('Cost_s', ascending=False)[['Race', 'GapTrend_s', 'Cost_s']], med


def verdicts(t, study='study'):
    """Verdict tiles (label, value, sub, tone) from the reference against shift numbers. Rules: degradation has
    shifted when error rises over 5% or bias changes sign; the predictor holds while it keeps at least 80% of its
    reference gain over grid; fresh tyre gaps are normal while the largest change is within 1.5 times the
    reference's; pit loss breaks when the circuit medians stop beating one global median; a neutralisation rate
    breaks when the seen rate is 0.2 or more from the predicted one."""
    h = headline(t, study)
    ref, sh = h['ref'], h['shift']
    y1 = int(sh['Test'])
    out = []
    shifted = h['mae_change'] > 0.05 or np.sign(ref['Bias_s']) != np.sign(sh['Bias_s'])
    out.append({'label': 'Degradation forecast', 'value': 'Shifted' if shifted else 'Holds', 'tone': 'bad' if shifted else 'good',
                'sub': f'error {h["mae_change"]:+.0%}, bias {ref["Bias_s"]:+.2f} to {sh["Bias_s"]:+.2f} s'})
    p = h['predictor']
    if p['reference'] and p['shift']:
        holds = p['shift']['Gain'] >= 0.8 * p['reference']['Gain']
        out.append({'label': 'Finishing predictor', 'value': 'Holds' if holds else 'Weakens', 'tone': 'good' if holds else 'bad',
                    'sub': f'{p["shift"]["Gain"]:.1%} better than grid on {y1}, {p["reference"]["Gain"]:.1%} before'})
    g = {s: other(t['other'], study, s, 'fresh gaps') for s in ['reference', 'shift']}
    if g['reference'] and g['shift']:
        big = max(map(abs, g['shift'].values())) > 1.5 * max(map(abs, g['reference'].values()))
        out.append({'label': 'Fresh tyre gaps', 'value': 'Moved' if big else 'Normal', 'tone': 'bad' if big else 'good',
                    'sub': 'a usual season to season change' if not big else 'beyond a usual season to season change'})
    pl = h['pit loss']['shift']
    if pl:
        broke = pl['MAE'] >= pl['GlobalMAE']
        out.append({'label': 'Pit loss', 'value': 'Breaks' if broke else 'Holds', 'tone': 'bad' if broke else 'good',
                    'sub': f'{pl["MAE"]:.2f} s off per race, one global number {pl["GlobalMAE"]:.2f}'})
    sc = h['safety cars']['shift']
    if sc:
        for kind in ['SC', 'VSC']:
            seen, pred = sc[f'Rate_{kind}_obs'], sc[f'Rate_{kind}_pred']
            broke = abs(seen - pred) >= 0.2
            out.append({'label': 'Safety cars' if kind == 'SC' else 'Virtual safety cars', 'value': 'Breaks' if broke else 'Holds',
                        'tone': 'bad' if broke else 'good', 'sub': f'in {seen:.0%} of races, {pred:.0%} predicted'})
    return out
