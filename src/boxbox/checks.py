"""Sanity checks on the processed tables. Errors stop the build, warnings are printed."""
import logging

log = logging.getLogger('boxbox.checks')


class DataError(Exception):
    pass


def run_checks(t, ev):
    errors, warns = [], []
    laps, res = t['laps'], t['results']

    dup = laps.duplicated(['Year', 'Round', 'Driver', 'LapNumber']).sum()
    if dup:
        errors.append(f'{dup} duplicate race laps (Year, Round, Driver, LapNumber)')

    past = ev[ev['EventDate'] < laps['EventDate'].max()] if 'EventDate' in laps else ev
    have = set(zip(laps['Year'], laps['Round']))
    missing = [f'{y} R{r}' for y, r in zip(past['Year'], past['Round']) if (y, r) not in have]
    if missing:
        warns.append(f'{len(missing)} past races have no laps: {", ".join(missing)}')

    per = laps.groupby(['Year', 'Round'])
    miss = per['LapTime_s'].apply(lambda s: s.isna().mean())
    for (y, r), m in miss[miss > 0.10].items():
        warns.append(f'{y} R{r}: {m:.0%} of lap times missing')
    small = per.size()
    for (y, r), n in small[small < 300].items():
        warns.append(f'{y} R{r}: only {n} laps (shortened or red flagged race?)')

    clean = per['IsClean'].mean()
    for (y, r), c in clean[clean < 0.2].items():
        warns.append(f'{y} R{r}: only {c:.0%} clean laps')

    # Compounds are modelled as event labels (modelling D1); anything else means a season the labels do not cover
    odd = laps.loc[~laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']) & laps['Compound'].notna(),
                   'Compound'].value_counts()
    if len(odd):
        warns.append(f'unexpected compound labels: {odd.to_dict()}')

    if res.duplicated(['Year', 'Round', 'Session', 'DriverNumber']).sum():
        errors.append('duplicate result rows')

    for w in warns:
        log.warning(w)
    if errors:
        raise DataError('; '.join(errors))
    return warns
