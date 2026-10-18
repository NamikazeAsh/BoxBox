"""Equal weight averages of fitted degradation models (no weights tuned, so no extra selection on validation).
Members are any models with the shared interface; config `degradation.ensembles` names them from the
comparison ladder.

WearAverage: mean of the members' wear and dirty air, so it is itself a wear model (a wear curve per circuit,
compound and stint length) that the strategy simulator can use. Members need wear and dirty_air.
ForecastAverage: mean of the members' anchored forecasts, so it can include models that only forecast from an
anchor (LightGBM). For live forecasts, not the pre race simulator.
"""
import pandas as pd

from boxbox.degradation.task import forecast


class WearAverage:
    def __init__(self, members, name='wear average'):
        self.members, self.name = members, name

    def fit(self, laps):
        for m in self.members:
            m.fit(laps)
        return self

    def wear(self, laps):
        return sum(m.wear(laps) for m in self.members) / len(self.members)

    def dirty_air(self, laps):
        return sum(m.dirty_air(laps) for m in self.members) / len(self.members)


class ForecastAverage:
    def __init__(self, members, name='forecast average'):
        self.members, self.name = members, name

    def fit(self, laps):
        for m in self.members:
            m.fit(laps)
        return self

    def forecast(self, laps, cfg):
        preds = [forecast(m, laps, cfg) for m in self.members]
        p = preds[0].drop(columns=[c for c in preds[0].columns if c.startswith('Q') or c == 'PredSD_s'])
        p['Pred_s'] = pd.concat([q['Pred_s'].reindex(p.index) for q in preds], axis=1).mean(axis=1)
        p['Err_s'] = p['Pred_s'] - p['LapTimeFC_s']
        return p


KINDS = {'wear': WearAverage, 'forecast': ForecastAverage}
