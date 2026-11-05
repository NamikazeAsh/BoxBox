"""Strategy simulator v1 (modelling D6): dry races, green flag only, no retirements, tyre change stops only.
Every car runs a fixed strategy (stop laps and compounds); rival strategies come from the pit model
(pitdecision.PlanPolicy), the real race or the user. Cars race lap by lap on the lap time model (laptime), with
overtaking by a per circuit threshold (boxbox.overtaking).

* laptime: the free air lap time of a car (wear, compound gap, fuel, track evolution, race start, dirty air, noise).
* pace: each car's base pace from its pre race features.
* strategy: fixed strategies (legal ones for the user's car, the real ones, the pit model's).
* race: the lap by lap engine and the strategy search.
* inputs: every fitted part in one object (SimInputs), saved to data/models/simulator_inputs.joblib.
* backtest: replays of historical green flag races.
"""
