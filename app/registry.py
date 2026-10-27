"""Every page of the app, in menu order. Adding a page: write app/views/<key>.py, add an entry here and a
switch under dashboard.pages in config.yaml. Status 'soon' shows a placeholder card while the view is being built."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entry:
    key: str
    title: str
    section: str
    icon: str
    status: str = 'ready'           # ready: app/views/<key>.py; soon: placeholder card
    blurb: str = ''
    plans: tuple = field(default=())


PAGES = [
    Entry('paddock', 'Paddock', 'Race weekend', ':material/flag:'),
    Entry('tyre_lab', 'Tyre Lab', 'Race weekend', ':material/tire_repair:',
          blurb='How fast each compound fades, circuit by circuit.'),
    Entry('race_predictor', 'Race Predictor', 'Race weekend', ':material/leaderboard:',
          blurb='Pick a race and compare predicted, actual and grid order.'),
    Entry('strategy_room', 'Strategy Room', 'Race weekend', ':material/route:', 'soon',
          'Build a strategy and race it a thousand times.',
          ('Pick stops and compounds for any driver and circuit',
           'Monte Carlo spread of race time, with safety cars sampled per circuit',
           'Every legal strategy ranked, with the undercut and overcut windows')),
    Entry('pit_wall', 'Pit Wall', 'Race weekend', ':material/headset_mic:', 'soon',
          'Replay a real race lap by lap and make the call to box.',
          ('Lap by lap replay with gaps, tyres and track status',
           'A "box this lap?" call each lap, scored against the simulator',
           'What the team actually did, side by side')),
    Entry('overtaking', 'Overtaking', 'Research', ':material/swap_horiz:', 'soon',
          'How hard it is to pass, circuit by circuit.',
          ('Pass probability from pace gap, tyre age gap and DRS',
           'Where a driver rejoins after a stop and whether they can get by')),
    Entry('shift_study', '2026 Shift', 'Research', ':material/compare_arrows:', 'soon',
          'Do models trained on 2022 to 2025 still hold for the 2026 cars?',
          ('Degradation and predictor error on 2026 against earlier seasons',
           'How fast each model recovers as 2026 races come in')),
    Entry('methods', 'Methods', 'Research', ':material/science:',
          blurb='Splits, metrics and baselines behind every number in the app.'),
    Entry('style_guide', 'Style guide', 'Research', ':material/palette:'),
]
