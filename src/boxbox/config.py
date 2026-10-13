from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / 'config.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['paths'] = {k: ROOT / v for k, v in cfg['paths'].items()}
    return cfg


def seasons_for(cfg, component):
    """Training seasons for a component. Refuses holdout seasons outside the shift study."""
    seasons = cfg['training_seasons'][component]
    leaked = set(seasons) & set(cfg['holdout_seasons'])
    if leaked and not component.startswith('shift_study'):
        raise ValueError(f'{component} includes holdout seasons {sorted(leaked)}')
    return seasons
