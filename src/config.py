import copy
from pathlib import Path
import yaml


class Cfg(dict):
    """dict with attribute access (recursive)."""
    __getattr__ = dict.get

    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) else v


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, debug=False, work_dir=None, overrides=None) -> Cfg:
    path = path or Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    dbg = raw.pop("debug", {})
    if debug:
        d = copy.deepcopy(dbg)
        if "max_subjects" in d:
            raw["data"]["max_subjects"] = d.pop("max_subjects")
        if "n_folds" in d:
            raw["split"]["n_folds"] = d.pop("n_folds")
        if "n_val_subjects" in d:
            raw["split"]["n_val_subjects"] = d.pop("n_val_subjects")
        _merge(raw, d)
    if work_dir:
        raw["work_dir"] = str(work_dir)
    for o in overrides or []:           # "section.key=value"
        k, v = o.split("=", 1)
        node = raw
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(v)
    raw["debug_mode"] = bool(debug)
    return Cfg(raw)


def work(cfg) -> Path:
    p = Path(cfg.work_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p
