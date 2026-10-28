"""Leave-subjects-out folds. Subjects never appear in more than one of train/val/test within a fold."""
import json
import numpy as np


def make_folds(subjects, n_folds, n_val, seed=42):
    subjects = np.array(sorted(int(s) for s in subjects))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects)
    chunks = np.array_split(perm, n_folds)
    folds = []
    for f in range(n_folds):
        test = sorted(int(s) for s in chunks[f])
        rest = [int(s) for s in perm if int(s) not in set(test)]
        rrng = np.random.RandomState(seed + 1000 + f)
        val = sorted(int(s) for s in rrng.choice(rest, size=min(n_val, len(rest) - 1), replace=False))
        train = sorted(s for s in rest if s not in set(val))
        folds.append(dict(fold=f, train=train, val=val, test=test))
    for fo in folds:
        assert not (set(fo["train"]) & set(fo["val"])) and not (set(fo["train"]) & set(fo["test"])) \
            and not (set(fo["val"]) & set(fo["test"]))
    return folds


def save_folds(folds, path):
    with open(path, "w") as f:
        json.dump(folds, f, indent=1)


def load_folds(path):
    with open(path) as f:
        return json.load(f)
