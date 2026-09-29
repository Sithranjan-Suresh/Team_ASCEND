#!/usr/bin/env python3
"""
Stage 1: build a stratified train/held-out split of the P3 puzzle set.

V1 (main.py / main_v2.py) trained and evaluated on the exact same slice of
puzzles.json, so every reported accuracy number measured training-set fit,
not generalization. This script fixes that by producing a reproducible
split, stratified by puzzle `module` (P3's built-in category field, e.g.
human_eval.py, codeforces.py, number_theory.py -- 18 categories total), so
both the train and held-out sets contain a representative mix of puzzle
types rather than an arbitrary index-ordered slice.

Usage:
    python split_dataset.py --seed 42 --heldout-frac 0.2

Writes PythonProgrammingPuzzles/puzzles/split_seed<seed>.json:
    {"seed": 42, "heldout_frac": 0.2, "train": [indices...], "heldout": [indices...]}
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

PUZZLES_PATH = "PythonProgrammingPuzzles/puzzles/puzzles.json"


def load_puzzles(path: str = PUZZLES_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_split(puzzles: list[dict], heldout_frac: float, seed: int) -> tuple[list[int], list[int]]:
    by_module: dict[str, list[int]] = defaultdict(list)
    for i, puzzle in enumerate(puzzles):
        by_module[puzzle.get("module", "unknown")].append(i)

    rng = random.Random(seed)
    train_idx: list[int] = []
    heldout_idx: list[int] = []

    for module, indices in by_module.items():
        indices = indices[:]
        rng.shuffle(indices)
        n_heldout = max(1, round(len(indices) * heldout_frac)) if len(indices) > 1 else 0
        heldout_idx.extend(indices[:n_heldout])
        train_idx.extend(indices[n_heldout:])

    rng.shuffle(train_idx)
    rng.shuffle(heldout_idx)
    return train_idx, heldout_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles-path", default=PUZZLES_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-frac", type=float, default=0.2)
    args = parser.parse_args()

    puzzles = load_puzzles(args.puzzles_path)
    train_idx, heldout_idx = stratified_split(puzzles, args.heldout_frac, args.seed)

    assert set(train_idx).isdisjoint(heldout_idx), "train/held-out overlap -- split is broken"
    assert len(train_idx) + len(heldout_idx) == len(puzzles)

    out = {
        "seed": args.seed,
        "heldout_frac": args.heldout_frac,
        "n_total": len(puzzles),
        "n_train": len(train_idx),
        "n_heldout": len(heldout_idx),
        "train": sorted(train_idx),
        "heldout": sorted(heldout_idx),
    }

    out_path = Path(args.puzzles_path).parent / f"split_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Total puzzles: {out['n_total']}")
    print(f"Train: {out['n_train']}  Held-out: {out['n_heldout']}")
    print(f"Wrote split to: {out_path}")


if __name__ == "__main__":
    main()
