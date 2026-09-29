#!/usr/bin/env python3
"""
Stage 1: rerun the existing GRPO pipeline (ladder_optimizer_v4 +
variant_generator_v2, same as main_v2.py) unmodified, except pre/post
accuracy is now measured on a held-out split instead of the training
split.

This is deliberately a thin wrapper around main_v2.py's logic, not a
rewrite -- the point of Stage 1 is to answer "does V1's reported gain
survive held-out evaluation?", not to introduce new methodology yet.

Requires split_seed<seed>.json to already exist (run split_dataset.py first).

Non-interactive by default (set RESUME_INTERACTIVE=1 to get the old
y/N prompt back) so this runs unattended in a Colab cell.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset

import ladder_optimizer_v4 as lo
import variant_generator_v2 as vg

PUZZLES_PATH = "PythonProgrammingPuzzles/puzzles/puzzles.json"
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))
SPLIT_PATH = f"PythonProgrammingPuzzles/puzzles/split_seed{SPLIT_SEED}.json"
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
RUNS_ROOT = Path("training_runs_holdout")

# How many train-split problems to actually train on this run, and how many
# held-out problems to evaluate on. Keep these equal to the sizes used by
# main_v2.py (100) unless you have Colab time to spare.
TRAIN_LIMIT = int(os.environ.get("TRAIN_LIMIT", "100"))
HELDOUT_EVAL_LIMIT = int(os.environ.get("HELDOUT_EVAL_LIMIT", "100"))

TRAINING_HYPERPARAMS = {
    "max_steps": 20,
    "target_accuracy": 0.8,
    "learning_rate": 5e-6,
    "group_size": 4,
    "exec_timeout_seconds": 1.0,
}

RESUME_INTERACTIVE = os.environ.get("RESUME_INTERACTIVE", "0") == "1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "unnamed"


def load_puzzles(path: str = PUZZLES_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"puzzles.json not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        puzzles = json.load(f)
    if not puzzles:
        raise ValueError("No puzzles found")
    return puzzles


def load_split(path: str = SPLIT_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split file not found at: {path}. Run split_dataset.py --seed {SPLIT_SEED} first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_problem_id(index: int, puzzle: dict) -> str:
    name = puzzle.get("name", f"problem_{index}")
    return f"{index:05d}_{sanitize_name(name)}"


def to_dataset_entries(puzzles: list[dict], indices: list[int]) -> list[dict]:
    return [
        {
            "prompt": puzzles[i]["sat"],
            "problem_id": build_problem_id(i, puzzles[i]),
            "variant_num": 0,
            "source_name": puzzles[i].get("name", f"problem_{i}"),
            "notes": puzzles[i].get("notes", ""),
        }
        for i in indices
    ]


def list_numeric_run_ids(runs_root: Path) -> list[int]:
    if not runs_root.exists():
        return []
    return sorted(int(c.name) for c in runs_root.iterdir() if c.is_dir() and c.name.isdigit())


def next_run_id(runs_root: Path) -> str:
    existing = list_numeric_run_ids(runs_root)
    return f"{(existing[-1] + 1) if existing else 1:05d}"


def load_run_json(run_dir: Path) -> dict:
    with open(run_dir / "run.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_run_json(run_dir: Path, data: dict):
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def find_latest_incomplete_run(runs_root: Path) -> Path | None:
    for run_id in reversed(list_numeric_run_ids(runs_root)):
        run_dir = runs_root / f"{run_id:05d}"
        if not (run_dir / "run.json").exists():
            continue
        try:
            data = load_run_json(run_dir)
        except Exception:
            continue
        if data.get("status") != "completed":
            return run_dir
    return None


def should_resume(run_dir: Path) -> bool:
    run_data = load_run_json(run_dir)
    completed = len(run_data.get("completed_problem_indices", []))
    total = run_data.get("problem_limit_effective", "?")
    print(f"Found incomplete run {run_dir.name} with {completed}/{total} completed problems.")
    if not RESUME_INTERACTIVE:
        print("RESUME_INTERACTIVE=0 -> auto-resuming (non-interactive mode).")
        return True
    answer = input("Resume from the most recent checkpoint? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def initialize_new_run(runs_root: Path, model_id: str, problem_limit_effective: int,
                        heldout_eval_limit: int, split_seed: int, hyperparams: dict) -> tuple[str, Path, dict]:
    run_id = next_run_id(runs_root)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    run_data = {
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "model_id": model_id,
        "split_seed": split_seed,
        "problem_limit_effective": problem_limit_effective,
        "heldout_eval_limit": heldout_eval_limit,
        "completed_problem_indices": [],
        "completed_problem_ids": [],
        "latest_checkpoint": None,
        "problem_checkpoints": {},
        "pretrain_evaluated": False,
        "posttrain_evaluated": False,
        "hyperparameters": hyperparams,
        "results": {},
    }
    save_run_json(run_dir, run_data)
    return run_id, run_dir, run_data


def resolve_run(runs_root: Path, model_id: str, problem_limit_effective: int,
                 heldout_eval_limit: int, split_seed: int, hyperparams: dict) -> tuple[str, Path, dict, bool]:
    runs_root.mkdir(parents=True, exist_ok=True)
    latest_incomplete = find_latest_incomplete_run(runs_root)

    if latest_incomplete is not None and should_resume(latest_incomplete):
        run_data = load_run_json(latest_incomplete)
        run_data["updated_at"] = utc_now_iso()
        save_run_json(latest_incomplete, run_data)
        return run_data["run_id"], latest_incomplete, run_data, True

    run_id, run_dir, run_data = initialize_new_run(
        runs_root, model_id, problem_limit_effective, heldout_eval_limit, split_seed, hyperparams
    )
    return run_id, run_dir, run_data, False


def main():
    puzzles = load_puzzles()
    split = load_split()
    print(f"Total puzzles: {len(puzzles)}  (train split: {split['n_train']}, held-out split: {split['n_heldout']})")

    train_indices = split["train"][:TRAIN_LIMIT]
    heldout_indices = split["heldout"][:HELDOUT_EVAL_LIMIT]
    assert set(train_indices).isdisjoint(heldout_indices), "train/held-out overlap -- refusing to run"

    run_id, run_dir, run_data, resumed = resolve_run(
        runs_root=RUNS_ROOT,
        model_id=MODEL_ID,
        problem_limit_effective=len(train_indices),
        heldout_eval_limit=len(heldout_indices),
        split_seed=SPLIT_SEED,
        hyperparams=TRAINING_HYPERPARAMS,
    )
    print(f"Run ID: {run_id} ({'resumed' if resumed else 'new'})")

    model, tokenizer, has_native_chat_template = lo.load_default_model(MODEL_ID)
    print(f"Model loaded: {MODEL_ID} (native chat template: {has_native_chat_template})")

    optimizer = lo.LadderOptimizer(
        model=model,
        tokenizer=tokenizer,
        run_dir=run_dir,
        debug=False,
        has_native_chat_template=has_native_chat_template,
        exec_timeout_seconds=TRAINING_HYPERPARAMS["exec_timeout_seconds"],
    )

    latest_checkpoint = run_data.get("latest_checkpoint")
    if latest_checkpoint and Path(latest_checkpoint).exists():
        print(f"Loading checkpoint from: {latest_checkpoint}")
        optimizer.load_adapter_checkpoint(Path(latest_checkpoint))

    train_dataset = to_dataset_entries(puzzles, train_indices)
    heldout_dataset = to_dataset_entries(puzzles, heldout_indices)
    print(f"Training on {len(train_dataset)} problems (train split); "
          f"evaluating on {len(heldout_dataset)} problems (held-out split, never trained on)")

    if not run_data.get("pretrain_evaluated", False):
        print("\n=== Evaluate PRE-train accuracy on HELD-OUT split ===")
        pre_eval_ds = Dataset.from_list(heldout_dataset)
        pre_results = optimizer.evaluate_performance(pre_eval_ds)

        run_data = load_run_json(run_dir)
        run_data["pretrain_evaluated"] = True
        run_data.setdefault("results", {})["pretrain_holdout"] = pre_results
        run_data["updated_at"] = utc_now_iso()
        save_run_json(run_dir, run_data)
        print(f"Pre-train held-out accuracy: {pre_results}")
    else:
        print("\n=== Skipping pre-train eval (already recorded for this run) ===")

    completed_indices = set(run_data.get("completed_problem_indices", []))
    print("\n=== Beginning Train Loop (on TRAIN split only) ===")
    for i, problem in enumerate(train_dataset):
        if i in completed_indices:
            print(f"Skipping already completed problem {i + 1}/{len(train_dataset)}: {problem['problem_id']}")
            continue

        problem_id = problem["problem_id"]
        problem_dir = run_dir / problem_id
        artifacts_dir = problem_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        print(f"    Problem {i + 1}/{len(train_dataset)}: {problem_id}")
        print("    Generating variant tree... ", end="")
        tree = vg.generate_variant_tree(
            model=model,
            tokenizer=tokenizer,
            has_native_chat_template=has_native_chat_template,
            sat_code=problem["prompt"],
            notes=problem.get("notes", "No description available"),
            problem_id=problem_id,
            run_dir=run_dir,
            max_new_tokens=800,
            max_retries=3,
        )
        print("Done!")
        print(f"    tree sizes -- hard: {len(tree['hard'])}  medium: {len(tree['medium'])}  easy: {len(tree['easy'])}")

        optimizer.train_on_tree(
            problem_id=problem_id,
            tree=tree,
            max_steps=TRAINING_HYPERPARAMS["max_steps"],
            target_accuracy=TRAINING_HYPERPARAMS["target_accuracy"],
            learning_rate=TRAINING_HYPERPARAMS["learning_rate"],
            group_size=TRAINING_HYPERPARAMS["group_size"],
            problem_artifacts_dir=artifacts_dir,
        )

        checkpoint_path = optimizer.save_problem_checkpoint(problem_id=problem_id)
        print(f"    Saved checkpoint: {checkpoint_path}")

        run_data = load_run_json(run_dir)
        run_data["completed_problem_indices"] = sorted(set(run_data.get("completed_problem_indices", [])) | {i})
        run_data["completed_problem_ids"] = sorted(set(run_data.get("completed_problem_ids", [])) | {problem_id})
        run_data.setdefault("problem_checkpoints", {})[problem_id] = str(checkpoint_path)
        run_data["latest_checkpoint"] = str(checkpoint_path)
        run_data["updated_at"] = utc_now_iso()
        save_run_json(run_dir, run_data)

    print("\n=== Evaluate POST-train accuracy on HELD-OUT split (never trained on) ===")
    post_eval_ds = Dataset.from_list(heldout_dataset)
    post_results = optimizer.evaluate_performance(post_eval_ds)

    final_path = optimizer.save_final_checkpoint()
    print(f"Final model saved to: {final_path}")

    run_data = load_run_json(run_dir)
    run_data["status"] = "completed"
    run_data["posttrain_evaluated"] = True
    run_data.setdefault("results", {})["posttrain_holdout"] = post_results
    run_data["latest_checkpoint"] = str(final_path)
    run_data["final_checkpoint"] = str(final_path)
    run_data["updated_at"] = utc_now_iso()
    save_run_json(run_dir, run_data)

    print("\n=== Stage 1 result ===")
    print(f"Pre-train (held-out):  {run_data['results'].get('pretrain_holdout')}")
    print(f"Post-train (held-out): {run_data['results'].get('posttrain_holdout')}")
    print("Compare this to the training-set numbers in the original result*.txt files --")
    print("a much smaller (or zero) gap here means V1's reported improvement was mostly training-set fit.")


if __name__ == "__main__":
    main()
