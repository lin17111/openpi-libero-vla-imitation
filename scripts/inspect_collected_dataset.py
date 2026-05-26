import dataclasses
import argparse
import json
import math
import pathlib
import random
from typing import Any

import numpy as np


DEFAULT_DATASET_DIR = pathlib.Path("results/libero_dataset_500")
SUMMARY_FILENAME = "dataset_summary.json"
ARRAY_FILENAMES = (
    "images.npy",
    "wrist_images.npy",
    "states.npy",
    "actions.npy",
    "rewards.npy",
    "dones.npy",
)


@dataclasses.dataclass
class Args:
    dataset_dir: str = str(DEFAULT_DATASET_DIR)
    seed: int = 7
    sample_episodes: int = 3


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _load_array(path: pathlib.Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _safe_min(arr: np.ndarray) -> Any:
    if arr.size == 0:
        return None
    return arr.min().item() if hasattr(arr.min(), "item") else arr.min()


def _safe_max(arr: np.ndarray) -> Any:
    if arr.size == 0:
        return None
    return arr.max().item() if hasattr(arr.max(), "item") else arr.max()


def _array_summary(arr: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": _safe_min(arr),
        "max": _safe_max(arr),
        "has_nan": bool(np.isnan(arr).any()) if np.issubdtype(arr.dtype, np.inexact) else False,
        "has_inf": bool(np.isinf(arr).any()) if np.issubdtype(arr.dtype, np.inexact) else False,
    }
    return summary


def _action_stats(actions: np.ndarray) -> dict[str, Any]:
    if actions.size == 0:
        return {
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
            "has_nan": False,
            "has_inf": False,
        }

    return {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "has_nan": bool(np.isnan(actions).any()),
        "has_inf": bool(np.isinf(actions).any()),
    }


def _print_array_stats(prefix: str, arr: np.ndarray) -> None:
    summary = _array_summary(arr)
    print(
        prefix,
        f"shape={summary['shape']}",
        f"dtype={summary['dtype']}",
        f"min={summary['min']}",
        f"max={summary['max']}",
        f"has_nan={summary['has_nan']}",
        f"has_inf={summary['has_inf']}",
    )


def inspect_dataset(dataset_dir: pathlib.Path, sample_episodes: int, seed: int) -> dict[str, Any]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    episode_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found in {dataset_dir}")

    rng = random.Random(seed)
    sample_count = min(sample_episodes, len(episode_dirs))
    sampled_dirs = rng.sample(episode_dirs, sample_count)

    print(f"Dataset dir: {dataset_dir}")
    print(f"Episode count: {len(episode_dirs)}")
    print(f"Random seed: {seed}")
    print(f"Sampled episodes: {[p.name for p in sampled_dirs]}")

    episode_summaries: list[dict[str, Any]] = []
    total_steps = 0
    episodes_with_any_nan = 0
    episodes_with_any_inf = 0

    for episode_dir in episode_dirs:
        meta_path = episode_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        arrays: dict[str, np.ndarray] = {}
        array_stats: dict[str, dict[str, Any]] = {}
        has_nan = False
        has_inf = False

        for filename in ARRAY_FILENAMES:
            array_path = episode_dir / filename
            if not array_path.exists():
                raise FileNotFoundError(f"Missing array file: {array_path}")
            arr = _load_array(array_path)
            arrays[filename] = arr
            stats = _array_summary(arr)
            array_stats[filename] = stats
            has_nan = has_nan or stats["has_nan"]
            has_inf = has_inf or stats["has_inf"]

        actions = arrays["actions.npy"]
        episode_len = int(meta.get("episode_len", actions.shape[0]))
        if actions.shape[0] != episode_len:
            raise ValueError(
                f"Episode length mismatch in {episode_dir.name}: "
                f"meta.json says {episode_len}, actions.npy has {actions.shape[0]}"
            )

        for name, arr in arrays.items():
            if arr.shape[0] != episode_len:
                raise ValueError(
                    f"Length mismatch in {episode_dir.name}: "
                    f"{name} has {arr.shape[0]} steps but episode_len is {episode_len}"
                )

        total_steps += episode_len
        episodes_with_any_nan += int(has_nan)
        episodes_with_any_inf += int(has_inf)

        episode_summary = {
            "episode_dir": episode_dir.name,
            "meta": meta,
            "episode_len": episode_len,
            "array_stats": array_stats,
            "action_stats": _action_stats(actions),
            "has_nan": has_nan,
            "has_inf": has_inf,
        }
        episode_summaries.append(episode_summary)

    avg_episode_len = total_steps / len(episode_dirs)

    print(f"Total steps: {total_steps}")
    print(f"Average episode_len: {avg_episode_len:.6f}")
    print(f"Episodes with NaN: {episodes_with_any_nan}")
    print(f"Episodes with Inf: {episodes_with_any_inf}")

    print("\nSampled episode details:")
    for episode_dir in sampled_dirs:
        episode_summary = next(item for item in episode_summaries if item["episode_dir"] == episode_dir.name)
        print(f"\n[{episode_dir.name}]")
        for filename in ARRAY_FILENAMES:
            _print_array_stats(f"  {filename}", _load_array(episode_dir / filename))
        action_stats = episode_summary["action_stats"]
        print(
            "  action_dim_stats",
            f"mean={np.array2string(np.asarray(action_stats['mean']), precision=6, separator=', ')}",
            f"std={np.array2string(np.asarray(action_stats['std']), precision=6, separator=', ')}",
            f"min={np.array2string(np.asarray(action_stats['min']), precision=6, separator=', ')}",
            f"max={np.array2string(np.asarray(action_stats['max']), precision=6, separator=', ')}",
            f"has_nan={action_stats['has_nan']}",
            f"has_inf={action_stats['has_inf']}",
        )

    summary = {
        "dataset_dir": str(dataset_dir),
        "episode_count": len(episode_dirs),
        "total_steps": total_steps,
        "average_episode_len": avg_episode_len,
        "sampled_episodes": [p.name for p in sampled_dirs],
        "episodes_with_nan": episodes_with_any_nan,
        "episodes_with_inf": episodes_with_any_inf,
        "episodes": episode_summaries,
    }

    summary_path = dataset_dir / SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nWrote summary: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a collected LIBERO dataset.")
    parser.add_argument("--dataset-dir", type=str, default=Args.dataset_dir)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--sample-episodes", type=int, default=Args.sample_episodes)
    args = parser.parse_args()
    inspect_dataset(pathlib.Path(args.dataset_dir), args.sample_episodes, args.seed)


if __name__ == "__main__":
    main()
