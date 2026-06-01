"""Audit action space statistics for rollout and trajectory files.

This script separates:
1. executed actions: arrays shaped [T, action_dim]
2. policy action chunks: arrays shaped [N, chunk_len, action_dim] or a single
   chunk shaped [chunk_len, action_dim] when the file/key name suggests chunk data.

It scans an input directory recursively for .npy and .npz files, computes summary
statistics, and writes JSON plus matplotlib figures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_KEYS = (
    "actions",
    "executed_actions",
    "raw_actions",
    "action_chunks",
    "policy_chunks",
)
ACTION_NAME_HINTS = ("action", "actions", "chunk")


def is_action_candidate(path: Path) -> bool:
    if path.suffix.lower() not in {".npy", ".npz"}:
        return False
    lowered = path.name.lower()
    return any(hint in lowered for hint in ACTION_NAME_HINTS)


def find_action_files(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.rglob("*") if path.is_file() and is_action_candidate(path)]
    return sorted(files)


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _load_npz(path: Path) -> list[tuple[str, np.ndarray]]:
    loaded: list[tuple[str, np.ndarray]] = []
    with np.load(path, allow_pickle=False) as data:
        for key in ACTION_KEYS:
            if key in data:
                loaded.append((key, data[key]))
    return loaded


def load_action_arrays(file_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    try:
        if file_path.suffix.lower() == ".npy":
            array = _load_npy(file_path)
            entries.append(
                {
                    "source_file": file_path,
                    "source_name": file_path.name,
                    "key": None,
                    "array": array,
                }
            )
            return entries

        if file_path.suffix.lower() == ".npz":
            loaded = _load_npz(file_path)
            if not loaded:
                print(f"[warning] No supported action keys found in npz file: {file_path}")
                return entries
            for key, array in loaded:
                entries.append(
                    {
                        "source_file": file_path,
                        "source_name": f"{file_path.name}::{key}",
                        "key": key,
                        "array": array,
                    }
                )
            return entries
    except Exception as exc:  # noqa: BLE001
        print(f"[warning] Failed to load {file_path}: {exc}")

    return entries


def classify_action_array(array: np.ndarray, source_name: str) -> tuple[str | None, np.ndarray | None]:
    if not isinstance(array, np.ndarray):
        print(f"[warning] Skipping non-numpy array from {source_name}")
        return None, None

    if array.ndim == 2:
        if "chunk" in source_name.lower():
            return "policy_chunks", array[None, ...]
        return "executed_actions", array

    if array.ndim == 3:
        return "policy_chunks", array

    print(f"[warning] Skipping {source_name}: unsupported ndim={array.ndim}, shape={array.shape}")
    return None, None


def _init_dim_stats(action_dim: int) -> dict[str, np.ndarray]:
    return {
        "count": np.array(0, dtype=np.int64),
        "sum": np.zeros(action_dim, dtype=np.float64),
        "sumsq": np.zeros(action_dim, dtype=np.float64),
        "min": np.full(action_dim, np.inf, dtype=np.float64),
        "max": np.full(action_dim, -np.inf, dtype=np.float64),
    }


def _update_dim_stats(stats: dict[str, np.ndarray], values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return
    stats["count"] += values.shape[0]
    stats["sum"] += values.sum(axis=0)
    stats["sumsq"] += np.square(values).sum(axis=0)
    stats["min"] = np.minimum(stats["min"], values.min(axis=0))
    stats["max"] = np.maximum(stats["max"], values.max(axis=0))


def _finalize_dim_stats(stats: dict[str, np.ndarray]) -> dict[str, list[float]]:
    count = int(stats["count"])
    if count <= 0:
        return {"mean": [], "std": [], "min": [], "max": []}

    mean = stats["sum"] / count
    var = np.maximum(stats["sumsq"] / count - np.square(mean), 0.0)
    std = np.sqrt(var)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": stats["min"].tolist(),
        "max": stats["max"].tolist(),
    }


def _scalar_stats(values: list[np.ndarray] | list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "max": None}
    flat_parts: list[np.ndarray] = []
    for value in values:
        flat_parts.append(np.asarray(value, dtype=np.float64).reshape(-1))
    flat = np.concatenate(flat_parts) if flat_parts else np.array([], dtype=np.float64)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "max": float(flat.max()),
    }


def compute_executed_stats(entries: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    compatible_arrays: list[np.ndarray] = []
    source_files: set[str] = set()
    action_dim: int | None = None

    dim_stats: dict[str, np.ndarray] | None = None
    norm_values: list[np.ndarray] = []
    jump_values: list[np.ndarray] = []

    for entry in entries:
        array = np.asarray(entry["array"])
        source_name = str(entry["source_name"])
        source_files.add(str(entry["source_file"]))

        if array.ndim != 2:
            print(f"[warning] Executed action entry is not 2D after classification: {source_name}")
            continue

        if action_dim is None:
            action_dim = int(array.shape[1])
            dim_stats = _init_dim_stats(action_dim)
        elif array.shape[1] != action_dim:
            print(
                f"[warning] Skipping executed array {source_name} with action_dim={array.shape[1]} "
                f"(expected {action_dim})"
            )
            continue

        compatible_arrays.append(array)
        source_files.add(str(entry["source_file"]))
        assert dim_stats is not None
        _update_dim_stats(dim_stats, array)
        norm_values.append(np.linalg.norm(array.astype(np.float64), axis=1))

        if array.shape[0] > 1:
            jump_values.append(np.linalg.norm(np.diff(array.astype(np.float64), axis=0), axis=1))

    total_steps = int(sum(arr.shape[0] for arr in compatible_arrays))
    action_stats = {
        "num_files": len(source_files) if compatible_arrays else 0,
        "total_steps": total_steps,
        "action_dim": action_dim,
        "per_dim": _finalize_dim_stats(dim_stats) if dim_stats is not None else {"mean": [], "std": [], "min": [], "max": []},
        "action_norm": _scalar_stats(norm_values),
        "action_jump": _scalar_stats(jump_values),
    }
    plot_data = {
        "arrays": compatible_arrays,
        "norms": norm_values,
        "jumps": jump_values,
    }
    return action_stats, plot_data


def compute_chunk_stats(entries: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    compatible_arrays: list[np.ndarray] = []
    source_files: set[str] = set()
    action_dim: int | None = None
    chunk_len: int | None = None

    dim_stats: dict[str, np.ndarray] | None = None
    horizon_norm_sum: np.ndarray | None = None
    horizon_norm_count: np.ndarray | None = None
    delta_values: list[np.ndarray] = []
    smooth_values: list[np.ndarray] = []

    for entry in entries:
        array = np.asarray(entry["array"])
        source_name = str(entry["source_name"])
        source_files.add(str(entry["source_file"]))

        if array.ndim != 3:
            print(f"[warning] Policy chunk entry is not 3D after classification: {source_name}")
            continue

        if action_dim is None:
            action_dim = int(array.shape[2])
            chunk_len = int(array.shape[1])
            dim_stats = _init_dim_stats(action_dim)
            horizon_norm_sum = np.zeros(chunk_len, dtype=np.float64)
            horizon_norm_count = np.zeros(chunk_len, dtype=np.int64)
        else:
            if array.shape[2] != action_dim:
                print(
                    f"[warning] Skipping chunk array {source_name} with action_dim={array.shape[2]} "
                    f"(expected {action_dim})"
                )
                continue
            if array.shape[1] != chunk_len:
                print(
                    f"[warning] Skipping chunk array {source_name} with chunk_len={array.shape[1]} "
                    f"(expected {chunk_len})"
                )
                continue

        compatible_arrays.append(array)
        source_files.add(str(entry["source_file"]))
        assert dim_stats is not None and horizon_norm_sum is not None and horizon_norm_count is not None
        _update_dim_stats(dim_stats, array.reshape(-1, action_dim))

        norms = np.linalg.norm(array.astype(np.float64), axis=2)
        horizon_norm_sum += norms.sum(axis=0)
        horizon_norm_count += norms.shape[0]

        deltas = np.diff(array.astype(np.float64), axis=1)
        if deltas.size > 0:
            delta_norms = np.linalg.norm(deltas, axis=2)
            delta_values.append(delta_norms)
            smooth_values.append(1.0 / (1.0 + delta_norms))

    if chunk_len is None:
        horizon_norm_mean: list[float] = []
    else:
        assert horizon_norm_sum is not None and horizon_norm_count is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            horizon_norm_mean = np.divide(
                horizon_norm_sum,
                horizon_norm_count,
                out=np.zeros_like(horizon_norm_sum),
                where=horizon_norm_count > 0,
            ).tolist()

    delta_concat = np.concatenate([values.reshape(-1) for values in delta_values]) if delta_values else np.array([])
    smooth_concat = np.concatenate([values.reshape(-1) for values in smooth_values]) if smooth_values else np.array([])

    chunk_stats = {
        "num_files": len(source_files) if compatible_arrays else 0,
        "num_chunks": int(sum(arr.shape[0] for arr in compatible_arrays)),
        "chunk_len": chunk_len,
        "action_dim": action_dim,
        "per_dim": _finalize_dim_stats(dim_stats) if dim_stats is not None else {"mean": [], "std": [], "min": [], "max": []},
        "horizon_norm_mean": horizon_norm_mean,
        "chunk_delta": _scalar_stats(delta_concat.tolist()),
        "chunk_temporal_smoothness": _scalar_stats(smooth_concat.tolist()),
    }
    plot_data = {
        "horizon_norm_mean": horizon_norm_mean,
        "delta_values": delta_values,
    }
    return chunk_stats, plot_data


def plot_figures(
    output_dir: Path,
    figure_dir: Path,
    executed_plot_data: dict[str, Any],
    chunk_plot_data: dict[str, Any],
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    executed_arrays = executed_plot_data.get("arrays", [])
    executed_norms = executed_plot_data.get("norms", [])
    executed_jumps = executed_plot_data.get("jumps", [])

    if executed_arrays:
        first = np.asarray(executed_arrays[0])
        action_dim = int(first.shape[1])
        fig, axes = plt.subplots(action_dim, 1, figsize=(10, max(3, 2.2 * action_dim)), sharex=False)
        if action_dim == 1:
            axes = [axes]
        for dim, ax in enumerate(axes):
            values = np.concatenate([arr[:, dim].reshape(-1) for arr in executed_arrays])
            ax.hist(values, bins=60, alpha=0.85, color="#2c7fb8")
            ax.set_title(f"Executed action dim {dim}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
        fig.suptitle("Executed Action Dimension Distributions")
        fig.tight_layout()
        fig.savefig(figure_dir / "action_dim_distribution.png", dpi=200)
        plt.close(fig)
    else:
        print("[warning] No executed action data available for action_dim_distribution.png")

    if executed_norms:
        norms = np.concatenate(executed_norms)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(norms, bins=80, color="#41ab5d", alpha=0.85)
        ax.set_title("Executed Action Norm Distribution")
        ax.set_xlabel("||action||")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(figure_dir / "action_norm_hist.png", dpi=200)
        plt.close(fig)
    else:
        print("[warning] No executed action norms available for action_norm_hist.png")

    if executed_jumps:
        jumps = np.concatenate(executed_jumps)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(jumps, bins=80, color="#f16913", alpha=0.85)
        ax.set_title("Executed Action Jump Distribution")
        ax.set_xlabel(r"$||a_t - a_{t-1}||$")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(figure_dir / "action_jump_hist.png", dpi=200)
        plt.close(fig)
    else:
        print("[warning] No executed action jumps available for action_jump_hist.png")

    horizon_norm_mean = chunk_plot_data.get("horizon_norm_mean", [])
    if horizon_norm_mean:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(horizon_norm_mean))
        ax.plot(x, horizon_norm_mean, marker="o", linewidth=2.0, color="#756bb1", label="Mean action norm")
        ax.set_title("Chunk Temporal Profile")
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("Mean ||action||")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "chunk_temporal_profile.png", dpi=200)
        plt.close(fig)
    else:
        print("[warning] No policy chunk data available for chunk_temporal_profile.png")

    delta_values = chunk_plot_data.get("delta_values", [])
    if delta_values:
        max_delta_len = max(values.shape[1] for values in delta_values)
        delta_sum = np.zeros(max_delta_len, dtype=np.float64)
        delta_count = np.zeros(max_delta_len, dtype=np.int64)
        for values in delta_values:
            delta_sum[: values.shape[1]] += values.sum(axis=0)
            delta_count[: values.shape[1]] += values.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_mean = np.divide(delta_sum, delta_count, out=np.zeros_like(delta_sum), where=delta_count > 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(max_delta_len)
        ax.plot(x, delta_mean, marker="o", linewidth=2.0, color="#dd1c77", label="Mean delta norm")
        ax.set_title("Chunk Delta Profile")
        ax.set_xlabel("Chunk step transition")
        ax.set_ylabel("Mean ||a_{t+1} - a_t||")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "chunk_delta_profile.png", dpi=200)
        plt.close(fig)
    else:
        print("[warning] No policy chunk data available for chunk_delta_profile.png")


def build_empty_executed_stats() -> dict[str, Any]:
    return {
        "num_files": 0,
        "total_steps": 0,
        "action_dim": None,
        "per_dim": {"mean": [], "std": [], "min": [], "max": []},
        "action_norm": {"mean": None, "std": None, "max": None},
        "action_jump": {"mean": None, "std": None, "max": None},
    }


def build_empty_chunk_stats() -> dict[str, Any]:
    return {
        "num_files": 0,
        "num_chunks": 0,
        "chunk_len": None,
        "action_dim": None,
        "per_dim": {"mean": [], "std": [], "min": [], "max": []},
        "horizon_norm_mean": [],
        "chunk_delta": {"mean": None, "std": None, "max": None},
        "chunk_temporal_smoothness": {"mean": None, "std": None, "max": None},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit executed actions and policy action chunks.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Trajectory or log directory to scan.")
    parser.add_argument("--output_dir", type=Path, default=Path("results_summary"), help="Directory for JSON output.")
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("figures/action_bridge"),
        help="Directory for generated figures.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    action_files = find_action_files(input_dir)
    print(f"[info] Scanned: {input_dir}")
    print(f"[info] Found {len(action_files)} candidate action files")

    loaded_entries: list[dict[str, Any]] = []
    for file_path in action_files:
        loaded_entries.extend(load_action_arrays(file_path))

    if not loaded_entries:
        print("[warning] No supported action arrays were loaded.")
        print("[warning] Pass a trajectory/log directory that contains .npy or .npz action files.")

    classified_entries: list[dict[str, Any]] = []
    for entry in loaded_entries:
        kind, normalized = classify_action_array(entry["array"], str(entry["source_name"]))
        if kind is None or normalized is None:
            continue
        classified_entries.append({**entry, "kind": kind, "array": normalized})

    executed_entries = [entry for entry in classified_entries if entry["kind"] == "executed_actions"]
    chunk_entries = [entry for entry in classified_entries if entry["kind"] == "policy_chunks"]

    print(f"[info] Executed action arrays: {len(executed_entries)}")
    print(f"[info] Policy chunk arrays: {len(chunk_entries)}")

    executed_stats = build_empty_executed_stats()
    executed_plot_data: dict[str, Any] = {"arrays": [], "norms": [], "jumps": []}
    if executed_entries:
        executed_stats, executed_plot_data = compute_executed_stats(executed_entries)
    else:
        print("[warning] No executed actions found.")

    chunk_stats = build_empty_chunk_stats()
    chunk_plot_data: dict[str, Any] = {"horizon_norm_mean": [], "delta_values": []}
    if chunk_entries:
        chunk_stats, chunk_plot_data = compute_chunk_stats(chunk_entries)
    else:
        print("[warning] No policy chunks found.")

    plot_figures(output_dir, figure_dir, executed_plot_data, chunk_plot_data)

    source_files = sorted({str(entry["source_file"]) for entry in classified_entries})
    summary = {
        "executed_actions": executed_stats,
        "policy_chunks": chunk_stats,
        "source_files": source_files,
    }

    output_json = output_dir / "action_space_stats.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[info] Wrote JSON summary to: {output_json}")
    print(f"[info] Wrote figures to: {figure_dir}")


if __name__ == "__main__":
    main()
