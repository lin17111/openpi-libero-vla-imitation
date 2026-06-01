"""Offline demo for the rule-based safety filter.

This script can run on a single actions file, recursively process a directory of
action files, or generate a synthetic sequence when no input is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge.safety_filter import SafetyFilter


DEFAULT_OUTPUT_DIR = Path("results_summary/safety_filter_demo")
DEFAULT_FIGURE_DIR = Path("figures/action_bridge")
DEFAULT_MAX_NORM = 1.5
DEFAULT_MAX_JUMP = 0.5


@dataclass
class SequenceResult:
    source_path: Path | None
    raw_actions: np.ndarray
    safe_actions: np.ndarray
    infos: list[dict[str, Any]]
    flattened_chunks: bool


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _normalize_actions_array(array: np.ndarray, source_name: str) -> tuple[np.ndarray, bool]:
    if array.ndim == 2:
        return array.astype(np.float32, copy=False), False
    if array.ndim == 3:
        print(
            "policy chunks are flattened for offline safety demo and do not necessarily represent executed actions."
        )
        flattened = array.reshape(-1, array.shape[-1])
        return flattened.astype(np.float32, copy=False), True
    raise ValueError(f"Unsupported action array shape from {source_name}: {array.shape}. Expected [T, D] or [N, K, D].")


def _select_input_files(input_dir: Path) -> list[Path]:
    selected: dict[Path, tuple[int, Path]] = {}
    priorities = {"actions.npy": 0, "adapted_actions.npy": 1}

    for name in ("actions.npy", "adapted_actions.npy"):
        for path in input_dir.rglob(name):
            if not path.is_file():
                continue
            rel_parent = path.relative_to(input_dir).parent
            priority = priorities[name]
            current = selected.get(rel_parent)
            if current is None or priority > current[0]:
                selected[rel_parent] = (priority, path)

    return [item[1] for item in sorted(selected.values(), key=lambda x: str(x[1]))]


def _generate_random_actions(length: int = 256, action_dim: int = 7) -> np.ndarray:
    rng = np.random.default_rng(7)
    motion = np.cumsum(rng.normal(scale=0.12, size=(length, min(6, action_dim))), axis=0)
    motion += rng.normal(scale=0.03, size=motion.shape)
    if motion.shape[1] > 0:
        spike_indices = [48, 96, 160, 208]
        for idx in spike_indices:
            if idx < length:
                motion[idx] += rng.normal(loc=0.0, scale=1.5, size=motion.shape[1])
    if action_dim > motion.shape[1]:
        gripper = rng.uniform(-1.5, 1.5, size=(length, action_dim - motion.shape[1]))
        actions = np.concatenate([motion, gripper], axis=1)
    else:
        actions = motion[:, :action_dim]
    return actions.astype(np.float32, copy=False)


def _motion_dims(action_dim: int, filter_obj: SafetyFilter) -> list[int]:
    resolved = filter_obj._resolved
    if resolved is None:
        raise RuntimeError("SafetyFilter was not resolved; filter a sequence first.")
    return list(resolved.motion_dims)


def _motion_norms(actions: np.ndarray, dims: list[int]) -> np.ndarray:
    if actions.size == 0 or not dims:
        return np.array([], dtype=np.float32)
    return np.linalg.norm(actions[:, dims].astype(np.float64), axis=1).astype(np.float32)


def _motion_jumps(actions: np.ndarray, dims: list[int]) -> np.ndarray:
    if actions.shape[0] < 2 or not dims:
        return np.array([], dtype=np.float32)
    diffs = np.diff(actions[:, dims].astype(np.float64), axis=0)
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


def _aggregate_metrics(
    raw_actions: np.ndarray,
    safe_actions: np.ndarray,
    infos: list[dict[str, Any]],
    filter_obj: SafetyFilter,
) -> dict[str, Any]:
    dims = _motion_dims(raw_actions.shape[1], filter_obj)
    raw_norms = _motion_norms(raw_actions, dims)
    safe_norms = _motion_norms(safe_actions, dims)
    raw_jumps = _motion_jumps(raw_actions, dims)
    safe_jumps = _motion_jumps(safe_actions, dims)

    safety_trigger_count = sum(bool(info["safety_triggered"]) for info in infos)
    bound_violation_count = sum(bool(info["bound_violation"]) for info in infos)
    norm_violation_count = sum(bool(info["norm_violation"]) for info in infos)
    jump_violation_count = sum(bool(info["jump_violation"]) for info in infos)
    workspace_violation_count = sum(bool(info["workspace_violation"]) for info in infos)
    abnormal_sequence_count = sum(bool(info["abnormal_sequence"]) for info in infos)
    rejected_action_count = sum(bool(info["reject_action"]) for info in infos)

    trigger_reason_counts: dict[str, int] = {}
    for info in infos:
        for reason in info.get("trigger_reasons", []):
            trigger_reason_counts[reason] = trigger_reason_counts.get(reason, 0) + 1

    return {
        "num_steps": int(raw_actions.shape[0]),
        "safety_trigger_count": int(safety_trigger_count),
        "safety_trigger_ratio": float(safety_trigger_count / max(int(raw_actions.shape[0]), 1)),
        "bound_violation_count": int(bound_violation_count),
        "norm_violation_count": int(norm_violation_count),
        "jump_violation_count": int(jump_violation_count),
        "workspace_violation_count": int(workspace_violation_count),
        "abnormal_sequence_count": int(abnormal_sequence_count),
        "rejected_action_count": int(rejected_action_count),
        "raw_motion_norm_mean": float(raw_norms.mean()) if raw_norms.size else None,
        "raw_motion_norm_max": float(raw_norms.max()) if raw_norms.size else None,
        "safe_motion_norm_mean": float(safe_norms.mean()) if safe_norms.size else None,
        "safe_motion_norm_max": float(safe_norms.max()) if safe_norms.size else None,
        "raw_action_jump_mean": float(raw_jumps.mean()) if raw_jumps.size else None,
        "raw_action_jump_max": float(raw_jumps.max()) if raw_jumps.size else None,
        "safe_action_jump_mean": float(safe_jumps.mean()) if safe_jumps.size else None,
        "safe_action_jump_max": float(safe_jumps.max()) if safe_jumps.size else None,
        "trigger_reason_counts": trigger_reason_counts,
    }


def _combine_batch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_steps": 0,
            "safety_trigger_count": 0,
            "safety_trigger_ratio": 0.0,
            "bound_violation_count": 0,
            "norm_violation_count": 0,
            "jump_violation_count": 0,
            "workspace_violation_count": 0,
            "abnormal_sequence_count": 0,
            "rejected_action_count": 0,
            "raw_motion_norm_mean": None,
            "raw_motion_norm_max": None,
            "safe_motion_norm_mean": None,
            "safe_motion_norm_max": None,
            "raw_action_jump_mean": None,
            "raw_action_jump_max": None,
            "safe_action_jump_mean": None,
            "safe_action_jump_max": None,
            "trigger_reason_counts": {},
        }

    total_steps = sum(int(row["num_steps"]) for row in rows)
    total_jump_steps = sum(max(int(row["num_steps"]) - 1, 0) for row in rows)

    def _weighted_mean(key: str, weight_key: str = "num_steps") -> float | None:
        numerator = 0.0
        denominator = 0
        for row in rows:
            value = row.get(key)
            weight = int(row.get(weight_key, 0))
            if value is None or weight <= 0:
                continue
            numerator += float(value) * weight
            denominator += weight
        return (numerator / denominator) if denominator > 0 else None

    def _max_value(key: str) -> float | None:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        return float(max(values)) if values else None

    trigger_reason_counts: dict[str, int] = {}
    count_keys = [
        "safety_trigger_count",
        "bound_violation_count",
        "norm_violation_count",
        "jump_violation_count",
        "workspace_violation_count",
        "abnormal_sequence_count",
        "rejected_action_count",
    ]
    aggregate_counts = {key: sum(int(row.get(key, 0)) for row in rows) for key in count_keys}
    for row in rows:
        for reason, count in row.get("trigger_reason_counts", {}).items():
            trigger_reason_counts[reason] = trigger_reason_counts.get(reason, 0) + int(count)

    return {
        "num_steps": int(total_steps),
        "safety_trigger_count": int(aggregate_counts["safety_trigger_count"]),
        "safety_trigger_ratio": float(aggregate_counts["safety_trigger_count"] / max(total_steps, 1)),
        "bound_violation_count": int(aggregate_counts["bound_violation_count"]),
        "norm_violation_count": int(aggregate_counts["norm_violation_count"]),
        "jump_violation_count": int(aggregate_counts["jump_violation_count"]),
        "workspace_violation_count": int(aggregate_counts["workspace_violation_count"]),
        "abnormal_sequence_count": int(aggregate_counts["abnormal_sequence_count"]),
        "rejected_action_count": int(aggregate_counts["rejected_action_count"]),
        "raw_motion_norm_mean": _weighted_mean("raw_motion_norm_mean"),
        "raw_motion_norm_max": _max_value("raw_motion_norm_max"),
        "safe_motion_norm_mean": _weighted_mean("safe_motion_norm_mean"),
        "safe_motion_norm_max": _max_value("safe_motion_norm_max"),
        "raw_action_jump_mean": _weighted_mean("raw_action_jump_mean", weight_key="num_steps_minus_one"),
        "raw_action_jump_max": _max_value("raw_action_jump_max"),
        "safe_action_jump_mean": _weighted_mean("safe_action_jump_mean", weight_key="num_steps_minus_one"),
        "safe_action_jump_max": _max_value("safe_action_jump_max"),
        "trigger_reason_counts": trigger_reason_counts,
        "_num_jump_steps": int(total_jump_steps),
    }


def _save_sequence_outputs(output_dir: Path, result: SequenceResult, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "raw_actions.npy", result.raw_actions)
    np.save(output_dir / "safe_actions.npy", result.safe_actions)

    info_path = output_dir / "safety_info.json"
    metrics_path = output_dir / "safety_metrics.json"
    info_payload = {
        "source_path": str(result.source_path) if result.source_path is not None else None,
        "flattened_chunks": bool(result.flattened_chunks),
        "infos": result.infos,
        "metrics": metrics,
    }
    info_path.write_text(json.dumps(_jsonable(info_payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_batch_summary(output_dir: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    json_path = output_dir / "safety_batch_summary.json"
    csv_path = output_dir / "safety_batch_summary.csv"

    json_payload = {
        "aggregate": aggregate,
        "files": rows,
    }
    json_path.write_text(json.dumps(_jsonable(json_payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fieldnames = [
        "relative_path",
        "num_steps",
        "safety_trigger_count",
        "safety_trigger_ratio",
        "bound_violation_count",
        "norm_violation_count",
        "jump_violation_count",
        "workspace_violation_count",
        "abnormal_sequence_count",
        "rejected_action_count",
        "raw_motion_norm_mean",
        "raw_motion_norm_max",
        "safe_motion_norm_mean",
        "safe_motion_norm_max",
        "raw_action_jump_mean",
        "raw_action_jump_max",
        "safe_action_jump_mean",
        "safe_action_jump_max",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
        writer.writerow({**{k: aggregate.get(k) for k in fieldnames}, "relative_path": "__aggregate__"})


def _plot_figures(
    figure_dir: Path,
    raw_actions: np.ndarray,
    safe_actions: np.ndarray,
    infos: list[dict[str, Any]],
    title_suffix: str = "",
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for path in figure_dir.glob("safety_*.png"):
        path.unlink()
    for path in figure_dir.glob("safety_*.pdf"):
        path.unlink()

    length = min(raw_actions.shape[0], 300)
    if length == 0:
        print("[warning] No data available for plotting.")
        return

    motion_dims = list(range(min(6, raw_actions.shape[1])))
    fig, axes = plt.subplots(len(motion_dims), 1, figsize=(12, max(2.2, 1.6 * len(motion_dims))), sharex=True)
    if len(motion_dims) == 1:
        axes = [axes]
    steps = np.arange(length)
    for idx, dim in enumerate(motion_dims):
        ax = axes[idx]
        ax.plot(steps, raw_actions[:length, dim], color="#C08D8D", linewidth=1.1, label="raw")
        ax.plot(steps, safe_actions[:length, dim], color="#8FB9B2", linewidth=1.1, label="safe")
        ax.set_ylabel(f"dim {dim}")
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5, color="#D4D8DD")
        if idx == 0:
            ax.legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"Raw vs Safe Motion Actions{title_suffix}", y=0.995)
    fig.tight_layout()
    fig.savefig(figure_dir / "safety_raw_vs_filtered_action.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "safety_raw_vs_filtered_action.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figure_dir / 'safety_raw_vs_filtered_action.png'}")
    print(f"saved {figure_dir / 'safety_raw_vs_filtered_action.pdf'}")

    raw_jumps = _motion_jumps(raw_actions[:length], motion_dims)
    safe_jumps = _motion_jumps(safe_actions[:length], motion_dims)
    jump_steps = np.arange(1, 1 + max(raw_jumps.shape[0], safe_jumps.shape[0]))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    if raw_jumps.size:
        ax.plot(jump_steps[: raw_jumps.shape[0]], raw_jumps, color="#C08D8D", linewidth=1.2, label="raw jump")
    else:
        print("[warning] No raw action jumps to plot.")
    if safe_jumps.size:
        ax.plot(jump_steps[: safe_jumps.shape[0]], safe_jumps, color="#8FB9B2", linewidth=1.2, label="safe jump")
    else:
        print("[warning] No safe action jumps to plot.")
    ax.set_title("Motion Jump Before / After Safety Filter")
    ax.set_xlabel("step")
    ax.set_ylabel("L2 jump")
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5, color="#D4D8DD")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "safety_action_jump_before_after.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "safety_action_jump_before_after.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figure_dir / 'safety_action_jump_before_after.png'}")
    print(f"saved {figure_dir / 'safety_action_jump_before_after.pdf'}")

    raw_norms = np.linalg.norm(raw_actions[:length, motion_dims].astype(np.float64), axis=1) if motion_dims else np.zeros(length)
    safe_norms = np.linalg.norm(safe_actions[:length, motion_dims].astype(np.float64), axis=1) if motion_dims else np.zeros(length)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(steps, raw_norms, color="#C08D8D", linewidth=1.2, label="raw norm")
    ax.plot(steps, safe_norms, color="#8FB9B2", linewidth=1.2, label="safe norm")
    ax.set_title("Motion Norm Before / After Safety Filter")
    ax.set_xlabel("step")
    ax.set_ylabel("L2 norm")
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5, color="#D4D8DD")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "safety_action_norm_before_after.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "safety_action_norm_before_after.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figure_dir / 'safety_action_norm_before_after.png'}")
    print(f"saved {figure_dir / 'safety_action_norm_before_after.pdf'}")

    trigger = np.asarray([1 if info["safety_triggered"] else 0 for info in infos[:length]], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.step(steps, trigger, where="mid", color="#7F8C99", linewidth=1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Safety Trigger Timeline")
    ax.set_xlabel("step")
    ax.set_ylabel("trigger")
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5, color="#D4D8DD")
    fig.tight_layout()
    fig.savefig(figure_dir / "safety_trigger_timeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "safety_trigger_timeline.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figure_dir / 'safety_trigger_timeline.png'}")
    print(f"saved {figure_dir / 'safety_trigger_timeline.pdf'}")

    reason_order = ["bound", "norm", "jump", "workspace", "consecutive_triggers", "reject_on_violation"]
    counts = {reason: 0 for reason in reason_order}
    for info in infos:
        for reason in info.get("trigger_reasons", []):
            if reason in counts:
                counts[reason] += 1
            else:
                counts[reason] = counts.get(reason, 0) + 1

    fig, ax = plt.subplots(figsize=(11, 3.8))
    labels = ["bound", "norm", "jump", "workspace", "abnormal", "reject"]
    values = [
        counts.get("bound", 0),
        counts.get("norm", 0),
        counts.get("jump", 0),
        counts.get("workspace", 0),
        counts.get("consecutive_triggers", 0),
        counts.get("reject_on_violation", 0),
    ]
    ax.bar(labels, values, color=["#C6A8A8", "#8FB9B2", "#D1C19A", "#B7B0C7", "#7F8C99", "#A8B5A8"], edgecolor="#4B5563", linewidth=0.6)
    ax.set_title("Safety Trigger Count by Reason")
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5, color="#D4D8DD")
    fig.tight_layout()
    fig.savefig(figure_dir / "safety_trigger_count.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "safety_trigger_count.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {figure_dir / 'safety_trigger_count.png'}")
    print(f"saved {figure_dir / 'safety_trigger_count.pdf'}")


def _run_sequence(
    source_path: Path | None,
    raw_actions: np.ndarray,
    output_dir: Path,
    figure_dir: Path,
    args: argparse.Namespace,
    flattened_chunks: bool = False,
) -> tuple[SequenceResult, dict[str, Any]]:
    filter_obj = SafetyFilter(
        action_low=args.action_low,
        action_high=args.action_high,
        max_norm=args.max_norm,
        max_jump=args.max_jump,
        max_consecutive_triggers=args.max_consecutive_triggers,
        reject_on_violation=args.reject_on_violation,
        fallback_to_previous=True,
    )
    safe_actions, infos = filter_obj.filter_sequence(raw_actions)
    flattened_chunks = False
    metrics = _aggregate_metrics(raw_actions, safe_actions, infos, filter_obj)
    result = SequenceResult(
        source_path=source_path,
        raw_actions=raw_actions,
        safe_actions=safe_actions,
        infos=infos,
        flattened_chunks=flattened_chunks,
    )
    _save_sequence_outputs(output_dir, result, metrics)
    title_suffix = f" [{source_path.name}]" if source_path is not None else " [synthetic]"
    _plot_figures(figure_dir, raw_actions, safe_actions, infos, title_suffix=title_suffix)
    return result, metrics


def _process_single_file(path: Path, output_dir: Path, figure_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    array = _load_npy(path)
    raw_actions, flattened_chunks = _normalize_actions_array(array, str(path))
    _, metrics = _run_sequence(path, raw_actions, output_dir, figure_dir, args, flattened_chunks=flattened_chunks)
    print(json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False))
    return metrics


def _process_batch(input_dir: Path, output_dir: Path, figure_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    files = _select_input_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"No actions.npy or adapted_actions.npy files were found under {input_dir}. "
            "Provide --actions_file, or point --input_dir to a directory containing action files."
        )

    rows = []
    source_for_figures: SequenceResult | None = None
    selected_action_dim: int | None = None

    safe_root = output_dir / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        rel = file_path.relative_to(input_dir)
        output_file_dir = safe_root / rel.parent
        output_file_dir.mkdir(parents=True, exist_ok=True)

        array = _load_npy(file_path)
        raw_actions, flattened_chunks = _normalize_actions_array(array, str(file_path))
        if selected_action_dim is None:
            selected_action_dim = raw_actions.shape[1]
        elif raw_actions.shape[1] != selected_action_dim:
            raise ValueError(
                f"Inconsistent action dimensions in batch: expected {selected_action_dim}, got {raw_actions.shape[1]} for {file_path}."
            )

        filter_obj = SafetyFilter(
            action_low=args.action_low,
            action_high=args.action_high,
            max_norm=args.max_norm,
            max_jump=args.max_jump,
            max_consecutive_triggers=args.max_consecutive_triggers,
            reject_on_violation=args.reject_on_violation,
            fallback_to_previous=True,
        )
        safe_actions, infos = filter_obj.filter_sequence(raw_actions)
        np.save(output_file_dir / "safe_actions.npy", safe_actions)

        metrics = _aggregate_metrics(raw_actions, safe_actions, infos, filter_obj)
        row = {
            "relative_path": str(rel),
            **{k: v for k, v in metrics.items() if k != "trigger_reason_counts"},
            "trigger_reason_counts": metrics.get("trigger_reason_counts", {}),
            "num_steps_minus_one": max(int(metrics["num_steps"]) - 1, 0),
        }
        rows.append(row)
        if source_for_figures is None:
            source_for_figures = SequenceResult(
                source_path=file_path,
                raw_actions=raw_actions,
                safe_actions=safe_actions,
                infos=infos,
                flattened_chunks=flattened_chunks,
            )

    aggregate_metrics = _combine_batch_rows(rows)
    aggregate_metrics.pop("_num_jump_steps", None)
    _write_batch_summary(output_dir, rows, aggregate_metrics)

    if source_for_figures is not None:
        _plot_figures(figure_dir, source_for_figures.raw_actions, source_for_figures.safe_actions, source_for_figures.infos, title_suffix=f" [{source_for_figures.source_path.name}]")

    print(json.dumps(_jsonable(aggregate_metrics), indent=2, ensure_ascii=False))
    return aggregate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline safety filter demo.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--actions_file", type=Path, default=None, help="Single raw/adapted actions.npy file.")
    group.add_argument("--input_dir", type=Path, default=None, help="Recursively process action files under a directory.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--max_norm", type=float, default=DEFAULT_MAX_NORM)
    parser.add_argument("--max_jump", type=float, default=DEFAULT_MAX_JUMP)
    parser.add_argument("--reject_on_violation", action="store_true")
    parser.add_argument("--max_consecutive_triggers", type=int, default=5)
    parser.add_argument("--action_low", type=float, default=-1.0)
    parser.add_argument("--action_high", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    if args.actions_file is not None:
        _process_single_file(args.actions_file, args.output_dir, args.figure_dir, args)
        print(f"saved {args.output_dir / 'raw_actions.npy'}")
        print(f"saved {args.output_dir / 'safe_actions.npy'}")
        print(f"saved {args.output_dir / 'safety_info.json'}")
        print(f"saved {args.output_dir / 'safety_metrics.json'}")
        return

    if args.input_dir is not None:
        _process_batch(args.input_dir, args.output_dir, args.figure_dir, args)
        print(f"saved {args.output_dir / 'safety_batch_summary.json'}")
        print(f"saved {args.output_dir / 'safety_batch_summary.csv'}")
        return

    random_actions = _generate_random_actions()
    filter_obj = SafetyFilter(
        action_low=args.action_low,
        action_high=args.action_high,
        max_norm=args.max_norm,
        max_jump=args.max_jump,
        max_consecutive_triggers=args.max_consecutive_triggers,
        reject_on_violation=args.reject_on_violation,
        fallback_to_previous=True,
    )
    safe_actions, infos = filter_obj.filter_sequence(random_actions)
    np.save(args.output_dir / "safe_actions.npy", safe_actions)
    np.save(args.output_dir / "raw_actions.npy", random_actions)
    synthetic_metrics = _aggregate_metrics(random_actions, safe_actions, infos, filter_obj)
    info_payload = {
        "source_path": None,
        "flattened_chunks": False,
        "infos": infos,
        "metrics": synthetic_metrics,
    }
    (args.output_dir / "safety_info.json").write_text(json.dumps(_jsonable(info_payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "safety_metrics.json").write_text(json.dumps(_jsonable(synthetic_metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_figures(args.figure_dir, random_actions, safe_actions, infos, title_suffix=" [synthetic]")
    print(json.dumps(_jsonable(synthetic_metrics), indent=2, ensure_ascii=False))
    print(f"saved {args.output_dir / 'raw_actions.npy'}")
    print(f"saved {args.output_dir / 'safe_actions.npy'}")
    print(f"saved {args.output_dir / 'safety_info.json'}")
    print(f"saved {args.output_dir / 'safety_metrics.json'}")


if __name__ == "__main__":
    main()
