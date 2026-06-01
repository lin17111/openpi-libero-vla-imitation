"""Offline demo for the LIBERO action adapter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPLCONFIGDIR = Path("/dev/shm") / "codex" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bridge.action_adapter import ActionAdapter


DEFAULT_OUTPUT_DIR = Path("results_summary/action_adapter_demo")
DEFAULT_FIGURE_DIR = Path("figures/action_bridge")
DEFAULT_RANDOM_STEPS = 120
DEFAULT_RANDOM_SEED = 7
MAX_PLOT_STEPS = 300


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return str(value)
    return value


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Actions file does not exist: {path}")
    return np.load(path, allow_pickle=False)


def _prepare_actions(array: np.ndarray, source_label: str) -> tuple[np.ndarray, bool, list[str]]:
    if array.ndim == 2:
        return np.asarray(array), False, []
    if array.ndim == 3:
        flattened = np.asarray(array).reshape(-1, array.shape[-1])
        return flattened, True, [
            "policy chunks are flattened for offline adapter demo and do not necessarily represent executed actions."
        ]
    raise ValueError(
        f"{source_label} has unsupported shape={array.shape}. Expected [T, action_dim] or [N, chunk_len, action_dim]."
    )


def _find_actions_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(path for path in input_dir.rglob("actions.npy") if path.is_file())


def _generate_random_actions(num_steps: int = DEFAULT_RANDOM_STEPS, action_dim: int = 7, seed: int = DEFAULT_RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    actions = rng.normal(loc=0.0, scale=0.8, size=(num_steps, action_dim)).astype(np.float64)
    if action_dim >= 6:
        actions[:, :6] += np.linspace(-0.8, 0.8, num_steps, dtype=np.float64)[:, None]
    if action_dim >= 7:
        gripper = np.sin(np.linspace(0.0, 8.0 * np.pi, num_steps, dtype=np.float64))
        actions[:, 6] = gripper + 0.25 * rng.normal(size=num_steps)
    return actions


def _motion_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "max": None}
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"mean": float(flat.mean()), "max": float(flat.max())}


def _compute_motion_values(actions: np.ndarray, motion_dims: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not motion_dims:
        empty = np.zeros(actions.shape[0], dtype=np.float64)
        return empty, np.zeros(0, dtype=np.float64)
    motion = np.asarray(actions[:, motion_dims], dtype=np.float64)
    norms = np.linalg.norm(motion, axis=1)
    jumps = np.linalg.norm(np.diff(motion, axis=0), axis=1) if motion.shape[0] > 1 else np.zeros(0, dtype=np.float64)
    return norms, jumps


def _compute_metrics(
    raw_actions: np.ndarray,
    adapted_actions: np.ndarray,
    infos: list[dict[str, Any]],
    motion_dims: list[int],
) -> dict[str, Any]:
    raw_motion_norms, raw_motion_jumps = _compute_motion_values(raw_actions, motion_dims)
    adapted_motion_norms, adapted_motion_jumps = _compute_motion_values(adapted_actions, motion_dims)

    metrics = {
        "num_steps": int(adapted_actions.shape[0]),
        "raw_motion_norm_mean": _motion_stats(raw_motion_norms)["mean"],
        "raw_motion_norm_max": _motion_stats(raw_motion_norms)["max"],
        "adapted_motion_norm_mean": _motion_stats(adapted_motion_norms)["mean"],
        "adapted_motion_norm_max": _motion_stats(adapted_motion_norms)["max"],
        "raw_motion_jump_mean": _motion_stats(raw_motion_jumps)["mean"],
        "raw_motion_jump_max": _motion_stats(raw_motion_jumps)["max"],
        "adapted_motion_jump_mean": _motion_stats(adapted_motion_jumps)["mean"],
        "adapted_motion_jump_max": _motion_stats(adapted_motion_jumps)["max"],
        "norm_clipped_count": int(sum(bool(info.get("clipped_by_norm", False)) for info in infos)),
        "jump_clipped_count": int(sum(bool(info.get("clipped_by_jump", False)) for info in infos)),
        "bound_clipped_count": int(sum(bool(info.get("clipped_by_bound", False)) for info in infos)),
        "gripper_thresholded_count": int(sum(bool(info.get("gripper_thresholded", False)) for info in infos)),
    }
    return metrics


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _plot_raw_vs_adapted(
    raw_actions: np.ndarray,
    adapted_actions: np.ndarray,
    motion_dims: list[int],
    figure_dir: Path,
) -> None:
    if not motion_dims:
        return

    steps = min(raw_actions.shape[0], MAX_PLOT_STEPS)
    raw = raw_actions[:steps, motion_dims[:6]]
    adapted = adapted_actions[:steps, motion_dims[:6]]
    num_dims = raw.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(14, max(2.0, 2.2 * num_dims)), sharex=True)
    if num_dims == 1:
        axes = [axes]
    x = np.arange(steps)
    for idx, ax in enumerate(axes):
        ax.plot(x, raw[:, idx], label="raw", linewidth=1.2)
        ax.plot(x, adapted[:, idx], label="adapted", linewidth=1.2)
        ax.set_ylabel(f"dim {motion_dims[idx]}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.suptitle("Raw vs Adapted Motion Action")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "adapter_raw_vs_adapted.png", dpi=200)
    plt.close(fig)


def _plot_jump_compare(
    raw_actions: np.ndarray,
    adapted_actions: np.ndarray,
    motion_dims: list[int],
    figure_dir: Path,
) -> None:
    if not motion_dims:
        return

    raw_motion = np.asarray(raw_actions[:, motion_dims], dtype=np.float64)
    adapted_motion = np.asarray(adapted_actions[:, motion_dims], dtype=np.float64)
    raw_jump = np.linalg.norm(np.diff(raw_motion, axis=0), axis=1) if raw_motion.shape[0] > 1 else np.zeros(0)
    adapted_jump = (
        np.linalg.norm(np.diff(adapted_motion, axis=0), axis=1) if adapted_motion.shape[0] > 1 else np.zeros(0)
    )

    steps = min(raw_jump.shape[0], MAX_PLOT_STEPS)
    x = np.arange(1, steps + 1)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, raw_jump[:steps], label="raw motion jump", linewidth=1.3)
    ax.plot(x, adapted_jump[:steps], label="adapted motion jump", linewidth=1.3)
    ax.set_title("Motion Jump Before vs After Adapter")
    ax.set_xlabel("step")
    ax.set_ylabel("L2 jump")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "adapter_action_jump_before_after.png", dpi=200)
    plt.close(fig)


def _plot_norm_compare(
    raw_actions: np.ndarray,
    adapted_actions: np.ndarray,
    motion_dims: list[int],
    figure_dir: Path,
) -> None:
    if not motion_dims:
        return

    raw_motion = np.asarray(raw_actions[:, motion_dims], dtype=np.float64)
    adapted_motion = np.asarray(adapted_actions[:, motion_dims], dtype=np.float64)
    raw_norm = np.linalg.norm(raw_motion, axis=1) if raw_motion.size else np.zeros(0)
    adapted_norm = np.linalg.norm(adapted_motion, axis=1) if adapted_motion.size else np.zeros(0)

    steps = min(raw_norm.shape[0], MAX_PLOT_STEPS)
    x = np.arange(steps)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, raw_norm[:steps], label="raw motion norm", linewidth=1.3)
    ax.plot(x, adapted_norm[:steps], label="adapted motion norm", linewidth=1.3)
    ax.set_title("Motion Norm Before vs After Adapter")
    ax.set_xlabel("step")
    ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "adapter_action_norm_before_after.png", dpi=200)
    plt.close(fig)


def _plot_gripper_compare(
    raw_actions: np.ndarray,
    adapted_actions: np.ndarray,
    gripper_dims: list[int],
    figure_dir: Path,
) -> None:
    if not gripper_dims:
        return

    steps = min(raw_actions.shape[0], MAX_PLOT_STEPS)
    raw = raw_actions[:steps, gripper_dims]
    adapted = adapted_actions[:steps, gripper_dims]
    fig, axes = plt.subplots(len(gripper_dims), 1, figsize=(14, max(2.0, 2.2 * len(gripper_dims))), sharex=True)
    if len(gripper_dims) == 1:
        axes = [axes]
    x = np.arange(steps)
    for idx, ax in enumerate(axes):
        ax.plot(x, raw[:, idx], label="raw", linewidth=1.2)
        ax.plot(x, adapted[:, idx], label="adapted", linewidth=1.2)
        ax.set_ylabel(f"dim {gripper_dims[idx]}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.suptitle("Raw vs Adapted Gripper Action")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "adapter_gripper_compare.png", dpi=200)
    plt.close(fig)


def _build_adapter(args: argparse.Namespace) -> ActionAdapter:
    return ActionAdapter(
        mode=args.mode,
        max_norm=args.max_norm,
        max_jump=args.max_jump,
        smooth_alpha=args.smooth_alpha,
        threshold_gripper=args.threshold_gripper,
    )


def _run_single_sequence(
    raw_actions: np.ndarray,
    adapter: ActionAdapter,
    *,
    source_label: str,
    output_dir: Path,
    figure_dir: Path,
    save_outputs: bool,
    plot_outputs: bool = True,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    local_notes = list(notes or [])
    if raw_actions.ndim == 2:
        processed_actions = np.asarray(raw_actions)
        flattened = False
    elif raw_actions.ndim == 3:
        processed_actions, flattened, chunk_notes = _prepare_actions(raw_actions, source_label)
        local_notes.extend(chunk_notes)
    else:
        raise ValueError(
            f"{source_label} has unsupported shape={raw_actions.shape}. Expected [T, action_dim] or [N, chunk_len, action_dim]."
        )

    adapted_actions, infos = adapter.adapt_sequence(processed_actions)
    motion_dims, gripper_dims = adapter.resolve_dims(processed_actions.shape[1])
    metrics = _compute_metrics(processed_actions, adapted_actions, infos, motion_dims)

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_npy(output_dir / "raw_actions.npy", processed_actions)
        _save_npy(output_dir / "adapted_actions.npy", adapted_actions)
        _save_json(
            output_dir / "adapter_info.json",
            {
                "source_label": source_label,
                "input_shape": list(raw_actions.shape),
                "processed_shape": list(processed_actions.shape),
                "flattened_policy_chunks": bool(flattened),
                "notes": local_notes,
                "mode": adapter.mode,
                "adapter_config": {
                    "max_norm": adapter.max_norm,
                    "max_jump": adapter.max_jump,
                    "smooth_alpha": adapter.smooth_alpha,
                    "scale": adapter.scale,
                    "threshold_gripper": adapter.threshold_gripper,
                    "gripper_threshold": adapter.gripper_threshold,
                },
                "motion_dims": motion_dims,
                "gripper_dims": gripper_dims,
                "infos": infos,
            },
        )
        _save_json(output_dir / "adapter_metrics.json", metrics)

    if plot_outputs:
        _plot_raw_vs_adapted(processed_actions, adapted_actions, motion_dims, figure_dir)
        _plot_jump_compare(processed_actions, adapted_actions, motion_dims, figure_dir)
        _plot_norm_compare(processed_actions, adapted_actions, motion_dims, figure_dir)
        _plot_gripper_compare(processed_actions, adapted_actions, gripper_dims, figure_dir)

    return {
        "source_label": source_label,
        "input_shape": list(raw_actions.shape),
        "processed_shape": list(processed_actions.shape),
        "flattened_policy_chunks": bool(flattened),
        "notes": local_notes,
        "motion_dims": motion_dims,
        "gripper_dims": gripper_dims,
        "metrics": metrics,
        "processed_actions": processed_actions,
        "adapted_actions": adapted_actions,
        "infos": infos,
    }


def _relative_output_dir(file_path: Path, input_dir: Path) -> Path:
    relative = file_path.relative_to(input_dir).parent
    if relative == Path("."):
        return Path("root")
    return relative


def _run_batch(
    input_dir: Path,
    adapter_args: argparse.Namespace,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    files = _find_actions_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No actions.npy files found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    batch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    aggregate_raw_norms: list[np.ndarray] = []
    aggregate_adapted_norms: list[np.ndarray] = []
    aggregate_raw_jumps: list[np.ndarray] = []
    aggregate_adapted_jumps: list[np.ndarray] = []
    first_plot_source: dict[str, Any] | None = None

    for file_path in files:
        raw_array = _load_npy(file_path)
        _, _, notes = _prepare_actions(raw_array, str(file_path))
        adapter = _build_adapter(adapter_args)
        result = _run_single_sequence(
            raw_array,
            adapter,
            source_label=str(file_path),
            output_dir=output_dir / "adapted" / _relative_output_dir(file_path, input_dir),
            figure_dir=figure_dir,
            save_outputs=True,
            plot_outputs=first_plot_source is None,
            notes=notes,
        )

        metrics = result["metrics"]
        motion_dims = result["motion_dims"]
        raw_motion_norms, raw_motion_jumps = _compute_motion_values(result["processed_actions"], motion_dims)
        adapted_motion_norms, adapted_motion_jumps = _compute_motion_values(result["adapted_actions"], motion_dims)
        aggregate_raw_norms.append(raw_motion_norms)
        aggregate_adapted_norms.append(adapted_motion_norms)
        aggregate_raw_jumps.append(raw_motion_jumps)
        aggregate_adapted_jumps.append(adapted_motion_jumps)

        row = {
            "source_file": str(file_path),
            "relative_path": str(file_path.relative_to(input_dir)),
            "output_file": str(output_dir / "adapted" / _relative_output_dir(file_path, input_dir) / "adapted_actions.npy"),
            "flattened_policy_chunks": bool(result["flattened_policy_chunks"]),
            **metrics,
        }
        batch_rows.append(row)
        summary_rows.append({k: _jsonable(v) for k, v in row.items()})

        if first_plot_source is None:
            first_plot_source = result

    aggregate = {
        "num_files": len(batch_rows),
        "num_steps": int(sum(int(row["num_steps"]) for row in batch_rows)),
        "raw_motion_norm_mean": float(np.concatenate(aggregate_raw_norms).mean()) if aggregate_raw_norms else None,
        "raw_motion_norm_max": float(np.concatenate(aggregate_raw_norms).max()) if aggregate_raw_norms else None,
        "adapted_motion_norm_mean": float(np.concatenate(aggregate_adapted_norms).mean()) if aggregate_adapted_norms else None,
        "adapted_motion_norm_max": float(np.concatenate(aggregate_adapted_norms).max()) if aggregate_adapted_norms else None,
        "raw_motion_jump_mean": float(np.concatenate(aggregate_raw_jumps).mean()) if aggregate_raw_jumps else None,
        "raw_motion_jump_max": float(np.concatenate(aggregate_raw_jumps).max()) if aggregate_raw_jumps else None,
        "adapted_motion_jump_mean": float(np.concatenate(aggregate_adapted_jumps).mean()) if aggregate_adapted_jumps else None,
        "adapted_motion_jump_max": float(np.concatenate(aggregate_adapted_jumps).max()) if aggregate_adapted_jumps else None,
        "norm_clipped_count": int(sum(int(row["norm_clipped_count"]) for row in batch_rows)),
        "jump_clipped_count": int(sum(int(row["jump_clipped_count"]) for row in batch_rows)),
        "bound_clipped_count": int(sum(int(row["bound_clipped_count"]) for row in batch_rows)),
        "gripper_thresholded_count": int(sum(int(row["gripper_thresholded_count"]) for row in batch_rows)),
        "figure_source_file": first_plot_source["source_label"] if first_plot_source else None,
    }

    _save_json(
        output_dir / "adapter_batch_summary.json",
        {
            "input_dir": str(input_dir),
            "mode": adapter_args.mode,
            "adapter_config": {
                "max_norm": adapter_args.max_norm,
                "max_jump": adapter_args.max_jump,
                "smooth_alpha": adapter_args.smooth_alpha,
                "threshold_gripper": adapter_args.threshold_gripper,
            },
            "aggregate": aggregate,
            "files": summary_rows,
        },
    )

    csv_path = output_dir / "adapter_batch_summary.csv"
    fieldnames = [
        "source_file",
        "relative_path",
        "output_file",
        "flattened_policy_chunks",
        "num_steps",
        "raw_motion_norm_mean",
        "raw_motion_norm_max",
        "adapted_motion_norm_mean",
        "adapted_motion_norm_max",
        "raw_motion_jump_mean",
        "raw_motion_jump_max",
        "adapted_motion_jump_mean",
        "adapted_motion_jump_max",
        "norm_clipped_count",
        "jump_clipped_count",
        "bound_clipped_count",
        "gripper_thresholded_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in batch_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline demo for LIBERO action adapter.")
    parser.add_argument("--actions_file", type=Path, default=None, help="Single actions.npy file to process.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Recursively process every actions.npy under this directory when --actions_file is not set.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="clip_and_smooth",
        choices=["identity", "clip_only", "smooth_only", "clip_and_smooth"],
        help="Adapter preset.",
    )
    parser.add_argument("--max_norm", type=float, default=1.5, help="Motion L2 norm clip threshold.")
    parser.add_argument("--max_jump", type=float, default=0.5, help="Motion jump clip threshold.")
    parser.add_argument("--smooth_alpha", type=float, default=0.7, help="Exponential smoothing alpha.")
    parser.add_argument(
        "--threshold_gripper",
        action="store_true",
        help="Threshold gripper values to {-1, 1} after clamping to [-1, 1].",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for adapter JSON and adapted_actions.npy outputs.",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for matplotlib figures.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.actions_file is not None and args.input_dir is not None:
        print("[info] --actions_file takes precedence over --input_dir; only the single file will be processed.")

    if args.actions_file is not None:
        raw_actions = _load_npy(args.actions_file)
        _, flattened, notes = _prepare_actions(raw_actions, str(args.actions_file))
        adapter = _build_adapter(args)
        result = _run_single_sequence(
            raw_actions,
            adapter,
            source_label=str(args.actions_file),
            output_dir=args.output_dir,
            figure_dir=args.figure_dir,
            save_outputs=True,
            notes=notes,
        )
        print(
            f"[info] Processed single file: {args.actions_file} | "
            f"input_shape={list(raw_actions.shape)} | processed_shape={result['processed_shape']} | "
            f"flattened_policy_chunks={flattened}"
        )
        return

    if args.input_dir is not None:
        _run_batch(args.input_dir, args, args.output_dir, args.figure_dir)
        print(f"[info] Batch adapter summary written to: {args.output_dir}")
        return

    raw_actions = _generate_random_actions()
    adapter = _build_adapter(args)
    result = _run_single_sequence(
        raw_actions,
        adapter,
        source_label="random_demo",
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        save_outputs=True,
        notes=["Generated random actions because neither --actions_file nor --input_dir was provided."],
    )
    print(
        f"[info] Generated random demo sequence | input_shape={list(raw_actions.shape)} | "
        f"processed_shape={result['processed_shape']}"
    )


if __name__ == "__main__":
    main()
