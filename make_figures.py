from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
DOCS_DIR = ROOT / "docs"

TEACHER_SUCCESS = 98.4
SINGLE_STEP_SUCCESS = 0.0
CHUNK_SUCCESS = 70.0
SINGLE_STEP_VAL_LOSS = 0.014299813140822078
CHUNK_VAL_LOSS = 0.015247196419122717

LOCAL_SINGLE_LOG = ROOT / "results" / "libero_student_bc" / "train_log.json"
LOCAL_CHUNK_LOG = ROOT / "results" / "libero_student_chunk_bc" / "train_log.json"
SOURCE_SINGLE_LOG = Path("/home/lin17/openpi/results/libero_student_bc/train_log.json")
SOURCE_CHUNK_LOG = Path("/home/lin17/openpi/results/libero_student_chunk_bc/train_log.json")


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 0.18, height: float = 0.14) -> tuple[float, float]:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="#222222",
        facecolor="#F8FAFC",
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=10)
    return x, y


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.4,
        color="#222222",
    )
    ax.add_patch(arrow)


def fig_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 2.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    labels = [
        "π0.5 Teacher\nRollout",
        "Trajectory\nCollection",
        "Single-step\nStudent BC",
        "Action-chunk\nStudent BC",
        "Closed-loop\nEvaluation",
    ]
    xs = [0.08, 0.29, 0.50, 0.71, 0.92]
    for idx, (x, label) in enumerate(zip(xs, labels)):
        draw_box(ax, (x, 0.52), label, width=0.17 if idx != 2 else 0.18, height=0.16)
        if idx < len(xs) - 1:
            draw_arrow(ax, (x + 0.09, 0.52), (xs[idx + 1] - 0.09, 0.52))

    ax.text(
        0.5,
        0.16,
        "OpenPI + LIBERO / MuJoCo imitation learning pipeline",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )
    save(fig, "pipeline_overview")


def fig_success_rate_comparison() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    names = ["Teacher π0.5", "Single-step\nStudent", "Action-chunk\nStudent"]
    rates = [TEACHER_SUCCESS, SINGLE_STEP_SUCCESS, CHUNK_SUCCESS]
    colors = ["#1D4ED8", "#94A3B8", "#0F766E"]

    bars = ax.bar(names, rates, color=colors, width=0.58)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Closed-loop success rate (%)")
    ax.set_title("Closed-loop success rate comparison on LIBERO spatial")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 2.2,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save(fig, "success_rate_comparison")


def fig_student_comparison() -> None:
    fig, (ax_loss, ax_success) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    names = ["Single-step", "Action-chunk"]
    val_losses = [SINGLE_STEP_VAL_LOSS, CHUNK_VAL_LOSS]
    success_rates = [0.0, 70.0]

    loss_bars = ax_loss.bar(names, val_losses, color=["#94A3B8", "#0F766E"], width=0.58)
    ax_loss.set_ylabel("Offline val loss")
    ax_loss.set_title("Offline loss is similar")
    ax_loss.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_loss.set_axisbelow(True)
    for bar, loss in zip(loss_bars, val_losses):
        ax_loss.text(
            bar.get_x() + bar.get_width() / 2,
            loss + 0.00025,
            f"{loss:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    success_bars = ax_success.bar(names, success_rates, color=["#94A3B8", "#0F766E"], width=0.58)
    ax_success.set_ylim(0, 105)
    ax_success.set_ylabel("Closed-loop success rate (%)")
    ax_success.set_title("Closed-loop performance diverges")
    ax_success.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_success.set_axisbelow(True)
    for bar, rate in zip(success_bars, success_rates):
        ax_success.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 2.2,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("Offline MSE is similar, but closed-loop behavior is very different", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "student_comparison")


def fig_error_accumulation_concept() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.3))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Single-step path
    ax.text(0.16, 0.86, "Single-step BC", ha="center", va="center", fontsize=12, weight="bold")
    steps_single = [
        (0.10, 0.62, "small action error"),
        (0.31, 0.62, "state drift"),
        (0.52, 0.62, "OOD observation"),
        (0.73, 0.62, "failure"),
    ]
    for x, y, text in steps_single:
        draw_box(ax, (x, y), text, width=0.16, height=0.12)
    for (x1, y1, _), (x2, y2, _) in zip(steps_single[:-1], steps_single[1:]):
        draw_arrow(ax, (x1 + 0.08, y1), (x2 - 0.08, y2))
    ax.text(0.42, 0.39, "compounding error", ha="center", va="center", fontsize=10, color="#B91C1C")

    # Chunk path
    ax.text(0.16, 0.22, "Action-chunk BC", ha="center", va="center", fontsize=12, weight="bold")
    steps_chunk = [
        (0.10, 0.00 + 0.18, "temporal segment"),
        (0.34, 0.00 + 0.18, "reduced drift"),
        (0.58, 0.00 + 0.18, "stable rollout"),
        (0.82, 0.00 + 0.18, "success"),
    ]
    for x, y, text in steps_chunk:
        draw_box(ax, (x, y), text, width=0.18, height=0.12)
    for (x1, y1, _), (x2, y2, _) in zip(steps_chunk[:-1], steps_chunk[1:]):
        draw_arrow(ax, (x1 + 0.09, y1), (x2 - 0.09, y2))
    ax.text(0.48, 0.04, "temporally consistent action segment", ha="center", va="center", fontsize=10, color="#0F766E")

    save(fig, "error_accumulation_concept")


def load_training_log(path_candidates: list[Path]) -> dict | None:
    for path in path_candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def fig_training_loss_curves() -> bool:
    single_log = load_training_log([LOCAL_SINGLE_LOG, SOURCE_SINGLE_LOG])
    chunk_log = load_training_log([LOCAL_CHUNK_LOG, SOURCE_CHUNK_LOG])

    if single_log is None or chunk_log is None:
        missing = []
        if single_log is None:
            missing.append(str(LOCAL_SINGLE_LOG))
            missing.append(str(SOURCE_SINGLE_LOG))
        if chunk_log is None:
            missing.append(str(LOCAL_CHUNK_LOG))
            missing.append(str(SOURCE_CHUNK_LOG))

        missing_doc = DOCS_DIR / "missing_training_logs.md"
        missing_doc.write_text(
            "# Missing Training Logs\n\n"
            "The following training log files were not found:\n\n"
            + "\n".join(f"- `{item}`" for item in missing)
            + "\n\n"
            "To generate the training curves, copy the logs from the original OpenPI directory into the showcase project, then rerun `python make_figures.py`.\n",
            encoding="utf-8",
        )
        print(f"saved {missing_doc}")
        return False

    fig, ax = plt.subplots(figsize=(10, 5.2))

    def extract(log: dict) -> tuple[list[int], list[float], list[float]]:
        epochs = [entry["epoch"] for entry in log["epochs"]]
        train = [entry["train_loss"] for entry in log["epochs"]]
        val = [entry["val_loss"] for entry in log["epochs"]]
        return epochs, train, val

    epochs_s, train_s, val_s = extract(single_log)
    epochs_c, train_c, val_c = extract(chunk_log)

    ax.plot(epochs_s, train_s, color="#2563EB", linewidth=2.0, label="Single-step train")
    ax.plot(epochs_s, val_s, color="#60A5FA", linewidth=2.0, linestyle="--", label="Single-step val")
    ax.plot(epochs_c, train_c, color="#0F766E", linewidth=2.0, label="Chunk train")
    ax.plot(epochs_c, val_c, color="#34D399", linewidth=2.0, linestyle="--", label="Chunk val")

    ax.set_title("Training loss curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "training_loss_curves")
    return True


def main() -> None:
    ensure_dirs()
    fig_pipeline_overview()
    fig_success_rate_comparison()
    fig_student_comparison()
    fig_error_accumulation_concept()
    fig_training_loss_curves()


if __name__ == "__main__":
    main()
