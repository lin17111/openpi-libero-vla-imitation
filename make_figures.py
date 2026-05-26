from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
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


def choose_serif_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Times New Roman", "DejaVu Serif"):
        if candidate in available:
            return candidate
    return "DejaVu Serif"


AVAILABLE_FONT = choose_serif_font()

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [AVAILABLE_FONT, "DejaVu Serif", "Times New Roman"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#4B5563",
        "axes.labelcolor": "#374151",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "text.color": "#111827",
        "legend.frameon": False,
        "axes.titleweight": "regular",
    }
)

PALETTE = {
    "teacher": "#8AA0B8",
    "single": "#C6A8A8",
    "chunk": "#9FC9C1",
    "dark": "#4B5563",
    "grid": "#D4D8DD",
    "teacher_light": "#F3F6F9",
    "single_light": "#F6F0F0",
    "chunk_light": "#F0F7F5",
}


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    for path in (png_path, pdf_path):
        if path.exists():
            path.unlink()
    fig.patch.set_facecolor("white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


def add_box(
    ax: plt.Axes,
    center: tuple[float, float],
    text: str,
    *,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str = "#4B5563",
    fontsize: float = 10.2,
    weight: str = "regular",
) -> None:
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.95,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=10,
            linewidth=0.95,
            color=PALETTE["dark"],
        )
    )


def fig_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")

    ax.text(0.5, 0.88, "OpenPI + LIBERO Imitation Learning Pipeline", ha="center", va="center", fontsize=13.2)

    labels = [
        "π0.5 Teacher\nRollout",
        "Trajectory\nCollection",
        "Single-step\nStudent BC",
        "Action-chunk\nStudent BC",
        "Closed-loop\nEvaluation",
    ]
    xs = [0.10, 0.30, 0.50, 0.70, 0.90]
    fills = [PALETTE["teacher_light"], PALETTE["teacher_light"], PALETTE["single_light"], PALETTE["chunk_light"], PALETTE["teacher_light"]]
    widths = [0.16, 0.16, 0.17, 0.18, 0.17]

    for idx, (x, label) in enumerate(zip(xs, labels)):
        add_box(ax, (x, 0.52), label, width=widths[idx], height=0.16, facecolor=fills[idx], fontsize=10.2)
        if idx < len(xs) - 1:
            add_arrow(ax, (x + widths[idx] / 2 + 0.01, 0.52), (xs[idx + 1] - widths[idx + 1] / 2 - 0.01, 0.52))

    ax.text(
        0.5,
        0.16,
        "Teacher rollout, trajectory collection, student distillation, and closed-loop evaluation.",
        ha="center",
        va="center",
        fontsize=9.8,
        color=PALETTE["dark"],
    )
    fig.tight_layout()
    save(fig, "pipeline_overview")


def fig_success_rate_comparison() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    names = ["Teacher π0.5", "Single-step\nStudent", "Action-chunk\nStudent"]
    rates = [TEACHER_SUCCESS, SINGLE_STEP_SUCCESS, CHUNK_SUCCESS]
    colors = [PALETTE["teacher"], PALETTE["single"], PALETTE["chunk"]]

    bars = ax.bar(names, rates, color=colors, width=0.55, edgecolor=PALETTE["dark"], linewidth=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Closed-loop success rate (%)")
    ax.set_title("Closed-loop Rollout Success Rate", pad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.28, color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 2.0,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.0,
        )

    fig.tight_layout()
    save(fig, "success_rate_comparison")


def fig_student_comparison() -> None:
    fig, (ax_loss, ax_success) = plt.subplots(1, 2, figsize=(11.8, 4.8))

    names = ["Single-step", "Action-chunk"]
    val_losses = [SINGLE_STEP_VAL_LOSS, CHUNK_VAL_LOSS]
    success_rates = [0.0, 70.0]

    loss_bars = ax_loss.bar(
        names,
        val_losses,
        color=[PALETTE["single"], PALETTE["chunk"]],
        width=0.56,
        edgecolor=PALETTE["dark"],
        linewidth=0.6,
    )
    ax_loss.set_ylabel("Offline validation MSE")
    ax_loss.set_title("Offline Fit", pad=10)
    ax_loss.set_ylim(0.0136, 0.0158)
    ax_loss.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.28, color=PALETTE["grid"])
    ax_loss.set_axisbelow(True)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    for bar, loss in zip(loss_bars, val_losses):
        ax_loss.text(
            bar.get_x() + bar.get_width() / 2,
            loss + 0.00006,
            f"{loss:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.8,
        )
    ax_loss.text(
        0.5,
        0.08,
        "similar offline loss",
        transform=ax_loss.transAxes,
            ha="center",
            va="center",
            fontsize=9.4,
            color=PALETTE["dark"],
        )

    success_bars = ax_success.bar(
        names,
        success_rates,
        color=[PALETTE["single"], PALETTE["chunk"]],
        width=0.56,
        edgecolor=PALETTE["dark"],
        linewidth=0.6,
    )
    ax_success.set_ylim(0, 105)
    ax_success.set_ylabel("Closed-loop success rate (%)")
    ax_success.set_title("Closed-loop Performance", pad=10)
    ax_success.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.28, color=PALETTE["grid"])
    ax_success.set_axisbelow(True)
    ax_success.spines["top"].set_visible(False)
    ax_success.spines["right"].set_visible(False)
    for bar, rate in zip(success_bars, success_rates):
        ax_success.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 2.0,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.8,
        )
    ax_success.text(
        0.5,
        0.08,
        "very different rollout success",
        transform=ax_success.transAxes,
            ha="center",
            va="center",
            fontsize=9.4,
            color=PALETTE["dark"],
        )

    fig.suptitle("Offline Fit vs Closed-loop Performance", y=1.03, fontsize=13.0)
    fig.text(
        0.5,
        0.015,
        "Similar offline loss, very different rollout success.",
        ha="center",
        va="bottom",
        fontsize=9.6,
        color=PALETTE["dark"],
    )
    fig.tight_layout()
    save(fig, "student_comparison")


def fig_error_accumulation_concept() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")

    ax.text(0.5, 0.92, "Single-step behavior cloning", ha="center", va="center", fontsize=12.4)
    single_steps = [
        (0.12, 0.74, "Small action error"),
        (0.35, 0.74, "State drift"),
        (0.58, 0.74, "Out-of-distribution\nobservation"),
        (0.81, 0.74, "Task failure"),
    ]
    widths_single = [0.17, 0.15, 0.22, 0.14]
    for (x, y, text), width in zip(single_steps, widths_single):
        add_box(ax, (x, y), text, width=width, height=0.13, facecolor=PALETTE["single_light"], fontsize=9.8)
    for (x1, y1, _), (x2, y2, _), w1, w2 in zip(single_steps[:-1], single_steps[1:], widths_single[:-1], widths_single[1:]):
        add_arrow(ax, (x1 + w1 / 2 + 0.01, y1), (x2 - w2 / 2 - 0.01, y2))
    ax.text(0.5, 0.57, "Compounding error", ha="center", va="center", fontsize=10.3, color=PALETTE["dark"])

    ax.text(0.5, 0.39, "Action-chunk behavior cloning", ha="center", va="center", fontsize=12.4)
    chunk_steps = [
        (0.12, 0.21, "Temporally consistent\naction segment"),
        (0.35, 0.21, "Reduced state drift"),
        (0.58, 0.21, "Stable rollout"),
        (0.81, 0.21, "Task success"),
    ]
    widths_chunk = [0.20, 0.17, 0.14, 0.14]
    for (x, y, text), width in zip(chunk_steps, widths_chunk):
        add_box(ax, (x, y), text, width=width, height=0.13, facecolor=PALETTE["chunk_light"], fontsize=9.8)
    for (x1, y1, _), (x2, y2, _), w1, w2 in zip(chunk_steps[:-1], chunk_steps[1:], widths_chunk[:-1], widths_chunk[1:]):
        add_arrow(ax, (x1 + w1 / 2 + 0.01, y1), (x2 - w2 / 2 - 0.01, y2))
    ax.text(0.5, 0.06, "Chunk-level temporal consistency", ha="center", va="center", fontsize=10.3, color=PALETTE["dark"])

    fig.tight_layout()
    save(fig, "error_accumulation_concept")


def load_training_log(path_candidates: list[Path]) -> dict | None:
    for path in path_candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def write_missing_logs_doc(missing_paths: list[str]) -> None:
    missing_doc = DOCS_DIR / "missing_training_logs.md"
    missing_doc.write_text(
        "# Missing Training Logs\n\n"
        "The following training log files were not found:\n\n"
        + "\n".join(f"- `{item}`" for item in missing_paths)
        + "\n\n"
        "To generate the training curves, copy the logs from the original OpenPI directory into the showcase project, then rerun `python make_figures.py`.\n",
        encoding="utf-8",
    )
    print(f"saved {missing_doc}")


def fig_training_loss_curves() -> bool:
    single_log = load_training_log([LOCAL_SINGLE_LOG, SOURCE_SINGLE_LOG])
    chunk_log = load_training_log([LOCAL_CHUNK_LOG, SOURCE_CHUNK_LOG])

    if single_log is None or chunk_log is None:
        missing = []
        if single_log is None:
            missing.extend([str(LOCAL_SINGLE_LOG), str(SOURCE_SINGLE_LOG)])
        if chunk_log is None:
            missing.extend([str(LOCAL_CHUNK_LOG), str(SOURCE_CHUNK_LOG)])
        write_missing_logs_doc(missing)
        return False

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    fig.set_facecolor("white")

    def extract(log: dict) -> tuple[list[int], list[float], list[float]]:
        epochs = [entry["epoch"] for entry in log["epochs"]]
        train = [entry["train_loss"] for entry in log["epochs"]]
        val = [entry["val_loss"] for entry in log["epochs"]]
        return epochs, train, val

    epochs_s, train_s, val_s = extract(single_log)
    epochs_c, train_c, val_c = extract(chunk_log)

    ax.plot(epochs_s, train_s, color=PALETTE["single"], linewidth=1.55, label="Single-step train")
    ax.plot(epochs_s, val_s, color=PALETTE["single"], linewidth=1.55, linestyle="--", alpha=0.9, label="Single-step val")
    ax.plot(epochs_c, train_c, color=PALETTE["chunk"], linewidth=1.55, label="Action-chunk train")
    ax.plot(epochs_c, val_c, color=PALETTE["chunk"], linewidth=1.55, linestyle="--", alpha=0.9, label="Action-chunk val")

    ax.set_title("Student Behavior Cloning Training Curves", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.28, color=PALETTE["grid"])
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=9.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
