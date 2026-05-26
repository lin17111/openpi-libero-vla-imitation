import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# Figure 1: success rate comparison
names = ["π0.5 Teacher", "Single-step\nStudent", "Action-chunk\nStudent"]
rates = [98.4, 0.0, 70.0]

plt.figure(figsize=(7, 4.5))
bars = plt.bar(names, rates)
plt.ylabel("Closed-loop success rate (%)")
plt.ylim(0, 105)
plt.title("LIBERO closed-loop rollout performance")

for bar, rate in zip(bars, rates):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        rate + 2,
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig("figures/success_rate_comparison.png", dpi=300)
plt.savefig("figures/success_rate_comparison.pdf")
print("Saved figures/success_rate_comparison.png")

# Figure 2: pipeline diagram
plt.figure(figsize=(11, 3.2))
ax = plt.gca()
ax.axis("off")

steps = [
    "π0.5 Teacher\nrollout",
    "Trajectory\ncollection",
    "Single-step\nstudent BC",
    "Action-chunk\nstudent BC",
    "Closed-loop\nstudent eval",
]

x_positions = [0.08, 0.30, 0.52, 0.74, 0.92]
y = 0.55

for i, (x, text) in enumerate(zip(x_positions, steps)):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black"),
        transform=ax.transAxes,
    )
    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(x_positions[i + 1] - 0.07, y),
            xytext=(x + 0.07, y),
            arrowprops=dict(arrowstyle="->", lw=1.5),
            xycoords=ax.transAxes,
        )

ax.text(
    0.5,
    0.15,
    "OpenPI + LIBERO/MuJoCo VLA imitation learning pipeline",
    ha="center",
    va="center",
    fontsize=11,
    transform=ax.transAxes,
)

plt.tight_layout()
plt.savefig("figures/pipeline_overview.png", dpi=300)
plt.savefig("figures/pipeline_overview.pdf")
print("Saved figures/pipeline_overview.png")
