import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input.csv> <output.png>")
    raise SystemExit(1)


input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

iterations = []
ineffective = []

with input_path.open(newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        iterations.append(int(row["iteration"]))
        ineffective.append(float(row["ineffective_percent"]))


if not iterations:
    raise SystemExit("ERROR: No iteration data found.")


window = 10
moving_average = []

for index in range(len(ineffective)):
    start = max(0, index - window + 1)
    values = ineffective[start:index + 1]
    moving_average.append(sum(values) / len(values))


overall_average = sum(ineffective) / len(ineffective)

plt.figure(figsize=(11, 6))

plt.plot(
    iterations,
    ineffective,
    color="#12B5CB",
    linewidth=1.2,
    alpha=0.45,
    label="Individual iteration",
)

plt.plot(
    iterations,
    moving_average,
    color="#864DEB",
    linewidth=3,
    label="10-iteration moving average",
)

plt.axhline(
    overall_average,
    color="#222222",
    linestyle="--",
    linewidth=1.5,
    label=f"Overall average: {overall_average:.2f}%",
)

plt.title(
    "Ineffective FPS Distance Updates Across Iterations",
    fontsize=18,
    fontweight="bold",
    color="#864DEB",
    pad=16,
)

plt.xlabel("FPS iteration", fontsize=13)
plt.ylabel("Ineffective distance updates (%)", fontsize=13)

plt.xlim(1, max(iterations))
plt.ylim(0, 100)

plt.grid(axis="y", linestyle="--", alpha=0.25)
plt.legend(loc="lower right", frameon=False, fontsize=10)

plt.text(
    0.01,
    1.01,
    "KITTI frame 0000000000 | N = 100,000 | M = 128",
    transform=plt.gca().transAxes,
    fontsize=11,
    color="#333333",
)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Input: {input_path}")
print(f"Iterations: {len(iterations)}")
print(f"Overall ineffective percentage: {overall_average:.6f}%")
print(f"First iteration: {ineffective[0]:.6f}%")
print(f"Last iteration: {ineffective[-1]:.6f}%")
print(f"Maximum iteration: {max(ineffective):.6f}%")
print(f"Figure: {output_path}")
