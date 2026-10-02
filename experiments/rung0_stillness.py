"""Rung 0: the field returns to stillness after a disturbance, no learning.

Run from the project root:  python3 experiments/rung0_stillness.py
Pass = energy falls below 1% of its peak, and stillness stays stable.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stillness import Field

field = Field(n=256)
rng = np.random.default_rng(1)
pulse = rng.standard_normal(field.n) * 5.0

PULSE_TICKS, TOTAL_TICKS = 50, 1500
energy = []
for t in range(TOTAL_TICKS):
    u = pulse if t < PULSE_TICKS else None
    energy.append(field.step(u=u, M=0.0))

energy = np.array(energy)
peak = energy.max()
below = np.where(energy[PULSE_TICKS:] < 0.01 * peak)[0]
settle = (below[0] * field.dt) if below.size else None

print(f"peak energy      {peak:.2f}")
print(f"final energy     {energy[-1]:.6f}")
print(f"settle time      {settle:.2f} s" if settle is not None else "settle time      never")
print(f"stillness stable {field.stable()}")
passed = settle is not None and field.stable()
print("RUNG 0", "PASS" if passed else "FAIL")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(__file__).with_name("rung0_energy.png")
    t = np.arange(TOTAL_TICKS) * field.dt
    plt.figure(figsize=(7, 3))
    plt.plot(t, energy)
    plt.axvspan(0, PULSE_TICKS * field.dt, alpha=0.15, label="disturbance")
    plt.xlabel("time (s)")
    plt.ylabel("energy (distance from stillness)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"plot             {out}")
except ImportError:
    pass
