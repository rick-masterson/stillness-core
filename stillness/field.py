"""The field: a continuous state that always falls back toward stillness.

Rung 0 of the test ladder. No learning happens unless the modulator M is
non-zero, so with M = 0 this is pure physics: disturb it, watch it settle.

    dx/dt = -leak * x + W @ x + u(t)       (field dynamics)
    dW    = eta * M * outer(x, x)          (gated local plasticity)

Stillness = the field's energy, 0.5 * |x|^2, near zero.
"""

import numpy as np


class Field:
    def __init__(self, n=256, leak=1.0, coupling=0.3, dt=0.01, eta=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.leak, self.dt, self.eta = n, leak, dt, eta
        self.x = np.zeros(n)
        # Symmetric random coupling, scaled so the leak always wins:
        # spectral radius of W = coupling < leak, so stillness is stable.
        w = rng.standard_normal((n, n))
        w = (w + w.T) / 2
        np.fill_diagonal(w, 0.0)
        self.W = w * (coupling / np.max(np.abs(np.linalg.eigvalsh(w))))

    def energy(self):
        """Distance from stillness."""
        return 0.5 * float(self.x @ self.x)

    def step(self, u=None, M=0.0):
        """Advance one tick. u = input disturbance, M = modulator (0 = no learning)."""
        drive = -self.leak * self.x + self.W @ self.x
        if u is not None:
            drive = drive + u
        self.x = self.x + self.dt * drive
        if M != 0.0:
            self.W += self.eta * M * np.outer(self.x, self.x)
            np.fill_diagonal(self.W, 0.0)
        return self.energy()

    def stable(self):
        """True while every mode decays, i.e. stillness is still the floor."""
        return np.max(np.linalg.eigvalsh(self.W)) < self.leak
