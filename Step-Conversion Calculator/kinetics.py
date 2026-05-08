"""
Friedman isoconversional kinetics.

Pure-numeric module: file parsing, Ea(α)/logA(α) interpolation, and
forward-Euler integration of α(t) with a stability diagnostic.

No UI / Tkinter dependencies — safe to import from tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

R: float = 8.314  # J/(mol·K)

KineticsFn = Callable[[float], float]


class KineticsBundle(NamedTuple):
    """Interpolators + raw arrays for plotting / inspection."""
    ea_fn: KineticsFn       # α  →  Ea  (J/mol)
    la_fn: KineticsFn       # α  →  log10(A)
    alpha_ea: np.ndarray    # α grid for Ea
    ea: np.ndarray          # Ea values (J/mol)
    alpha_la: np.ndarray    # α grid for logA
    la: np.ndarray          # log10(A) values
    ea_unit_converted: bool # True if input was kJ/mol and we scaled to J/mol


@dataclass
class IntegrationDiagnostics:
    """Diagnostic info from `compute_alpha` for stability checks."""
    max_step: float           # max value of k·(1-α)·Δt encountered
    max_step_index: int       # index where max_step occurred
    max_dt: float             # largest Δt in the input (s)
    n_steps: int              # number of integration steps
    final_alpha: float

    @property
    def is_stable(self) -> bool:
        # Forward Euler on dα/dt = k·(1-α) is stable for k·Δt ≤ ~0.1
        # (kept conservative — paper recommends < 1e-2 for high accuracy).
        return self.max_step <= 0.1

    def warning_message(self) -> str | None:
        if self.is_stable:
            return None
        return (
            f"Forward-Euler step is large (max k·Δt·(1-α) = {self.max_step:.3f} "
            f"at sample {self.max_step_index}). "
            f"Largest Δt in data = {self.max_dt:.2f} s. "
            f"Consider denser sampling — the integration may overshoot."
        )


# ─── Parsing ─────────────────────────────────────────────────────────────────


def _parse_neo(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Kinetics-Neo TSV export. Reads columns 1 (α) and 2 (value)."""
    alphas: list[float] = []
    vals: list[float] = []
    in_data = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("---"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = s.split("\t")
            if len(parts) < 2:
                continue
            try:
                alphas.append(float(parts[0].replace(",", ".")))
                vals.append(float(parts[1].replace(",", ".")))
            except ValueError:
                continue
    if not alphas:
        raise ValueError(
            f"Kinetics Neo data not found in: {os.path.basename(path)}"
        )
    return np.array(alphas), np.array(vals)


def load_kinetics(ea_path: str, la_path: str) -> KineticsBundle:
    """
    Load Friedman Ea(α) and logA(α) from two Kinetics-Neo exports.

    Auto-detects kJ/mol vs J/mol via mean magnitude (typical Friedman Ea is
    50–200 kJ/mol = 5e4–2e5 J/mol; mean < 1000 ⇒ kJ/mol).
    """
    a_e, ea = _parse_neo(ea_path)
    a_l, la = _parse_neo(la_path)
    ea_converted = bool(np.mean(ea) < 1000.0)
    if ea_converted:
        ea = ea * 1000.0

    # Pad with edge values so np.interp doesn't extrapolate when α∈[0,1].
    a_e_p = np.r_[0.0, a_e, 1.0]
    ea_p = np.r_[ea[0], ea, ea[-1]]
    a_l_p = np.r_[0.0, a_l, 1.0]
    la_p = np.r_[la[0], la, la[-1]]

    def ea_fn(a: float) -> float:
        return float(np.interp(np.clip(float(a), 0.0, 1.0), a_e_p, ea_p))

    def la_fn(a: float) -> float:
        return float(np.interp(np.clip(float(a), 0.0, 1.0), a_l_p, la_p))

    return KineticsBundle(
        ea_fn=ea_fn,
        la_fn=la_fn,
        alpha_ea=a_e,
        ea=ea,
        alpha_la=a_l,
        la=la,
        ea_unit_converted=ea_converted,
    )


# ─── Integration ─────────────────────────────────────────────────────────────


def compute_alpha(
    t_s: np.ndarray,
    T_C: np.ndarray,
    ea_fn: KineticsFn,
    la_fn: KineticsFn,
    *,
    a0: float = 0.0,
) -> tuple[np.ndarray, IntegrationDiagnostics]:
    """
    Forward-Euler integration of dα/dt = A(α)·(1-α)·exp(-Ea(α)/(R·T)).

    Returns (α array, diagnostics).
    """
    n = len(t_s)
    alpha = np.zeros(n)
    alpha[0] = a0
    max_step = 0.0
    max_step_index = 0
    max_dt = 0.0

    for i in range(1, n):
        a = alpha[i - 1]
        dt = t_s[i] - t_s[i - 1]
        if dt > max_dt:
            max_dt = dt

        if a >= 1.0:
            alpha[i:] = 1.0
            break

        T_K = T_C[i - 1] + 273.15
        k = 10.0 ** la_fn(a) * np.exp(-ea_fn(a) / (R * T_K))
        step = k * (1.0 - a) * dt

        if step > max_step:
            max_step = step
            max_step_index = i

        new_a = a + step
        alpha[i] = 1.0 if new_a >= 1.0 else new_a

    diag = IntegrationDiagnostics(
        max_step=max_step,
        max_step_index=max_step_index,
        max_dt=max_dt,
        n_steps=n - 1,
        final_alpha=float(alpha[-1]) if n > 0 else 0.0,
    )
    return alpha, diag
