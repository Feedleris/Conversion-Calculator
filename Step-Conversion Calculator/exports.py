"""
Export helpers — CSV, PNG, summary text — all with reproducibility metadata.

Metadata is written as `#`-prefixed lines at the top of CSV/Summary outputs
so that recenzent/co-author can see exactly which kinetics files, cutoff,
threshold values and app version produced the result.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

APP_VERSION = "v12"


@dataclass
class ExportMetadata:
    """Provenance of a computation. Embedded in CSV / Summary."""
    ea_path: str = ""
    la_path: str = ""
    data_path: str = ""
    file_type: str = ""
    cutoff_used: bool = False
    cutoff_c: float | None = None
    n_trimmed: int = 0
    ea_unit_converted: bool = False
    integration_max_step: float | None = None
    integration_warning: str | None = None
    thresholds: list[float] = field(default_factory=lambda: [0.8, 1.0])

    def header_lines(self) -> list[str]:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"# Friedman Conversion Calculator {APP_VERSION}",
            f"# Generated: {ts}",
            f"# Ea file: {self.ea_path or '(unset)'}",
            f"# logA file: {self.la_path or '(unset)'}",
            f"# Data file: {self.data_path or '(unset)'}",
            f"# Data type: {self.file_type or '(unset)'}",
            f"# T_cutoff applied: {self.cutoff_used} ({self.cutoff_c} °C)"
            f"  — trimmed {self.n_trimmed} samples",
            f"# Ea unit auto-converted (kJ→J): {self.ea_unit_converted}",
            f"# Thresholds: {', '.join(f'{t:.2f}' for t in self.thresholds)}",
        ]
        if self.integration_max_step is not None:
            lines.append(
                f"# Integration max k·Δt·(1-α): {self.integration_max_step:.4f}"
            )
        if self.integration_warning:
            lines.append(f"# WARNING: {self.integration_warning}")
        return lines


def export_csv(
    path: str,
    t_min: np.ndarray,
    T_surf: np.ndarray,
    T_core: np.ndarray,
    a_surf: np.ndarray,
    a_core: np.ndarray,
    metadata: ExportMetadata | None = None,
) -> None:
    """Write CSV with metadata header (lines prefixed with `#`)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if metadata is not None:
            for line in metadata.header_lines():
                fh.write(line + "\n")
            fh.write("#\n")
        w = csv.writer(fh)
        w.writerow(["time_min", "T_surface_C", "T_core_C", "alpha_surface", "alpha_core"])
        for i in range(len(t_min)):
            w.writerow([
                f"{t_min[i]:.4f}",
                f"{T_surf[i]:.3f}",
                f"{T_core[i]:.3f}",
                f"{a_surf[i]:.6f}",
                f"{a_core[i]:.6f}",
            ])


def export_summary(
    path: str,
    t_min: np.ndarray,
    T_surf: np.ndarray,
    T_core: np.ndarray,
    a_surf: np.ndarray,
    a_core: np.ndarray,
    metadata: ExportMetadata | None = None,
) -> None:
    """Write human-readable summary text with metadata block at the top."""
    sep = "─" * 50
    lines: list[str] = [
        f"Friedman Conversion Calculator — Summary ({APP_VERSION})",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
    ]
    if metadata is not None:
        lines.extend(metadata.header_lines())
        lines.append(sep)

    lines.extend([
        f"α surface (final): {a_surf[-1]:.4f}",
        f"α core    (final): {a_core[-1]:.4f}",
        f"T max core:        {T_core.max():.1f} °C",
        f"T max surface:     {T_surf.max():.1f} °C",
        f"Total time:        {t_min[-1]:.1f} min",
        sep,
        "Threshold crossing times:",
    ])
    thresholds = metadata.thresholds if metadata else [0.8, 1.0]
    for label, arr in [("Surface", a_surf), ("Core", a_core)]:
        for th in thresholds:
            idx = np.where(arr >= th)[0]
            if len(idx) > 0:
                lines.append(f"  α {label} ≥ {th:.2f}:  t = {t_min[idx[0]]:.1f} min")
            else:
                lines.append(f"  α {label} ≥ {th:.2f}:  not reached")

    lines.extend([sep, "α core over time:"])
    n = len(t_min)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        idx = min(int(n * frac) - 1, n - 1)
        lines.append(
            f"  t={t_min[idx]:6.1f} min → α core={a_core[idx]:.4f}  "
            f"α surf={a_surf[idx]:.4f}"
        )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def default_filename(prefix: str, extension: str) -> str:
    """Return e.g. `conversion_20260508_143205.png`."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{extension.lstrip('.')}"
