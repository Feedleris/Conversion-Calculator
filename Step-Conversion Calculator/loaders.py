"""
Temperature-profile loaders: Excel (openpyxl) and Testo TSV/CSV.

Both loaders support `start_row` / `start_line` so the live monitor can read
only newly-appended rows after the file has grown, instead of re-parsing the
whole file from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TempProfile:
    """A loaded temperature profile, normalised to seconds-since-start."""
    t_s: np.ndarray         # seconds since first sample
    T_surf: np.ndarray      # °C
    T_core: np.ndarray      # °C
    last_row: int           # 1-based last row (Excel) or line index (Testo)
    t0_raw: float | None    # raw time of first sample (for resumed reads)


def _to_seconds(tv: object) -> float:
    """Excel time cell → seconds. Accepts timedelta or fraction-of-day floats."""
    if hasattr(tv, "total_seconds"):
        return float(tv.total_seconds())  # type: ignore[union-attr]
    return float(tv) * 86400.0  # type: ignore[arg-type]


def load_excel(
    path: str,
    time_col: int,
    surf_col: int,
    core_col: int,
    data_row: int,
    *,
    start_row: int | None = None,
    t0_override: float | None = None,
) -> TempProfile:
    """
    Load a temperature profile from an Excel file.

    Args:
        path: Workbook path.
        time_col / surf_col / core_col: 1-based column indices.
        data_row: 1-based row where data begins.
        start_row: If set, start reading from this row (inclusive).
            Use this for incremental reads when the file is being appended.
        t0_override: If set, subtract this value from raw time to keep
            t=0 anchored to the original first sample (resume mode).
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        first_row = start_row if start_row is not None else data_row

        t0: float | None = t0_override
        ts: list[float] = []
        su: list[float] = []
        co: list[float] = []
        last_row = first_row - 1

        for row_idx, row in enumerate(
            ws.iter_rows(min_row=first_row, values_only=True),
            start=first_row,
        ):
            try:
                tv = row[time_col - 1]
                sv = row[surf_col - 1]
                cv = row[core_col - 1]
            except IndexError:
                continue
            if tv is None or sv is None or cv is None:
                continue
            try:
                t_s = _to_seconds(tv)
                if t0 is None:
                    t0 = t_s
                ts.append(t_s - t0)
                su.append(float(sv))
                co.append(float(cv))
                last_row = row_idx
            except (TypeError, ValueError):
                continue
    finally:
        wb.close()

    if not ts:
        if start_row is not None:
            # No new rows since last read — return empty profile.
            return TempProfile(
                t_s=np.empty(0),
                T_surf=np.empty(0),
                T_core=np.empty(0),
                last_row=last_row,
                t0_raw=t0,
            )
        raise ValueError("No data loaded from Excel — check column/row numbers.")

    return TempProfile(
        t_s=np.array(ts),
        T_surf=np.array(su),
        T_core=np.array(co),
        last_row=last_row,
        t0_raw=t0,
    )


def load_testo(
    path: str,
    surf_ch: int,
    core_ch: int,
    *,
    start_line: int = 0,
    t0_override: float | None = None,
) -> TempProfile:
    """
    Load a Testo TSV/CSV file. Auto-detects tab vs semicolon separator.

    Args:
        surf_ch / core_ch: 0-based channel indices into the split row.
        start_line: 0-based line index to start from (incremental reads).
        t0_override: If set, anchor t=0 to this raw time value.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    sep = "\t" if content.count("\t") > content.count(";") else ";"
    t0: float | None = t0_override
    ts: list[float] = []
    su: list[float] = []
    co: list[float] = []
    lines = content.splitlines()
    last_line = start_line - 1

    for line_idx, line in enumerate(lines):
        if line_idx < start_line:
            continue
        parts = line.split(sep)
        if len(parts) <= max(surf_ch, core_ch):
            continue
        try:
            t_s = float(parts[0])
            sv = float(parts[surf_ch].replace(",", "."))
            cv = float(parts[core_ch].replace(",", "."))
        except (ValueError, IndexError):
            continue
        if t0 is None:
            t0 = t_s
        ts.append(t_s - t0)
        su.append(sv)
        co.append(cv)
        last_line = line_idx

    if not ts:
        if start_line > 0:
            return TempProfile(
                t_s=np.empty(0),
                T_surf=np.empty(0),
                T_core=np.empty(0),
                last_row=last_line,
                t0_raw=t0,
            )
        raise ValueError("No data loaded from Testo file.")

    return TempProfile(
        t_s=np.array(ts),
        T_surf=np.array(su),
        T_core=np.array(co),
        last_row=last_line,
        t0_raw=t0,
    )


def trim_after_cooling(
    t_s: np.ndarray,
    T_surf: np.ndarray,
    T_core: np.ndarray,
    cutoff_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Drop tail samples where BOTH T_surf and T_core have fallen below `cutoff_c`
    after the maximum temperature was reached.

    Returns (t_s, T_surf, T_core, n_dropped).
    """
    if len(t_s) == 0:
        return t_s, T_surf, T_core, 0
    T_max = np.maximum(T_surf, T_core)
    peak = int(np.argmax(T_max))
    cut = len(t_s)
    for i in range(peak, len(T_max)):
        if T_surf[i] < cutoff_c and T_core[i] < cutoff_c:
            cut = i
            break
    return t_s[:cut], T_surf[:cut], T_core[:cut], len(t_s) - cut
