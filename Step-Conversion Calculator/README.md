# Conversion Calculator

Desktop app for real-time monitoring of binder **conversion α(t)** during
hot-pressing of wood-based panels. Reads thermocouple temperature data, applies
the **Friedman isoconversional kinetic model**, and integrates α numerically
for both panel surface and core in real time.

Originally developed alongside a study on adhesive curing kinetics (PhD work
at BOKU Vienna).

---

## What it does

- Loads kinetic parameters Ea(α) and log A(α) from **Kinetics Neo** exports.
- Loads temperature profiles T_surf(t), T_core(t) from **Excel** (.xlsx) or
  **Testo** loggers (.txt/.csv).
- Integrates `dα/dt = A(α)·(1−α)·exp(−Ea(α)/(R·T))` with forward Euler.
- Shows live α and T plots, a hot-press schematic with temperature gradient,
  and exports CSV / PNG / summary text.
- Three operating modes:
  - **Files** — one-shot computation on a recorded profile.
  - **Simulation** — playback with scrubbing and adjustable speed.
  - **Live** — polls the data file by mtime and reads only newly-appended
    rows for incremental updates during pressing.
- **Overlay** mode lets you save multiple runs and compare them on one chart.

---

## Requirements

- macOS (tested), Linux, or Windows
- Python **3.10+** (developed on 3.14)
- Python packages:
  - `numpy`
  - `matplotlib`
  - `openpyxl`
  - `tkinter` (ships with most Python distributions)

Install dependencies if needed:

```bash
pip install numpy matplotlib openpyxl
```

---

## Launching

**macOS:** double-click `Conversion Calculator.command` in Finder.

**From the terminal (any platform):**

```bash
cd "Step-Conversion Calculator"
python3 conversion_ui.py
```

The app remembers file paths and settings between runs in `ui_settings.json`.

---

## Project layout

```
Step-Conversion Calculator/
├── conversion_ui.py          ← entry point + main App class
├── kinetics.py               ← Friedman parsing + α integration + diagnostics
├── loaders.py                ← Excel & Testo loaders (with incremental reads)
├── viz_press.py              ← Hot-press schematic Canvas widget
├── exports.py                ← CSV / PNG / summary writers with metadata
├── tests/
│   └── test_kinetics.py      ← unit tests for the numerical core
├── Conversion Calculator.command  ← macOS launcher
└── ui_settings.json          ← persisted settings
```

The four supporting modules are pure-numeric / pure-rendering — they have no
Tkinter dependencies and can be reused from scripts or notebooks:

```python
from kinetics import load_kinetics, compute_alpha
from loaders import load_excel, trim_after_cooling

kin = load_kinetics("Ea.txt", "logA.txt")
profile = load_excel("data.xlsx", time_col=4, surf_col=6, core_col=7, data_row=3)
t, Ts, Tc, _ = trim_after_cooling(profile.t_s, profile.T_surf, profile.T_core, 45.0)
alpha_surf, diag = compute_alpha(t, Ts, kin.ea_fn, kin.la_fn)
print(f"α final = {alpha_surf[-1]:.4f}, stable: {diag.is_stable}")
```

---

## Input file formats

### Kinetics (Kinetics Neo TSV export)

Tab-separated text file with a header block ending in a `---` line, then rows
of `α  value  ±error`:

```
Project: my-binder.kinx2
Export: Friedman, ActivationEnergy

Conversion	Ea / (kJ/mol)	±Error
--------------------------------------
0.05	72.43	0.33
0.10	74.03	2.27
...
```

Two files are required: one for **Ea(α)** and one for **log A(α)**. Units
(kJ/mol vs J/mol) are auto-detected from the magnitude.

### Temperature data — Excel

Default mapping: time in column **D**, surface in column **F**, core in
column **G**, data starting at row **3**. Adjust via the *Settings* tab.
Time cells must be either Excel time (fraction of day) or Python timedelta.

### Temperature data — Testo

TSV or CSV with time in the first column (seconds since start) and
configurable channel columns for surface/core. Tab vs semicolon is
auto-detected.

---

## Quick start

1. **Files tab** → select Ea and log A files (Kinetics Neo exports).
2. Click **🔍 Preview kinetics** to verify Ea(α) and log A(α) curves look
   reasonable (no spurious jumps at the boundaries).
3. Select your temperature data file (Excel or Testo).
4. Adjust **T_cutoff** if needed (default 45 °C — integration stops once
   *both* surface and core have cooled below this).
5. Click **▶ Load & Compute**.
6. Watch the chart animate, the press schematic update, and α-thresholds
   (default 0.80 and 1.00) get annotated.
7. Export results: **📷 PNG**, **📊 CSV**, or **📄 Summary** — all include
   provenance metadata (paths, cutoff, integration warnings, etc.).

For overlay comparison, save the current run via **💾 Save current as run…**
in the Files tab, then load and compute a different profile. The saved runs
appear in the **📚 Runs** tab and can be toggled on/off.

For continuous monitoring, point the *Live* tab at the file your
data-acquisition software is appending to. The poll interval defaults to
**2 seconds** and only newly-added rows are parsed.

---

## Numerical notes

- Integration scheme: **forward Euler** with α-clamping to [0, 1].
- Stability heuristic: a warning is shown if any single step
  `k·(1−α)·Δt > 0.1` (rare with 1–5 s sampling, common with coarser logs).
- Cutoff: applies after the temperature peak — tail samples where *both*
  T_surf and T_core have fallen below the cutoff are excluded from the
  integration (negligible reaction rate at low T).
- The kinetics array is padded at α = 0 and α = 1 with the boundary values
  to avoid extrapolation artifacts when α drifts slightly outside the
  Friedman fit range.

---

## Running tests

```bash
cd "Step-Conversion Calculator"
python3 -m unittest tests.test_kinetics -v
```

Eight tests cover: analytic vs forward-Euler agreement, α clamping,
unstable-step detection, parser format, kJ→J conversion, and trim logic.

---

## Author

**Ondřej Fiedler**
Email: [Fiedler.ond@gmail.com](mailto:Fiedler.ond@gmail.com)

If you use this tool in a publication, please cite the accompanying paper
(Section 2.X "Step-Conversion Calculator").
