#!/usr/bin/env python3
"""
Friedman Conversion Calculator — GUI v12

Tk GUI for real-time monitoring of binder conversion α(t) computed from
panel surface/core temperatures via the Friedman isoconversional model.

This file is the application shell. Pure-numeric and rendering code lives in
sibling modules:

    kinetics.py  — Friedman parsing + Forward-Euler α integration
    loaders.py   — Excel and Testo data loaders (with incremental reads)
    viz_press.py — Hot-press schematic Canvas widget
    exports.py   — CSV / PNG / summary writers, metadata block
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import numpy as np

from kinetics import KineticsBundle, compute_alpha, load_kinetics
from loaders import TempProfile, load_excel, load_testo, trim_after_cooling
from viz_press import COL_CORE, COL_SURF, PressViz
from exports import (
    APP_VERSION,
    ExportMetadata,
    default_filename,
    export_csv,
    export_summary,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(SCRIPT_DIR, "ui_settings.json")
COL_CUT = "#95a5a6"
COL_CUR = "#f39c12"

# Palette for overlay runs (cycles through these for additional saved runs).
OVERLAY_PALETTE = [
    ("#9b59b6", "#16a085"),
    ("#d35400", "#2980b9"),
    ("#27ae60", "#c0392b"),
    ("#8e44ad", "#f39c12"),
    ("#7f8c8d", "#2c3e50"),
]


# ─── Result dataclasses ──────────────────────────────────────────────────────


@dataclass
class ComputationResult:
    """A complete α(t) computation, with provenance."""
    t_min: np.ndarray
    T_surf: np.ndarray
    T_core: np.ndarray
    a_surf: np.ndarray
    a_core: np.ndarray
    metadata: ExportMetadata


@dataclass
class SavedRun:
    """A saved computation kept for overlay comparison."""
    name: str
    result: ComputationResult
    visible: bool = True
    color_pair: tuple[str, str] = ("#888", "#555")


@dataclass
class LiveCache:
    """Cumulative state for incremental file polling in live mode."""
    t_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    T_surf: np.ndarray = field(default_factory=lambda: np.empty(0))
    T_core: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_row: int = 0
    t0_raw: float | None = None

    def reset(self) -> None:
        self.t_s = np.empty(0)
        self.T_surf = np.empty(0)
        self.T_core = np.empty(0)
        self.last_row = 0
        self.t0_raw = None


# ─── App ─────────────────────────────────────────────────────────────────────


class App:
    """Main GUI controller."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Conversion Calculator")
        root.geometry("1540x800")
        root.minsize(1200, 660)

        # ── File / kinetics inputs ───────────────────────────────────────────
        self.v_ea = tk.StringVar()
        self.v_la = tk.StringVar()
        self.v_data = tk.StringVar()
        self.v_ftype = tk.StringVar(value="Excel")
        self.v_cutoff = tk.DoubleVar(value=45.0)
        self.v_use_cutoff = tk.BooleanVar(value=True)
        self.v_surf_ch = tk.IntVar(value=1)
        self.v_core_ch = tk.IntVar(value=2)
        self.v_tc = tk.IntVar(value=4)
        self.v_sc = tk.IntVar(value=6)
        self.v_cc = tk.IntVar(value=7)
        self.v_dr = tk.IntVar(value=3)
        self.v_animate = tk.BooleanVar(value=True)
        self.v_aspeed = tk.IntVar(value=80)
        self.v_live = tk.BooleanVar(value=False)
        # Default polling interval matches paper description (2 s).
        self.v_linterv = tk.IntVar(value=2)
        self.v_press_temp = tk.DoubleVar(value=150.0)
        self.v_thresholds = tk.StringVar(value="0.80, 1.00")

        # ── Sim tab ──────────────────────────────────────────────────────────
        self.v_sim_ea = tk.StringVar()
        self.v_sim_la = tk.StringVar()
        self.v_sim_data = tk.StringVar()
        self.v_sim_speed = tk.DoubleVar(value=10.0)
        self.v_sim_status = tk.StringVar(value="Select files and press Prepare")
        self.v_sim_time = tk.StringVar(value="0.0 / — min")
        self.v_sim_pos_frac = tk.DoubleVar(value=0.0)

        # ── Kinetics state (loaded once, shared across modes) ────────────────
        self._kinetics: KineticsBundle | None = None

        # ── Main computation result ──────────────────────────────────────────
        self._result: ComputationResult | None = None
        self._computing = False

        # ── Overlay (saved runs) ─────────────────────────────────────────────
        self._runs: list[SavedRun] = []

        # ── Live monitor cache ───────────────────────────────────────────────
        self._live_job: str | None = None
        self._last_mt: float | None = None
        self._live_cache = LiveCache()

        # ── Animation state machine ──────────────────────────────────────────
        self._anim_job: str | None = None
        self._anim_state: dict = {}

        # ── Simulation state ─────────────────────────────────────────────────
        self._sim_loaded = False
        self._sim_running = False
        self._sim_pos = 0
        self._sim_n = 0
        self._sim_t: np.ndarray | None = None
        self._sim_T_surf: np.ndarray | None = None
        self._sim_T_core: np.ndarray | None = None
        self._sim_a_surf: np.ndarray | None = None
        self._sim_a_core: np.ndarray | None = None
        self._sim_job: str | None = None
        self._sim_lines: dict | None = None
        self._sim_time_leg = None
        self._sim_val_leg_a = None
        self._sim_val_leg_T = None

        # ── Threshold annotations ────────────────────────────────────────────
        self._thresh_anns: dict[str, dict] = {}
        self._thresh_placed_y: list[float] = []

        # ── Live readout (results bar) ───────────────────────────────────────
        self.v_live_info = tk.StringVar(value="")

        self._build()
        self._load_settings()
        self._set_status("Ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))
        left = ttk.Frame(pane, width=320)
        left.pack_propagate(False)
        right = ttk.Frame(pane)
        viz = ttk.Frame(pane, width=380)
        viz.pack_propagate(False)
        pane.add(left, weight=0)
        pane.add(right, weight=1)
        pane.add(viz, weight=1)

        # Create PressViz first — _build_left's Settings tab references it.
        self.press = PressViz(viz, self.v_press_temp)
        self._build_left(left)
        self._build_right(right)

        # Bottom strip: status (left, light) + credit (right, white & bigger).
        bottom = tk.Frame(self.root, bg="#2c3e50")
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_status = tk.StringVar()
        tk.Label(
            bottom,
            textvariable=self.v_status,
            bg="#2c3e50",
            fg="#cccccc",
            anchor=tk.W,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            bottom,
            text="Created by Ondřej Fiedler  ·  Fiedler.ond@gmail.com",
            bg="#2c3e50",
            fg="#ffffff",
            font=("TkDefaultFont", 11, "bold"),
            padx=12,
            pady=4,
        ).pack(side=tk.RIGHT)

    def _build_left(self, p: ttk.Frame) -> None:
        nb = ttk.Notebook(p)
        nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        t1 = ttk.Frame(nb)
        t2 = ttk.Frame(nb)
        t3 = ttk.Frame(nb)
        t4 = ttk.Frame(nb)
        t5 = ttk.Frame(nb)
        nb.add(t1, text="📁  Files")
        nb.add(t2, text="⚙  Settings")
        nb.add(t3, text="📚  Runs")
        nb.add(t4, text="🎬  Simulation")
        nb.add(t5, text="📡  Live")
        self._tab_files(t1)
        self._tab_settings(t2)
        self._tab_runs(t3)
        self._tab_simulation(t4)
        self._tab_live(t5)

    # ── Files tab ────────────────────────────────────────────────────────────

    def _tab_files(self, p: ttk.Frame) -> None:
        g = ttk.LabelFrame(p, text=" Kinetics (Kinetics Neo export) ", padding=6)
        g.pack(fill=tk.X, padx=6, pady=(8, 4))
        self._frow(g, "Ea file:", self.v_ea, [("Text", "*.txt"), ("All", "*.*")])
        self._frow(g, "logA file:", self.v_la, [("Text", "*.txt"), ("All", "*.*")])
        ttk.Button(g, text="🔍  Preview kinetics",
                   command=self._preview_kinetics).pack(fill=tk.X, pady=(4, 0))

        g2 = ttk.LabelFrame(p, text=" Temperature data ", padding=6)
        g2.pack(fill=tk.X, padx=6, pady=4)
        ft = ttk.Frame(g2)
        ft.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(ft, text="File type:").pack(side=tk.LEFT)
        ttk.Radiobutton(ft, text="Excel", variable=self.v_ftype, value="Excel").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(ft, text="Testo", variable=self.v_ftype, value="Testo").pack(side=tk.LEFT)
        self._frow(
            g2, "File:", self.v_data,
            [("Excel", "*.xlsx *.xls"), ("Text", "*.txt *.csv"), ("All", "*.*")],
        )

        cf = ttk.Frame(g2)
        cf.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(cf, text="T_cutoff (°C):", variable=self.v_use_cutoff).pack(side=tk.LEFT)
        self.entry_cutoff = ttk.Entry(cf, textvariable=self.v_cutoff, width=8)
        self.entry_cutoff.pack(side=tk.LEFT, padx=4)
        ttk.Label(cf, text="← stop on cooling", foreground="gray").pack(side=tk.LEFT)
        self.v_use_cutoff.trace_add(
            "write",
            lambda *_: self.entry_cutoff.configure(
                state=tk.NORMAL if self.v_use_cutoff.get() else tk.DISABLED
            ),
        )

        bf = ttk.Frame(p)
        bf.pack(fill=tk.X, padx=6, pady=10)
        self.btn_run = ttk.Button(bf, text="▶  Load & Compute", command=self._start)
        self.btn_run.pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="💾  Save current as run…",
                   command=self._save_current_as_run).pack(fill=tk.X, pady=2)
        ttk.Button(bf, text="✕  Clear", command=self._clear).pack(fill=tk.X, pady=2)
        self.progress = ttk.Progressbar(p, mode="determinate")
        self.progress.pack(fill=tk.X, padx=6, pady=2)

    # ── Settings tab ─────────────────────────────────────────────────────────

    def _tab_settings(self, p: ttk.Frame) -> None:
        g = ttk.LabelFrame(p, text=" Excel — column & row mapping ", padding=6)
        g.pack(fill=tk.X, padx=6, pady=(8, 4))
        for lbl, var in [
            ("Time (col):", self.v_tc),
            ("Surface (col):", self.v_sc),
            ("Core (col):", self.v_cc),
            ("Data from row:", self.v_dr),
        ]:
            f = ttk.Frame(g)
            f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=lbl, width=16).pack(side=tk.LEFT)
            ttk.Spinbox(f, textvariable=var, from_=1, to=50, width=5).pack(side=tk.LEFT)

        g2 = ttk.LabelFrame(p, text=" Testo channels ", padding=6)
        g2.pack(fill=tk.X, padx=6, pady=4)
        for lbl, var in [
            ("Surface ch.:", self.v_surf_ch),
            ("Core ch.:", self.v_core_ch),
        ]:
            f = ttk.Frame(g2)
            f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=lbl, width=16).pack(side=tk.LEFT)
            ttk.Spinbox(f, textvariable=var, from_=1, to=10, width=5).pack(side=tk.LEFT)

        g3 = ttk.LabelFrame(p, text=" Computation animation ", padding=6)
        g3.pack(fill=tk.X, padx=6, pady=4)
        ttk.Checkbutton(g3, text="Animate computation", variable=self.v_animate).pack(anchor=tk.W)
        af = ttk.Frame(g3)
        af.pack(fill=tk.X, pady=2)
        ttk.Label(af, text="Total frames:").pack(side=tk.LEFT)
        ttk.Spinbox(af, textvariable=self.v_aspeed, from_=10, to=500, width=6).pack(side=tk.LEFT, padx=4)

        g4 = ttk.LabelFrame(p, text=" Hot-press visualization ", padding=6)
        g4.pack(fill=tk.X, padx=6, pady=4)
        pf = ttk.Frame(g4)
        pf.pack(fill=tk.X, pady=1)
        ttk.Label(pf, text="Press temp (°C):", width=16).pack(side=tk.LEFT)
        ttk.Spinbox(
            pf, textvariable=self.v_press_temp, from_=50, to=300, width=6, increment=10,
            command=self.press.refresh,
        ).pack(side=tk.LEFT)

        g5 = ttk.LabelFrame(p, text=" α thresholds (annotations + summary) ", padding=6)
        g5.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(g5, text="Comma-separated, range 0–1:").pack(anchor=tk.W)
        ttk.Entry(g5, textvariable=self.v_thresholds).pack(fill=tk.X, pady=2)
        ttk.Label(
            g5, text="Default: 0.80, 1.00 — used to mark crossings on chart and in summary.",
            foreground="gray", wraplength=260, font=("TkDefaultFont", 8),
        ).pack(anchor=tk.W)

        ttk.Button(p, text="💾  Save settings",
                   command=self._save_settings).pack(fill=tk.X, padx=6, pady=10)

    # ── Runs tab (overlay management) ────────────────────────────────────────

    def _tab_runs(self, p: ttk.Frame) -> None:
        ttk.Label(
            p,
            text="Save the current computation (📁 Files tab) to overlay it for comparison.",
            wraplength=290, foreground="gray",
        ).pack(anchor=tk.W, padx=8, pady=(8, 4))

        bf = ttk.Frame(p)
        bf.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(bf, text="💾 Save current",
                   command=self._save_current_as_run).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="🗑 Clear all",
                   command=self._clear_all_runs).pack(side=tk.LEFT, padx=2)

        list_frame = ttk.LabelFrame(p, text=" Saved runs ", padding=4)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        canvas = tk.Canvas(list_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self._runs_inner = ttk.Frame(canvas)
        self._runs_inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._runs_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._refresh_runs_list()

    def _tab_simulation(self, p: ttk.Frame) -> None:
        gk = ttk.LabelFrame(p, text=" Kinetics (Kinetics Neo export) ", padding=6)
        gk.pack(fill=tk.X, padx=6, pady=(8, 4))
        self._frow(gk, "Ea file:", self.v_sim_ea, [("Text", "*.txt"), ("All", "*.*")])
        self._frow(gk, "logA file:", self.v_sim_la, [("Text", "*.txt"), ("All", "*.*")])
        ttk.Label(
            gk, text="Leave empty to use kinetics from Files tab.",
            foreground="gray", wraplength=260, font=("TkDefaultFont", 8),
        ).pack(anchor=tk.W, pady=(2, 0))

        g = ttk.LabelFrame(p, text=" Temperature data (Excel) ", padding=6)
        g.pack(fill=tk.X, padx=6, pady=4)
        self._frow(g, "Excel:", self.v_sim_data, [("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        ttk.Button(g, text="📂  Prepare simulation",
                   command=self._sim_load).pack(fill=tk.X, pady=(6, 2))
        ttk.Label(g, textvariable=self.v_sim_status, foreground="gray",
                  wraplength=260).pack(anchor=tk.W, pady=2)

        g2 = ttk.LabelFrame(p, text=" Playback ", padding=6)
        g2.pack(fill=tk.X, padx=6, pady=4)
        ctrl = ttk.Frame(g2)
        ctrl.pack(pady=4)
        self.btn_s_rew = ttk.Button(ctrl, text="⏮", command=self._sim_rewind, width=3)
        self.btn_s_play = ttk.Button(ctrl, text="▶", command=self._sim_toggle, width=3)
        for b in [self.btn_s_rew, self.btn_s_play]:
            b.pack(side=tk.LEFT, padx=3)
            b.configure(state=tk.DISABLED)
        sf = ttk.Frame(g2)
        sf.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(sf, text="Speed:").pack(side=tk.LEFT)
        self.lbl_spd = ttk.Label(sf, text="10×", width=5, anchor=tk.E)
        self.lbl_spd.pack(side=tk.RIGHT)
        ttk.Scale(
            sf, variable=self.v_sim_speed, from_=1, to=200, orient=tk.HORIZONTAL,
            command=lambda v: self.lbl_spd.configure(text=f"{float(v):.0f}×"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        sb = ttk.Frame(g2)
        sb.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(sb, text="Scrub:").pack(side=tk.LEFT)
        self.scrub = ttk.Scale(
            sb, variable=self.v_sim_pos_frac, from_=0, to=1,
            orient=tk.HORIZONTAL, command=self._sim_scrub,
        )
        self.scrub.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(g2, textvariable=self.v_sim_time, foreground="gray").pack(anchor=tk.W)

    def _tab_live(self, p: ttk.Frame) -> None:
        g = ttk.LabelFrame(p, text=" File monitor ", padding=6)
        g.pack(fill=tk.X, padx=6, pady=(8, 4))
        ttk.Checkbutton(
            g, text="Enable live monitor",
            variable=self.v_live, command=self._live_toggle,
        ).pack(anchor=tk.W, pady=2)
        lf = ttk.Frame(g)
        lf.pack(fill=tk.X, pady=2)
        ttk.Label(lf, text="Interval (s):").pack(side=tk.LEFT)
        ttk.Spinbox(lf, textvariable=self.v_linterv, from_=1, to=60, width=5).pack(side=tk.LEFT, padx=4)
        self.lbl_live = ttk.Label(g, text="● Inactive", foreground="gray")
        self.lbl_live.pack(anchor=tk.W, pady=4)
        ttk.Label(
            g, text="Polls the data file by mtime, reads only newly-appended rows, and re-runs α.",
            foreground="gray", wraplength=260, font=("TkDefaultFont", 8),
        ).pack(anchor=tk.W)

    # ── Right panel ──────────────────────────────────────────────────────────

    def _build_right(self, p: ttk.Frame) -> None:
        fig_f = ttk.Frame(p)
        fig_f.pack(fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        gs = self.fig.add_gridspec(2, 1, hspace=0.06, top=0.95, bottom=0.08, left=0.09, right=0.97)
        self.ax_T = self.fig.add_subplot(gs[0])
        self.ax_a = self.fig.add_subplot(gs[1], sharex=self.ax_T)
        self._init_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_f)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_f = ttk.Frame(p)
        tb_f.pack(fill=tk.X)
        self._tb = NavigationToolbar2Tk(self.canvas, tb_f)
        self._tb.update()

        # Data-reader toggle
        self._dr_active = False
        self._dr_ann = None
        self._dr_cid = None
        self.btn_dr = tk.Button(
            tb_f, text="🎯 Data Reader", relief=tk.RAISED,
            font=("TkDefaultFont", 9), padx=4, command=self._dr_toggle,
        )
        self.btn_dr.pack(side=tk.LEFT, padx=(8, 2), pady=2)

        self._build_results(p)

    def _build_results(self, p: ttk.Frame) -> None:
        res = ttk.LabelFrame(p, text=" Results ", padding=6)
        res.pack(fill=tk.X, padx=4, pady=(0, 4))

        row0 = ttk.Frame(res)
        row0.pack(fill=tk.X)
        self._rv: dict[str, tk.StringVar] = {}
        for i, (lbl, key, unit) in enumerate([
            ("α surface", "a_surf", ""),
            ("α core", "a_core", ""),
            ("T max core", "T_max", "°C"),
            ("Time", "t_tot", "min"),
        ]):
            ttk.Label(row0, text=lbl + ":", foreground="gray").grid(
                row=0, column=i * 3, padx=(8, 0), pady=2, sticky=tk.W,
            )
            v = tk.StringVar(value="—")
            self._rv[key] = v
            ttk.Label(
                row0, textvariable=v, font=("TkDefaultFont", 12, "bold"),
            ).grid(row=0, column=i * 3 + 1, padx=(3, 0), pady=2, sticky=tk.W)
            if unit:
                ttk.Label(row0, text=unit, foreground="gray").grid(
                    row=0, column=i * 3 + 2, padx=(1, 8), pady=2, sticky=tk.W,
                )

        row1 = ttk.Frame(res)
        row1.pack(fill=tk.X)
        self.lbl_live_info = ttk.Label(
            row1, textvariable=self.v_live_info,
            foreground="#555555", font=("TkFixedFont", 9),
        )
        self.lbl_live_info.pack(side=tk.LEFT, padx=8)

        ef = ttk.Frame(res)
        ef.pack(side=tk.RIGHT, padx=8, anchor=tk.NE)
        ttk.Button(ef, text="📷 PNG", command=self._export_png).pack(side=tk.LEFT, padx=3)
        ttk.Button(ef, text="📊 CSV", command=self._export_csv).pack(side=tk.LEFT, padx=3)
        ttk.Button(ef, text="📄 Summary", command=self._export_txt).pack(side=tk.LEFT, padx=3)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _frow(
        self,
        parent: tk.Misc,
        label: str,
        var: tk.StringVar,
        ftypes: list[tuple[str, str]],
    ) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(4, 0))
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=(0, 2))
        ttk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(
            f, text="📂 Browse", width=10,
            command=lambda v=var, ft=ftypes: self._browse(v, ft),
        ).pack(side=tk.LEFT)

    def _browse(self, var: tk.StringVar, ftypes: list[tuple[str, str]]) -> None:
        p = filedialog.askopenfilename(
            filetypes=ftypes, initialdir=SCRIPT_DIR, parent=self.root,
        )
        if p:
            var.set(p)

    def _add_val_leg(
        self,
        ax,
        labels: list[str],
        colors: list[str],
        loc: str = "lower left",
    ):
        """Add a draggable live-value legend; call AFTER ax.legend()."""
        proxies = [Line2D([0], [0], color=c, lw=1.5) for c in colors]
        leg = ax.legend(
            proxies, labels, loc=loc, fontsize=8, framealpha=0.92, edgecolor="#aaaaaa",
        )
        leg.set_draggable(True)
        return leg

    def _set_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.v_status.set(msg))

    def _set_progress(self, v: float) -> None:
        self.root.after(0, lambda: self.progress.configure(value=v))

    def _set_live(self, t_min: float, a_s: float, a_c: float) -> None:
        self.v_live_info.set(
            f"t = {t_min:.2f} min   │   "
            f"α surface = {a_s:.4f}   │   "
            f"α core = {a_c:.4f}"
        )

    def _parse_thresholds(self) -> list[float]:
        """Parse threshold entry into a sorted list of floats in (0, 1]."""
        raw = self.v_thresholds.get()
        out: list[float] = []
        for tok in raw.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = float(tok)
            except ValueError:
                continue
            if 0.0 < v <= 1.0:
                out.append(v)
        if not out:
            out = [0.8, 1.0]
        return sorted(set(round(v, 4) for v in out))

    # ─────────────────────────────────────────────────────────────────────────
    # Data reader
    # ─────────────────────────────────────────────────────────────────────────

    def _dr_toggle(self) -> None:
        self._dr_active = not self._dr_active
        if self._dr_active:
            self.btn_dr.configure(relief=tk.SUNKEN, bg="#ffe066")
            self._dr_cid = self.canvas.mpl_connect("button_press_event", self._dr_click)
            self._set_status("Data Reader ON — click on chart")
        else:
            self.btn_dr.configure(relief=tk.RAISED, bg="SystemButtonFace")
            if self._dr_cid is not None:
                self.canvas.mpl_disconnect(self._dr_cid)
                self._dr_cid = None
            self._dr_clear()
            self._set_status("Data Reader OFF")

    def _dr_click(self, event) -> None:
        if event.inaxes not in (self.ax_T, self.ax_a):
            return
        ax = event.inaxes
        xc = event.xdata
        yc = event.ydata
        if xc is None or yc is None:
            return
        # Normalize distances by axis ranges so x and y contribute comparably.
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        xr = max(xhi - xlo, 1e-9)
        yr = max(yhi - ylo, 1e-9)
        best_d, best_x, best_y, best_lbl = float("inf"), None, None, ""
        for line in ax.get_lines():
            xd = line.get_xdata()
            yd = line.get_ydata()
            if len(xd) < 2:
                continue
            lbl = line.get_label()
            # Skip auto-labelled internal lines and reference lines (cutoff/threshold).
            if lbl.startswith("_") or lbl.startswith("T_cut"):
                continue
            idx = int(np.clip(np.searchsorted(xd, xc), 0, len(xd) - 1))
            for i in [max(0, idx - 1), idx, min(len(xd) - 1, idx + 1)]:
                dx = (xd[i] - xc) / xr
                dy = (yd[i] - yc) / yr
                d = (dx * dx + dy * dy) ** 0.5
                if d < best_d:
                    best_d = d
                    best_x = xd[i]
                    best_y = yd[i]
                    best_lbl = lbl
        if best_x is None:
            return
        self._dr_clear()
        if ax is self.ax_T:
            txt = f"t = {best_x:.2f} min\nT = {best_y:.1f} °C\n({best_lbl})"
        else:
            txt = f"t = {best_x:.2f} min\nα = {best_y:.4f}\n({best_lbl})"
        self._dr_ann = ax.annotate(
            txt, xy=(best_x, best_y),
            xytext=(18, 18), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffcc", ec="#888888", alpha=0.96),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
            fontsize=9, zorder=10,
        )
        self.canvas.draw_idle()

    def _dr_clear(self) -> None:
        if self._dr_ann is not None:
            try:
                self._dr_ann.remove()
            except (ValueError, AttributeError):
                pass
            self._dr_ann = None
        self.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Axes
    # ─────────────────────────────────────────────────────────────────────────

    def _init_axes(self) -> None:
        self.ax_T.set_ylabel("Temperature (°C)")
        self.ax_T.grid(True, alpha=0.25, lw=0.6)
        self.ax_T.tick_params(labelbottom=False)
        self.ax_a.set_xlabel("Time (min)")
        self.ax_a.set_ylabel("Conversion α")
        self.ax_a.set_ylim(-0.02, 1.05)
        self.ax_a.grid(True, alpha=0.25, lw=0.6)
        self._thresh_anns = {}
        self._thresh_placed_y = []

    # ─────────────────────────────────────────────────────────────────────────
    # Threshold annotations
    # ─────────────────────────────────────────────────────────────────────────

    def _get_y_pos(self, preferred_alpha: float) -> float:
        ymin, ymax = self.ax_a.get_ylim()
        pref = (preferred_alpha - ymin) / (ymax - ymin)
        blocked = [(0.82, 1.0)]
        min_gap = 0.15
        candidates: list[float] = []
        for delta in [0, 0.18, -0.18, 0.36, -0.36, 0.54, -0.54]:
            c = pref + delta
            if 0.04 <= c <= 0.96:
                in_blocked = any(lo <= c <= hi for lo, hi in blocked)
                if not in_blocked:
                    candidates.append(c)
        for c in candidates:
            if all(abs(c - py) >= min_gap for py in self._thresh_placed_y):
                return c
        for c in np.arange(0.75, 0.04, -0.10):
            if all(abs(c - py) >= min_gap for py in self._thresh_placed_y):
                return float(c)
        return 0.5

    def _update_thresholds(
        self,
        t_min: np.ndarray,
        a_surf: np.ndarray,
        a_core: np.ndarray,
    ) -> bool:
        thresholds = self._parse_thresholds()
        checks: list[tuple[str, np.ndarray, float, str, str]] = []
        for th in thresholds:
            # Use 0.999 fudge for thresholds at exactly 1.0.
            th_eff = 0.999 if th >= 0.999 else th
            checks.append((
                f"surf_{th:.2f}", a_surf, th_eff,
                f"Surface ≥ {th:.2f}" if th < 0.999 else "Surface = 1.00",
                COL_SURF,
            ))
            checks.append((
                f"core_{th:.2f}", a_core, th_eff,
                f"Core ≥ {th:.2f}" if th < 0.999 else "Core = 1.00",
                COL_CORE,
            ))

        # Drop annotations whose thresholds were removed from the list.
        active_keys = {key for key, *_ in checks}
        for key in list(self._thresh_anns.keys()):
            if key not in active_keys:
                self._remove_threshold(key)

        changed = False
        for key, arr, th, label, col in checks:
            idx = np.where(arr >= th)[0]
            visible = len(idx) > 0 and len(arr) > 0 and arr[-1] >= th

            if visible and key not in self._thresh_anns:
                t_cross = t_min[idx[0]]
                y_frac = self._get_y_pos(th)
                self._thresh_placed_y.append(y_frac)
                ylo, yhi = self.ax_a.get_ylim()
                y_text_d = ylo + y_frac * (yhi - ylo)
                xlo, xhi = self.ax_a.get_xlim()
                xr = xhi - xlo
                x_offset = xr * 0.04
                x_text = (
                    t_cross + x_offset
                    if t_cross < xlo + xr * 0.65
                    else t_cross - xr * 0.30
                )
                ann = self.ax_a.annotate(
                    f"{label}\nt = {t_cross:.1f} min",
                    xy=(t_cross, th),
                    xytext=(x_text, y_text_d),
                    arrowprops=dict(
                        arrowstyle="-|>", color=col, lw=0.9,
                        connectionstyle="arc3,rad=0.20",
                    ),
                    fontsize=7.5, color=col, va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              alpha=0.92, ec=col, lw=0.7),
                    zorder=6,
                )
                vl = self.ax_a.axvline(t_cross, color=col, ls="--", lw=0.8, alpha=0.45, zorder=2)
                self._thresh_anns[key] = {"ann": ann, "vl": vl, "y": y_frac}
                changed = True

            elif not visible and key in self._thresh_anns:
                self._remove_threshold(key)
                changed = True

        return changed

    def _remove_threshold(self, key: str) -> None:
        obj = self._thresh_anns.pop(key, None)
        if obj is None:
            return
        for art_key in ("ann", "vl"):
            try:
                obj[art_key].remove()
            except (ValueError, AttributeError):
                pass
        if obj["y"] in self._thresh_placed_y:
            self._thresh_placed_y.remove(obj["y"])

    def _clear_all_thresholds(self) -> None:
        for key in list(self._thresh_anns.keys()):
            self._remove_threshold(key)

    # ─────────────────────────────────────────────────────────────────────────
    # Main computation
    # ─────────────────────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self._computing:
            return
        for lbl, v in [("Ea", self.v_ea), ("logA", self.v_la), ("Data", self.v_data)]:
            if not v.get():
                messagebox.showerror("Error", f"{lbl} file not selected.")
                return
            if not os.path.isfile(v.get()):
                messagebox.showerror("Error", f"File not found:\n{v.get()}")
                return
        self._computing = True
        self.btn_run.configure(state=tk.DISABLED)
        self._set_progress(0)
        self.v_live_info.set("")
        self._cancel_animation()
        threading.Thread(target=self._thread, daemon=True).start()

    def _thread(self) -> None:
        warning_msg: str | None = None
        try:
            self._set_status("Loading kinetics…")
            self._set_progress(10)
            kin = load_kinetics(self.v_ea.get(), self.v_la.get())
            self._kinetics = kin

            self._set_status("Loading temperature profile…")
            self._set_progress(30)
            profile = self._load_profile(self.v_data.get())

            if self.v_use_cutoff.get():
                t_s, T_surf, T_core, n_drop = trim_after_cooling(
                    profile.t_s, profile.T_surf, profile.T_core, self.v_cutoff.get(),
                )
            else:
                t_s, T_surf, T_core, n_drop = (
                    profile.t_s, profile.T_surf, profile.T_core, 0,
                )

            if len(t_s) == 0:
                raise ValueError("Trimmed profile is empty.")

            self._set_status(f"{len(t_s)} points (trimmed {n_drop}). Computing α…")
            self._set_progress(55)
            a_surf, diag_s = compute_alpha(t_s, T_surf, kin.ea_fn, kin.la_fn)
            self._set_progress(78)
            a_core, diag_c = compute_alpha(t_s, T_core, kin.ea_fn, kin.la_fn)
            self._set_progress(92)

            max_step = max(diag_s.max_step, diag_c.max_step)
            warn = diag_s.warning_message() or diag_c.warning_message()

            metadata = ExportMetadata(
                ea_path=self.v_ea.get(),
                la_path=self.v_la.get(),
                data_path=self.v_data.get(),
                file_type=self.v_ftype.get(),
                cutoff_used=self.v_use_cutoff.get(),
                cutoff_c=self.v_cutoff.get() if self.v_use_cutoff.get() else None,
                n_trimmed=n_drop,
                ea_unit_converted=kin.ea_unit_converted,
                integration_max_step=max_step,
                integration_warning=warn,
                thresholds=self._parse_thresholds(),
            )
            self._result = ComputationResult(
                t_min=t_s / 60.0,
                T_surf=T_surf, T_core=T_core,
                a_surf=a_surf, a_core=a_core,
                metadata=metadata,
            )
            warning_msg = warn
            self.root.after(0, self._on_done)
            if warn:
                self.root.after(0, lambda w=warn: messagebox.showwarning(
                    "Integration stability", w,
                ))
        except (OSError, ValueError) as e:
            self._set_status(f"ERROR: {e}")
            self.root.after(0, lambda err=e: messagebox.showerror(
                "Computation error", str(err),
            ))
        finally:
            self._computing = False
            self.root.after(0, lambda: self.btn_run.configure(state=tk.NORMAL))

    def _load_profile(self, path: str) -> TempProfile:
        if self.v_ftype.get() == "Excel":
            return load_excel(
                path, self.v_tc.get(), self.v_sc.get(),
                self.v_cc.get(), self.v_dr.get(),
            )
        return load_testo(
            path, self.v_surf_ch.get(), self.v_core_ch.get(),
        )

    def _on_done(self) -> None:
        if self.v_animate.get():
            self._animate_start()
        else:
            self._draw_full()
        self._show_results()

    # ─────────────────────────────────────────────────────────────────────────
    # Static draw + overlay
    # ─────────────────────────────────────────────────────────────────────────

    def _overlay_runs(self) -> None:
        """Draw saved runs as faded background lines on both axes."""
        for run in self._runs:
            if not run.visible:
                continue
            r = run.result
            cs, cc = run.color_pair
            self.ax_T.plot(r.t_min, r.T_surf, color=cs, lw=0.9, alpha=0.45,
                           label=f"_{run.name} surf")
            self.ax_T.plot(r.t_min, r.T_core, color=cc, lw=0.9, alpha=0.45,
                           label=f"_{run.name} core")
            self.ax_a.plot(r.t_min, r.a_surf, color=cs, lw=0.9, alpha=0.45,
                           label=f"{run.name} α-surf")
            self.ax_a.plot(r.t_min, r.a_core, color=cc, lw=0.9, alpha=0.45,
                           label=f"{run.name} α-core")

    def _draw_full(self) -> None:
        if self._result is None:
            return
        r = self._result
        self.ax_T.cla()
        self.ax_a.cla()
        self._init_axes()
        self._overlay_runs()
        t = r.t_min
        self.ax_T.plot(t, r.T_surf, COL_SURF, lw=1.5, label="Surface")
        self.ax_T.plot(t, r.T_core, COL_CORE, lw=1.5, label="Core")
        if self.v_use_cutoff.get():
            self.ax_T.axhline(self.v_cutoff.get(), color=COL_CUT, ls="--", lw=1,
                              label=f"T_cut = {self.v_cutoff.get():.0f} °C")
        leg_T = self.ax_T.legend(fontsize=8, loc="upper right")
        leg_T.set_draggable(True)
        self.ax_T.add_artist(leg_T)
        self._add_val_leg(
            self.ax_T,
            [f"Surface: {r.T_surf[-1]:.1f} °C", f"Core: {r.T_core[-1]:.1f} °C"],
            [COL_SURF, COL_CORE], loc="lower left",
        )
        self.ax_a.plot(t, r.a_surf, COL_SURF, lw=1.5, label="α surface")
        self.ax_a.plot(t, r.a_core, COL_CORE, lw=1.5, label="α core")
        leg_a = self.ax_a.legend(fontsize=8, loc="upper left")
        leg_a.set_draggable(True)
        self.ax_a.add_artist(leg_a)
        self._add_val_leg(
            self.ax_a,
            [f"α surface = {r.a_surf[-1]:.4f}", f"α core = {r.a_core[-1]:.4f}"],
            [COL_SURF, COL_CORE], loc="lower right",
        )
        self._update_thresholds(t, r.a_surf, r.a_core)
        self.press.update(r.T_surf[-1], r.T_core[-1], r.a_surf[-1], r.a_core[-1])
        self.canvas.draw()
        self._set_status("Done.")
        self._set_progress(100)
        self._set_live(t[-1], r.a_surf[-1], r.a_core[-1])

    def _show_results(self) -> None:
        if self._result is None:
            return
        r = self._result
        self._rv["a_surf"].set(f"{r.a_surf[-1]:.4f}")
        self._rv["a_core"].set(f"{r.a_core[-1]:.4f}")
        self._rv["T_max"].set(f"{r.T_core.max():.1f}")
        self._rv["t_tot"].set(f"{r.t_min[-1]:.1f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Async animation (root.after-driven, cancellable)
    # ─────────────────────────────────────────────────────────────────────────

    def _cancel_animation(self) -> None:
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None

    def _animate_start(self) -> None:
        if self._result is None:
            return
        r = self._result
        n = len(r.t_min)
        stp = max(1, n // max(1, self.v_aspeed.get()))
        self.ax_T.cla()
        self.ax_a.cla()
        self._init_axes()
        self._overlay_runs()
        self.ax_T.set_xlim(0, 10)
        self.ax_T.set_ylim(
            min(r.T_surf.min(), r.T_core.min()) - 5,
            max(r.T_surf.max(), r.T_core.max()) + 10,
        )
        self.ax_a.set_xlim(0, 10)
        if self.v_use_cutoff.get():
            self.ax_T.axhline(self.v_cutoff.get(), color=COL_CUT, ls="--", lw=1, alpha=0.8)
        lTs, = self.ax_T.plot([], [], COL_SURF, lw=1.5, label="Surface")
        lTc, = self.ax_T.plot([], [], COL_CORE, lw=1.5, label="Core")
        laS, = self.ax_a.plot([], [], COL_SURF, lw=1.5, label="α surface")
        laC, = self.ax_a.plot([], [], COL_CORE, lw=1.5, label="α core")
        leg_T = self.ax_T.legend(fontsize=8, loc="upper right")
        leg_T.set_draggable(True)
        self.ax_T.add_artist(leg_T)
        vleg_T = self._add_val_leg(
            self.ax_T, ["Surface: — °C", "Core: — °C"],
            [COL_SURF, COL_CORE], loc="lower left",
        )
        leg_a = self.ax_a.legend(fontsize=8, loc="upper left")
        leg_a.set_draggable(True)
        self.ax_a.add_artist(leg_a)
        vleg_a = self._add_val_leg(
            self.ax_a, ["α surface = —", "α core = —"],
            [COL_SURF, COL_CORE], loc="lower right",
        )

        # Build the sequence of indices to draw, ending exactly at n.
        indices = list(range(stp, n, stp)) + [n]
        self._anim_state = dict(
            indices=indices, idx_pos=0,
            lTs=lTs, lTc=lTc, laS=laS, laC=laC,
            vleg_T=vleg_T, vleg_a=vleg_a, n=n,
        )
        self._anim_job = self.root.after(10, self._animate_step)

    def _animate_step(self) -> None:
        if self._result is None or not self._anim_state:
            return
        s = self._anim_state
        if s["idx_pos"] >= len(s["indices"]):
            self._anim_finish()
            return
        i = s["indices"][s["idx_pos"]]
        s["idx_pos"] += 1
        r = self._result
        s["lTs"].set_data(r.t_min[:i], r.T_surf[:i])
        s["lTc"].set_data(r.t_min[:i], r.T_core[:i])
        s["laS"].set_data(r.t_min[:i], r.a_surf[:i])
        s["laC"].set_data(r.t_min[:i], r.a_core[:i])
        cur_t = r.t_min[i - 1]
        x_max = max(10.0, cur_t * 1.05)
        self.ax_T.set_xlim(0, x_max)
        self.ax_a.set_xlim(0, x_max)
        s["vleg_T"].get_texts()[0].set_text(f"Surface: {r.T_surf[i-1]:.1f} °C")
        s["vleg_T"].get_texts()[1].set_text(f"Core:    {r.T_core[i-1]:.1f} °C")
        s["vleg_a"].get_texts()[0].set_text(f"α surface = {r.a_surf[i-1]:.4f}")
        s["vleg_a"].get_texts()[1].set_text(f"α core    = {r.a_core[i-1]:.4f}")
        self._set_live(cur_t, r.a_surf[i - 1], r.a_core[i - 1])
        self.press.update(r.T_surf[i - 1], r.T_core[i - 1], r.a_surf[i - 1], r.a_core[i - 1])
        self.progress["value"] = int(i / s["n"] * 100)
        self.canvas.draw_idle()
        self._anim_job = self.root.after(20, self._animate_step)

    def _animate_finish(self) -> None:
        if self._result is None:
            return
        r = self._result
        self._update_thresholds(r.t_min, r.a_surf, r.a_core)
        self._show_results()
        self._set_status("Done.")
        self._set_progress(100)
        self.canvas.draw_idle()
        self._anim_job = None
        self._anim_state = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Saved runs (overlay)
    # ─────────────────────────────────────────────────────────────────────────

    def _save_current_as_run(self) -> None:
        if self._result is None:
            messagebox.showinfo("Save run", "Run a computation first.")
            return
        default_name = (
            os.path.splitext(os.path.basename(self.v_data.get()))[0]
            if self.v_data.get() else f"run_{len(self._runs) + 1}"
        )
        name = self._ask_string("Save run", "Name for this run:", default_name)
        if not name:
            return
        color_pair = OVERLAY_PALETTE[len(self._runs) % len(OVERLAY_PALETTE)]
        self._runs.append(SavedRun(
            name=name, result=self._result, color_pair=color_pair,
        ))
        self._refresh_runs_list()
        self._draw_full()
        self._set_status(f"Saved run: {name}")

    def _ask_string(self, title: str, prompt: str, initial: str) -> str | None:
        from tkinter.simpledialog import askstring
        return askstring(title, prompt, initialvalue=initial, parent=self.root)

    def _clear_all_runs(self) -> None:
        if not self._runs:
            return
        if not messagebox.askyesno("Clear runs", f"Remove all {len(self._runs)} saved runs?"):
            return
        self._runs = []
        self._refresh_runs_list()
        if self._result is not None:
            self._draw_full()

    def _toggle_run_visibility(self, idx: int) -> None:
        self._runs[idx].visible = not self._runs[idx].visible
        if self._result is not None:
            self._draw_full()

    def _remove_run(self, idx: int) -> None:
        del self._runs[idx]
        self._refresh_runs_list()
        if self._result is not None:
            self._draw_full()

    def _refresh_runs_list(self) -> None:
        for child in self._runs_inner.winfo_children():
            child.destroy()
        if not self._runs:
            ttk.Label(
                self._runs_inner, text="(no saved runs)",
                foreground="gray", font=("TkDefaultFont", 9, "italic"),
            ).pack(anchor=tk.W, padx=4, pady=4)
            return
        for i, run in enumerate(self._runs):
            row = ttk.Frame(self._runs_inner)
            row.pack(fill=tk.X, padx=2, pady=1)
            var = tk.BooleanVar(value=run.visible)
            ttk.Checkbutton(
                row, variable=var,
                command=lambda idx=i, v=var: self._set_run_visible(idx, v.get()),
            ).pack(side=tk.LEFT)
            cs, cc = run.color_pair
            sw = tk.Canvas(row, width=14, height=14, highlightthickness=0)
            sw.create_rectangle(0, 0, 7, 14, fill=cs, outline=cs)
            sw.create_rectangle(7, 0, 14, 14, fill=cc, outline=cc)
            sw.pack(side=tk.LEFT, padx=2)
            ttk.Label(
                row, text=run.name, font=("TkDefaultFont", 9),
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                row, text="✕", width=2,
                command=lambda idx=i: self._remove_run(idx),
            ).pack(side=tk.RIGHT)

    def _set_run_visible(self, idx: int, visible: bool) -> None:
        self._runs[idx].visible = visible
        if self._result is not None:
            self._draw_full()

    # ─────────────────────────────────────────────────────────────────────────
    # Kinetics preview window
    # ─────────────────────────────────────────────────────────────────────────

    def _preview_kinetics(self) -> None:
        ea_path = self.v_ea.get()
        la_path = self.v_la.get()
        if not (ea_path and la_path and os.path.isfile(ea_path) and os.path.isfile(la_path)):
            messagebox.showinfo(
                "Preview kinetics",
                "Select valid Ea and logA files first.",
            )
            return
        try:
            kin = load_kinetics(ea_path, la_path)
        except (OSError, ValueError) as e:
            messagebox.showerror("Preview kinetics", str(e))
            return

        win = tk.Toplevel(self.root)
        win.title("Kinetics preview — Ea(α), logA(α)")
        win.geometry("720x560")
        fig = Figure(figsize=(7, 5), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        ax1.plot(kin.alpha_ea, kin.ea / 1000.0, "o-", color="#c0392b", ms=3, lw=1.2)
        ax1.set_ylabel("Ea (kJ/mol)")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Friedman kinetics  ({len(kin.alpha_ea)} α-points)")
        if kin.ea_unit_converted:
            ax1.text(
                0.98, 0.05, "Ea auto-scaled kJ→J·1000",
                transform=ax1.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray",
            )

        ax2.plot(kin.alpha_la, kin.la, "o-", color="#2980b9", ms=3, lw=1.2)
        ax2.set_xlabel("Conversion α")
        ax2.set_ylabel("log₁₀ A")
        ax2.grid(True, alpha=0.3)

        # Sanity check: warn if Ea or logA jump abruptly between grid points.
        warn_lines: list[str] = []
        if len(kin.ea) > 1:
            d_ea = np.abs(np.diff(kin.ea))
            if np.max(d_ea) > 0.5 * np.mean(np.abs(kin.ea)):
                warn_lines.append("Ea(α) has large jumps — check Friedman fit at boundaries.")
        if len(kin.la) > 1 and np.max(np.abs(np.diff(kin.la))) > 5.0:
            warn_lines.append("logA(α) has jumps > 5 — likely fit instability.")
        if warn_lines:
            ax1.text(
                0.02, 0.95, "\n".join("⚠ " + w for w in warn_lines),
                transform=ax1.transAxes, ha="left", va="top",
                fontsize=8, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#c0392b"),
            )

        fig.tight_layout()
        fc = FigureCanvasTkAgg(fig, master=win)
        fc.draw()
        fc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(fc, win).update()

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────────────────────────────────

    def _sim_load(self) -> None:
        sea = self.v_sim_ea.get()
        sla = self.v_sim_la.get()
        if sea and sla and os.path.isfile(sea) and os.path.isfile(sla):
            try:
                kin = load_kinetics(sea, sla)
            except (OSError, ValueError) as e:
                messagebox.showerror("Error", f"Cannot load simulation kinetics:\n{e}")
                return
        elif self._kinetics is not None:
            kin = self._kinetics
        else:
            messagebox.showerror(
                "Error",
                "No kinetics found.\nEnter Ea + logA files above, "
                "or load them in the Files tab.",
            )
            return

        path = self.v_sim_data.get() or self.v_data.get()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Select an Excel file for simulation.")
            return
        self.v_sim_status.set("Loading and computing…")
        self.root.update()
        try:
            profile = load_excel(
                path, self.v_tc.get(), self.v_sc.get(),
                self.v_cc.get(), self.v_dr.get(),
            )
            t_s, T_surf, T_core, _ = trim_after_cooling(
                profile.t_s, profile.T_surf, profile.T_core,
                self.v_cutoff.get() if self.v_use_cutoff.get() else -273.15,
            )
            a_surf, _ = compute_alpha(t_s, T_surf, kin.ea_fn, kin.la_fn)
            a_core, _ = compute_alpha(t_s, T_core, kin.ea_fn, kin.la_fn)
            self._sim_t = t_s
            self._sim_T_surf = T_surf
            self._sim_T_core = T_core
            self._sim_a_surf = a_surf
            self._sim_a_core = a_core
            self._sim_n = len(t_s)
            self._sim_pos = 0
            self._sim_loaded = True
            dur = t_s[-1] / 60
            self.v_sim_status.set(
                f"✓ Ready — {dur:.1f} min, {len(t_s)} points\n"
                f"α core={a_core[-1]:.4f}  α surface={a_surf[-1]:.4f}"
            )
            for b in [self.btn_s_rew, self.btn_s_play]:
                b.configure(state=tk.NORMAL)
            self._sim_setup_chart()
        except (OSError, ValueError) as e:
            self.v_sim_status.set(f"Error: {e}")
            messagebox.showerror("Simulation error", str(e))

    def _sim_setup_chart(self) -> None:
        if self._sim_t is None:
            return
        self.ax_T.cla()
        self.ax_a.cla()
        self._init_axes()
        self.ax_T.set_xlim(0, 10)
        self.ax_T.set_ylim(
            min(self._sim_T_surf.min(), self._sim_T_core.min()) - 5,
            max(self._sim_T_surf.max(), self._sim_T_core.max()) + 10,
        )
        self.ax_a.set_xlim(0, 10)
        if self.v_use_cutoff.get():
            self.ax_T.axhline(self.v_cutoff.get(), color=COL_CUT, ls="--", lw=1, alpha=0.8)
        # Faded full preview
        self.ax_T.plot(self._sim_t / 60, self._sim_T_surf, COL_SURF, lw=0.5, alpha=0.15)
        self.ax_T.plot(self._sim_t / 60, self._sim_T_core, COL_CORE, lw=0.5, alpha=0.15)
        self.ax_a.plot(self._sim_t / 60, self._sim_a_surf, COL_SURF, lw=0.5, alpha=0.15)
        self.ax_a.plot(self._sim_t / 60, self._sim_a_core, COL_CORE, lw=0.5, alpha=0.15)
        lTs, = self.ax_T.plot([], [], COL_SURF, lw=1.5, label="Surface")
        lTc, = self.ax_T.plot([], [], COL_CORE, lw=1.5, label="Core")
        laS, = self.ax_a.plot([], [], COL_SURF, lw=1.5, label="α surface")
        laC, = self.ax_a.plot([], [], COL_CORE, lw=1.5, label="α core")
        leg_T = self.ax_T.legend(fontsize=8, loc="upper right")
        leg_T.set_draggable(True)
        self.ax_T.add_artist(leg_T)
        self._sim_val_leg_T = self._add_val_leg(
            self.ax_T, ["Surface: — °C", "Core: — °C"],
            [COL_SURF, COL_CORE], loc="lower left",
        )
        leg_a = self.ax_a.legend(fontsize=8, loc="upper left")
        leg_a.set_draggable(True)
        self.ax_a.add_artist(leg_a)
        _tproxy = Line2D([0], [0], color=COL_CUR, lw=1.5, ls=":")
        time_leg = self.ax_a.legend(
            [_tproxy], ["t = 0.00 min"], loc="lower right",
            fontsize=9, framealpha=0.92, edgecolor=COL_CUR,
        )
        time_leg.set_draggable(True)
        self.ax_a.add_artist(time_leg)
        self._sim_time_leg = time_leg
        self._sim_val_leg_a = self._add_val_leg(
            self.ax_a, ["α surface = —", "α core = —"],
            [COL_SURF, COL_CORE], loc="lower left",
        )
        vT = self.ax_T.axvline(0, color=COL_CUR, lw=1.2, alpha=0.8, ls=":")
        va = self.ax_a.axvline(0, color=COL_CUR, lw=1.2, alpha=0.8, ls=":")
        self._sim_lines = dict(lTs=lTs, lTc=lTc, laS=laS, laC=laC, vT=vT, va=va)
        self.canvas.draw()
        self.v_sim_time.set(f"0.0 / {self._sim_t[-1]/60:.1f} min")
        self.v_sim_pos_frac.set(0.0)
        self.v_live_info.set("")
        self._set_status("Simulation ready — press ▶")

    def _sim_update_chart(self) -> None:
        if not self._sim_lines or self._sim_t is None:
            return
        pos = max(self._sim_pos, 1)
        t = self._sim_t[:pos] / 60
        ln = self._sim_lines
        ln["lTs"].set_data(t, self._sim_T_surf[:pos])
        ln["lTc"].set_data(t, self._sim_T_core[:pos])
        ln["laS"].set_data(t, self._sim_a_surf[:pos])
        ln["laC"].set_data(t, self._sim_a_core[:pos])
        cur_t = self._sim_t[pos - 1] / 60
        ln["vT"].set_xdata([cur_t, cur_t])
        ln["va"].set_xdata([cur_t, cur_t])
        self.v_sim_pos_frac.set(pos / self._sim_n)
        self.v_sim_time.set(f"{cur_t:.2f} / {self._sim_t[-1]/60:.1f} min")
        self._rv["a_surf"].set(f"{self._sim_a_surf[pos-1]:.4f}")
        self._rv["a_core"].set(f"{self._sim_a_core[pos-1]:.4f}")
        self._rv["T_max"].set(f"{self._sim_T_core[:pos].max():.1f}")
        self._rv["t_tot"].set(f"{cur_t:.1f}")
        x_max = max(10.0, cur_t * 1.05)
        self.ax_T.set_xlim(0, x_max)
        self.ax_a.set_xlim(0, x_max)
        self._set_live(cur_t, self._sim_a_surf[pos - 1], self._sim_a_core[pos - 1])
        self._update_thresholds(t, self._sim_a_surf[:pos], self._sim_a_core[:pos])
        if self._sim_time_leg:
            self._sim_time_leg.get_texts()[0].set_text(f"t = {cur_t:.2f} min")
        if self._sim_val_leg_a:
            self._sim_val_leg_a.get_texts()[0].set_text(f"α surface = {self._sim_a_surf[pos-1]:.4f}")
            self._sim_val_leg_a.get_texts()[1].set_text(f"α core    = {self._sim_a_core[pos-1]:.4f}")
        if self._sim_val_leg_T:
            self._sim_val_leg_T.get_texts()[0].set_text(f"Surface: {self._sim_T_surf[pos-1]:.1f} °C")
            self._sim_val_leg_T.get_texts()[1].set_text(f"Core:    {self._sim_T_core[pos-1]:.1f} °C")
        self.press.update(
            self._sim_T_surf[pos - 1], self._sim_T_core[pos - 1],
            self._sim_a_surf[pos - 1], self._sim_a_core[pos - 1],
        )
        self.canvas.draw_idle()

    def _sim_tick(self) -> None:
        if not self._sim_running or self._sim_t is None:
            return
        if self._sim_pos >= self._sim_n:
            self._sim_running = False
            self.btn_s_play.configure(text="▶")
            self._set_status("Simulation complete.")
            return
        self._sim_pos += 1
        self._sim_update_chart()
        if self._sim_pos < self._sim_n:
            dt = self._sim_t[self._sim_pos] - self._sim_t[self._sim_pos - 1]
            self._sim_job = self.root.after(
                max(16, int(dt * 1000 / self.v_sim_speed.get())),
                self._sim_tick,
            )
        else:
            self._sim_running = False
            self.btn_s_play.configure(text="▶")
            self._set_status("Simulation complete.")

    def _sim_toggle(self) -> None:
        if not self._sim_loaded:
            return
        if self._sim_running:
            self._sim_running = False
            if self._sim_job:
                self.root.after_cancel(self._sim_job)
                self._sim_job = None
            self.btn_s_play.configure(text="▶")
            self._set_status("Paused.")
        else:
            if self._sim_pos >= self._sim_n:
                self._sim_pos = 0
            self._sim_running = True
            self.btn_s_play.configure(text="⏸")
            self._set_status("Simulation running…")
            self._sim_tick()

    def _sim_rewind(self) -> None:
        self._sim_running = False
        if self._sim_job:
            self.root.after_cancel(self._sim_job)
            self._sim_job = None
        self.btn_s_play.configure(text="▶")
        self._sim_pos = 0
        self.v_sim_pos_frac.set(0.0)
        if self._sim_t is not None:
            self.v_sim_time.set(f"0.0 / {self._sim_t[-1]/60:.1f} min")
        if self._sim_lines:
            for k in ["lTs", "lTc", "laS", "laC"]:
                self._sim_lines[k].set_data([], [])
            self._sim_lines["vT"].set_xdata([0, 0])
            self._sim_lines["va"].set_xdata([0, 0])
            self._clear_all_thresholds()
            self.canvas.draw_idle()
        self.ax_T.set_xlim(0, 10)
        self.ax_a.set_xlim(0, 10)
        if self._sim_time_leg:
            self._sim_time_leg.get_texts()[0].set_text("t = 0.00 min")
        if self._sim_val_leg_a:
            self._sim_val_leg_a.get_texts()[0].set_text("α surface = —")
            self._sim_val_leg_a.get_texts()[1].set_text("α core    = —")
        if self._sim_val_leg_T:
            self._sim_val_leg_T.get_texts()[0].set_text("Surface: — °C")
            self._sim_val_leg_T.get_texts()[1].set_text("Core:    — °C")
        self.press.update(20.0, 20.0, 0.0, 0.0)
        for k in ["a_surf", "a_core", "T_max", "t_tot"]:
            self._rv[k].set("—")
        self.v_live_info.set("")
        self._set_status("Rewound to start.")

    def _sim_scrub(self, val: str | float) -> None:
        if not self._sim_loaded:
            return
        frac = float(val)
        self._sim_pos = max(1, int(frac * self._sim_n))
        self._sim_update_chart()

    # ─────────────────────────────────────────────────────────────────────────
    # Live monitor (incremental file reads)
    # ─────────────────────────────────────────────────────────────────────────

    def _live_toggle(self) -> None:
        if self.v_live.get():
            self.lbl_live.configure(text="● Active", foreground="green")
            self._live_cache.reset()
            self._last_mt = None
            self._live_poll()
        else:
            if self._live_job:
                self.root.after_cancel(self._live_job)
                self._live_job = None
            self.lbl_live.configure(text="● Inactive", foreground="gray")

    def _live_poll(self) -> None:
        if not self.v_live.get():
            return
        p = self.v_data.get()
        if p and os.path.isfile(p):
            try:
                mt = os.path.getmtime(p)
                if self._last_mt is None or mt > self._last_mt:
                    self._last_mt = mt
                    if not self._computing:
                        self._live_incremental_update(p)
            except OSError:
                pass
        self._live_job = self.root.after(
            self.v_linterv.get() * 1000, self._live_poll,
        )

    def _live_incremental_update(self, path: str) -> None:
        """Read only newly-appended rows, append to cache, recompute α."""
        # Need kinetics for the recompute. If never loaded, fall back to _start.
        if self._kinetics is None:
            for v in (self.v_ea, self.v_la):
                if not v.get() or not os.path.isfile(v.get()):
                    self._set_status("Live: kinetics not loaded — skipping update.")
                    return
            try:
                self._kinetics = load_kinetics(self.v_ea.get(), self.v_la.get())
            except (OSError, ValueError) as e:
                self._set_status(f"Live: kinetics error: {e}")
                return

        try:
            if self.v_ftype.get() == "Excel":
                start_row = self._live_cache.last_row + 1 if self._live_cache.last_row else None
                profile = load_excel(
                    path, self.v_tc.get(), self.v_sc.get(),
                    self.v_cc.get(), self.v_dr.get(),
                    start_row=start_row,
                    t0_override=self._live_cache.t0_raw,
                )
            else:
                profile = load_testo(
                    path, self.v_surf_ch.get(), self.v_core_ch.get(),
                    start_line=self._live_cache.last_row + 1 if self._live_cache.last_row else 0,
                    t0_override=self._live_cache.t0_raw,
                )
        except (OSError, ValueError) as e:
            self._set_status(f"Live: load error: {e}")
            return

        if len(profile.t_s) == 0 and self._live_cache.t_s.size == 0:
            self._set_status("Live: no data yet.")
            return

        # Append new rows.
        if len(profile.t_s) > 0:
            self._live_cache.t_s = np.concatenate([self._live_cache.t_s, profile.t_s])
            self._live_cache.T_surf = np.concatenate([self._live_cache.T_surf, profile.T_surf])
            self._live_cache.T_core = np.concatenate([self._live_cache.T_core, profile.T_core])
            self._live_cache.last_row = profile.last_row
            self._live_cache.t0_raw = profile.t0_raw

        # Trim + recompute α from scratch on the cumulative dataset.
        if self.v_use_cutoff.get():
            t_s, T_surf, T_core, n_drop = trim_after_cooling(
                self._live_cache.t_s, self._live_cache.T_surf,
                self._live_cache.T_core, self.v_cutoff.get(),
            )
        else:
            t_s, T_surf, T_core, n_drop = (
                self._live_cache.t_s, self._live_cache.T_surf,
                self._live_cache.T_core, 0,
            )
        if len(t_s) == 0:
            return
        kin = self._kinetics
        a_surf, diag_s = compute_alpha(t_s, T_surf, kin.ea_fn, kin.la_fn)
        a_core, diag_c = compute_alpha(t_s, T_core, kin.ea_fn, kin.la_fn)
        warn = diag_s.warning_message() or diag_c.warning_message()
        metadata = ExportMetadata(
            ea_path=self.v_ea.get(),
            la_path=self.v_la.get(),
            data_path=path,
            file_type=self.v_ftype.get(),
            cutoff_used=self.v_use_cutoff.get(),
            cutoff_c=self.v_cutoff.get() if self.v_use_cutoff.get() else None,
            n_trimmed=n_drop,
            ea_unit_converted=kin.ea_unit_converted,
            integration_max_step=max(diag_s.max_step, diag_c.max_step),
            integration_warning=warn,
            thresholds=self._parse_thresholds(),
        )
        self._result = ComputationResult(
            t_min=t_s / 60.0, T_surf=T_surf, T_core=T_core,
            a_surf=a_surf, a_core=a_core, metadata=metadata,
        )
        self._draw_full()
        self._show_results()
        self._set_status(
            f"Live: updated ({len(t_s)} pts, +{len(profile.t_s)} new rows)."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────

    def _get_export_data(self) -> ComputationResult | None:
        if self._result is not None:
            return self._result
        if self._sim_loaded and self._sim_pos > 0 and self._sim_t is not None:
            return ComputationResult(
                t_min=self._sim_t / 60,
                T_surf=self._sim_T_surf, T_core=self._sim_T_core,
                a_surf=self._sim_a_surf, a_core=self._sim_a_core,
                metadata=ExportMetadata(
                    ea_path=self.v_sim_ea.get() or self.v_ea.get(),
                    la_path=self.v_sim_la.get() or self.v_la.get(),
                    data_path=self.v_sim_data.get() or self.v_data.get(),
                    file_type="Excel (simulation)",
                    cutoff_used=self.v_use_cutoff.get(),
                    cutoff_c=self.v_cutoff.get() if self.v_use_cutoff.get() else None,
                    thresholds=self._parse_thresholds(),
                ),
            )
        messagebox.showinfo("Export", "Run computation or simulation first.")
        return None

    def _export_png(self) -> None:
        if self._get_export_data() is None:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png", parent=self.root,
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg")],
            initialdir=SCRIPT_DIR,
            initialfile=default_filename("conversion", "png"),
        )
        if p:
            self.fig.savefig(p, dpi=300, bbox_inches="tight")
            self._set_status(f"PNG: {os.path.basename(p)}")

    def _export_csv(self) -> None:
        d = self._get_export_data()
        if d is None:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".csv", parent=self.root,
            filetypes=[("CSV", "*.csv")],
            initialdir=SCRIPT_DIR,
            initialfile=default_filename("conversion", "csv"),
        )
        if not p:
            return
        export_csv(p, d.t_min, d.T_surf, d.T_core, d.a_surf, d.a_core, d.metadata)
        self._set_status(f"CSV: {os.path.basename(p)}")

    def _export_txt(self) -> None:
        d = self._get_export_data()
        if d is None:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".txt", parent=self.root,
            filetypes=[("Text", "*.txt")],
            initialdir=SCRIPT_DIR,
            initialfile=default_filename("summary", "txt"),
        )
        if not p:
            return
        export_summary(p, d.t_min, d.T_surf, d.T_core, d.a_surf, d.a_core, d.metadata)
        self._set_status(f"Summary: {os.path.basename(p)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Clear / settings
    # ─────────────────────────────────────────────────────────────────────────

    def _clear(self) -> None:
        self._cancel_animation()
        self._result = None
        self.ax_T.cla()
        self.ax_a.cla()
        self._init_axes()
        self._overlay_runs()
        self.canvas.draw()
        for v in self._rv.values():
            v.set("—")
        self.v_live_info.set("")
        self.progress["value"] = 0
        self._set_status("Cleared.")

    def _settings_map(self) -> dict[str, tk.Variable]:
        return {
            "ea": self.v_ea, "la": self.v_la, "data": self.v_data,
            "ftype": self.v_ftype, "cutoff": self.v_cutoff,
            "use_cutoff": self.v_use_cutoff,
            "surf_ch": self.v_surf_ch, "core_ch": self.v_core_ch,
            "tc": self.v_tc, "sc": self.v_sc, "cc": self.v_cc, "dr": self.v_dr,
            "animate": self.v_animate, "aspeed": self.v_aspeed,
            "linterv": self.v_linterv,
            "sim_ea": self.v_sim_ea, "sim_la": self.v_sim_la,
            "sim_data": self.v_sim_data, "press_temp": self.v_press_temp,
            "thresholds": self.v_thresholds,
        }

    def _save_settings(self) -> None:
        s = {k: v.get() for k, v in self._settings_map().items()}
        try:
            with open(SETTINGS, "w") as f:
                json.dump(s, f, indent=2)
            self._set_status("Settings saved.")
        except OSError as e:
            messagebox.showerror("Settings", f"Cannot save settings: {e}")

    def _load_settings(self) -> None:
        if not os.path.isfile(SETTINGS):
            return
        try:
            with open(SETTINGS) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._set_status(f"Settings load error: {e}")
            return
        for k, v in self._settings_map().items():
            if k in s:
                try:
                    v.set(s[k])
                except (tk.TclError, ValueError):
                    pass


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
