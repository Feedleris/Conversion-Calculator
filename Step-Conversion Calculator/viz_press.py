"""
Hot-press visualization widget.

Encapsulates the Tkinter Canvas drawing of the press schematic and the
temperature/conversion readout table.
"""

from __future__ import annotations

import tkinter as tk

COL_SURF = "#e74c3c"
COL_CORE = "#3498db"


class PressViz:
    """
    Draws a schematic hot-press (top/bottom plates, hydraulic ram, board with
    a vertical temperature gradient) and a small T/α readout table below it.

    All geometry is computed from the canvas size on each <Configure>, so the
    widget adapts to its parent's layout.
    """

    def __init__(
        self,
        parent: tk.Misc,
        press_temp_var: tk.DoubleVar,
        bg: str = "#e0e0e0",
    ) -> None:
        self.canvas = tk.Canvas(parent, bg=bg, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._press_temp_var = press_temp_var
        self._geo: dict | None = None
        self._last_state: tuple[float, float, float, float] = (20.0, 20.0, 0.0, 0.0)
        self.canvas.bind("<Configure>", lambda _e: self._redraw_all())

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, T_surf: float, T_core: float, a_surf: float, a_core: float) -> None:
        """Update the dynamic content (board gradient + readouts)."""
        self._last_state = (T_surf, T_core, a_surf, a_core)
        self._draw_dynamic()

    def refresh(self) -> None:
        """Force a full redraw (e.g. when press_temp changed)."""
        self._redraw_all()

    # ── Color helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _alpha_to_hex(alpha: float) -> str:
        """Map [0, 1] → hex color along blue→cyan→yellow→orange→red."""
        a = max(0.0, min(1.0, float(alpha)))
        stops = [
            (0.00, (44, 95, 167)),
            (0.25, (68, 154, 195)),
            (0.50, (253, 213, 65)),
            (0.75, (239, 133, 37)),
            (1.00, (192, 44, 44)),
        ]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if a <= t1 or i == len(stops) - 2:
                f = max(0.0, min(1.0, (a - t0) / (t1 - t0) if t1 > t0 else 1.0))
                r = int(c0[0] + f * (c1[0] - c0[0]))
                g = int(c0[1] + f * (c1[1] - c0[1]))
                b = int(c0[2] + f * (c1[2] - c0[2]))
                return f"#{r:02x}{g:02x}{b:02x}"
        c = stops[-1][1]
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    def _temp_to_hex(self, T: float, T_room: float = 20.0) -> str:
        T_max = max(self._press_temp_var.get(), T_room + 1.0)
        return self._alpha_to_hex((T - T_room) / (T_max - T_room))

    # ── Geometry ─────────────────────────────────────────────────────────────

    def _compute_geo(self) -> dict | None:
        cv = self.canvas
        W = cv.winfo_width()
        H = cv.winfo_height()
        if W < 10 or H < 10:
            return None
        SH = int(H * 0.67)
        TH = H - SH
        cx = W // 2
        pw_h = int(W * 0.24)
        bw_h = int(W * 0.19)
        cw = max(12, int(W * 0.048))
        cgap = 3
        px = cx - pw_h
        pw = 2 * pw_h
        bx = cx - bw_h
        bw = 2 * bw_h
        cx1 = px - cgap - cw
        cx2 = px + pw + cgap
        ph = max(28, int(SH * 0.13))
        bh = max(18, int(SH * 0.16))
        by = int(SH * 0.53)
        beam_y = int(SH * 0.03)
        beam_h = max(16, int(SH * 0.07))
        cyl_h = max(20, int(SH * 0.09))
        base_y = int(SH * 0.87)
        base_h = max(14, int(SH * 0.07))
        col_top = beam_y + beam_h
        col_bot = base_y + base_h
        cyl_hw = max(14, int(W * 0.065))
        rod_hw = max(8, int(W * 0.038))
        cyl_y = col_top
        rod_top = cyl_y + cyl_h
        rod_bot = by - ph
        return dict(
            W=W, H=H, SH=SH, TH=TH, cx=cx,
            px=px, pw=pw, bx=bx, bw=bw,
            cx1=cx1, cx2=cx2, cw=cw,
            ph=ph, bh=bh, by=by,
            beam_y=beam_y, beam_h=beam_h,
            cyl_h=cyl_h, cyl_hw=cyl_hw,
            rod_hw=rod_hw, rod_top=rod_top, rod_bot=rod_bot,
            base_y=base_y, base_h=base_h,
            col_top=col_top, col_bot=col_bot,
        )

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _redraw_all(self) -> None:
        self._draw_static()
        self._draw_dynamic()

    def _draw_static(self) -> None:
        cv = self.canvas
        cv.delete("static")
        g = self._compute_geo()
        if g is None:
            return
        self._geo = g
        cx, cx1, cx2, cw = g["cx"], g["cx1"], g["cx2"], g["cw"]
        beam_y, beam_h = g["beam_y"], g["beam_h"]
        base_y, base_h = g["base_y"], g["base_h"]
        col_top, col_bot = g["col_top"], g["col_bot"]
        cyl_h, cyl_hw = g["cyl_h"], g["cyl_hw"]
        rod_hw = g["rod_hw"]
        rod_top, rod_bot = g["rod_top"], g["rod_bot"]
        SH = g["SH"]

        def S(*a, **kw):
            cv.create_rectangle(*a, tags="static", **kw)

        def L(*a, **kw):
            cv.create_line(*a, tags="static", **kw)

        # Top crosshead beam
        S(cx1, beam_y, cx2 + cw, beam_y + beam_h, fill="#2e2e2e", outline="#111", width=1)
        for hx in range(cx1 + 8, cx2 + cw - 4, max(8, (cx2 + cw - cx1) // 12)):
            L(hx, beam_y + 2, hx - 6, beam_y + beam_h - 2, fill="#484848")

        # Columns
        S(cx1, col_top, cx1 + cw, col_bot, fill="#5e5e5e", outline="#3a3a3a", width=1)
        L(cx1 + 3, col_top + 4, cx1 + 3, col_bot - 4, fill="#909090", width=2)
        S(cx2, col_top, cx2 + cw, col_bot, fill="#5e5e5e", outline="#3a3a3a", width=1)
        L(cx2 + 3, col_top + 4, cx2 + 3, col_bot - 4, fill="#909090", width=2)

        # Bottom base
        S(cx1, base_y, cx2 + cw, base_y + base_h, fill="#2e2e2e", outline="#111", width=1)
        for hx in range(cx1 + 8, cx2 + cw - 4, max(8, (cx2 + cw - cx1) // 12)):
            L(hx, base_y + 2, hx - 6, base_y + base_h - 2, fill="#484848")

        # Hydraulic cylinder
        S(cx - cyl_hw, col_top, cx + cyl_hw, col_top + cyl_h,
          fill="#4a4a4a", outline="#222", width=1)
        L(cx - cyl_hw + 2, col_top + 3, cx + cyl_hw - 2, col_top + 3, fill="#7a7a7a")
        S(cx + cyl_hw, col_top + 6, cx + cyl_hw + max(4, cw // 3), col_top + 14,
          fill="#3a3a3a", outline="#222", width=1)

        # Ram rod
        if rod_bot > rod_top:
            S(cx - rod_hw, rod_top, cx + rod_hw, rod_bot,
              fill="#8a8a8a", outline="#5a5a5a", width=1)
            L(cx - rod_hw + 3, rod_top + 2, cx - rod_hw + 3, rod_bot - 2,
              fill="#b8b8b8", width=1)

        # Separator
        L(4, SH, g["W"] - 4, SH, fill="#aaaaaa", width=1)

    def _draw_dynamic(self) -> None:
        cv = self.canvas
        cv.delete("dyn")
        g = self._geo
        if g is None:
            return
        T_surf, T_core, a_surf, a_core = self._last_state
        cx = g["cx"]
        px, pw = g["px"], g["pw"]
        bx, bw = g["bx"], g["bw"]
        ph, bh, by = g["ph"], g["bh"], g["by"]
        SH, TH, W = g["SH"], g["TH"], g["W"]
        T_press = self._press_temp_var.get()
        pc = self._temp_to_hex(T_press)

        fpl = ("TkDefaultFont", max(8, ph // 4))
        ftC = ("TkDefaultFont", max(9, ph // 3), "bold")

        # Top press plate
        py0 = by - ph
        cv.create_rectangle(px, py0, px + pw, by,
                            fill=pc, outline="#111111", width=2, tags="dyn")
        step = max(8, pw // 16)
        for hx in range(px + step, px + pw - 2, step):
            cv.create_line(hx, py0 + 3, hx - step // 2, by - 3,
                           fill="#444444", width=1, tags="dyn")
        cv.create_text(cx, py0 + ph // 3, text="Press plate", font=fpl, fill="#fff", tags="dyn")
        cv.create_text(cx, py0 + ph * 2 // 3, text=f"{T_press:.0f} °C", font=ftC, fill="#fff", tags="dyn")

        # Bottom press plate
        py0b = by + bh
        cv.create_rectangle(px, py0b, px + pw, py0b + ph,
                            fill=pc, outline="#111111", width=2, tags="dyn")
        for hx in range(px + step, px + pw - 2, step):
            cv.create_line(hx, py0b + 3, hx - step // 2, py0b + ph - 3,
                           fill="#444444", width=1, tags="dyn")
        cv.create_text(cx, py0b + ph // 3, text="Press plate", font=fpl, fill="#fff", tags="dyn")
        cv.create_text(cx, py0b + ph * 2 // 3, text=f"{T_press:.0f} °C", font=ftC, fill="#fff", tags="dyn")

        # Board gradient
        N = 50
        for j in range(N):
            frac = j / N
            d_surf = min(frac, 1.0 - frac)
            T_here = T_surf + (T_core - T_surf) * (d_surf / 0.5)
            color = self._temp_to_hex(T_here)
            y0 = by + int(j * bh / N)
            y1 = by + int((j + 1) * bh / N) + 1
            cv.create_rectangle(bx, y0, bx + bw, y1, fill=color, outline="", tags="dyn")
        cv.create_rectangle(bx, by, bx + bw, by + bh,
                            fill="", outline="#111111", width=2, tags="dyn")
        cv.create_text(cx, by + bh // 2, text="Board",
                       font=("TkDefaultFont", max(7, bh // 5)), fill="#ffffff", tags="dyn")

        # Sensor dots
        r = max(4, bw // 18)
        for y, col in [(by, COL_SURF), (by + bh // 2, COL_CORE), (by + bh, COL_SURF)]:
            cv.create_oval(bx + bw - r, y - r, bx + bw + r, y + r,
                           fill=col, outline="white", width=1.5, tags="dyn")

        # Data table
        row_h = TH // 2
        cy1 = SH + row_h // 2
        cy2 = SH + row_h + row_h // 2
        pad_l = int(W * 0.05)
        dot_r = max(5, row_h // 5)
        dot_x = pad_l + dot_r
        lbl_x = dot_x + dot_r + 8
        val_x = int(W * 0.38)
        alp_x = int(W * 0.68)
        fs = max(9, min(14, row_h // 4))
        fnt_l = ("TkDefaultFont", fs, "bold")
        fnt_v = ("TkFixedFont", fs)

        cv.create_oval(dot_x - dot_r, cy1 - dot_r, dot_x + dot_r, cy1 + dot_r,
                       fill=COL_SURF, outline="white", width=1, tags="dyn")
        cv.create_text(lbl_x, cy1, text="Surface", anchor="w",
                       font=fnt_l, fill=COL_SURF, tags="dyn")
        cv.create_text(val_x, cy1, text=f"T = {T_surf:6.1f} °C", anchor="w",
                       font=fnt_v, fill="#111111", tags="dyn")
        cv.create_text(alp_x, cy1, text=f"α = {a_surf:.4f}", anchor="w",
                       font=fnt_v, fill=COL_SURF, tags="dyn")

        cv.create_oval(dot_x - dot_r, cy2 - dot_r, dot_x + dot_r, cy2 + dot_r,
                       fill=COL_CORE, outline="white", width=1, tags="dyn")
        cv.create_text(lbl_x, cy2, text="Core", anchor="w",
                       font=fnt_l, fill=COL_CORE, tags="dyn")
        cv.create_text(val_x, cy2, text=f"T = {T_core:6.1f} °C", anchor="w",
                       font=fnt_v, fill="#111111", tags="dyn")
        cv.create_text(alp_x, cy2, text=f"α = {a_core:.4f}", anchor="w",
                       font=fnt_v, fill=COL_CORE, tags="dyn")
