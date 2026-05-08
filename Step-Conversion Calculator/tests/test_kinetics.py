"""
Sanity tests for kinetics module.

Run from the project root:
    python3 -m unittest tests/test_kinetics.py

These tests exercise the core integration logic without requiring real
Kinetics-Neo files, so they protect against regressions when refactoring.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinetics import R, _parse_neo, compute_alpha, load_kinetics  # noqa: E402
from loaders import trim_after_cooling  # noqa: E402


def _const_kinetics(ea_J: float, log_A: float):
    """Constant-Ea, constant-logA Friedman kinetics (independent of α)."""
    ea_fn = lambda _a: ea_J
    la_fn = lambda _a: log_A
    return ea_fn, la_fn


class TestComputeAlpha(unittest.TestCase):
    def test_isothermal_first_order_matches_analytic(self):
        """
        For constant T and α-independent k, dα/dt = k(1-α) ⇒
        α(t) = 1 - exp(-k·t). FE should converge as Δt → 0.
        """
        T_C = 150.0
        T_K = T_C + 273.15
        ea = 80_000.0
        log_A = 8.0
        k = 10.0 ** log_A * math.exp(-ea / (R * T_K))

        # Use a fine time grid so FE is accurate.
        t_s = np.linspace(0, 60, 6001)
        T = np.full_like(t_s, T_C)
        ea_fn, la_fn = _const_kinetics(ea, log_A)

        alpha, diag = compute_alpha(t_s, T, ea_fn, la_fn)
        analytic = 1.0 - np.exp(-k * t_s)
        # FE should be within 1% in the steep part.
        np.testing.assert_allclose(alpha, analytic, atol=1e-2)
        self.assertTrue(diag.is_stable, msg=diag.warning_message())

    def test_alpha_clamped_to_one(self):
        """Once α reaches 1, remaining values stay at 1."""
        ea = 50_000.0
        log_A = 12.0
        T_C = 200.0
        ea_fn, la_fn = _const_kinetics(ea, log_A)
        t_s = np.linspace(0, 600, 1001)
        T = np.full_like(t_s, T_C)
        alpha, _ = compute_alpha(t_s, T, ea_fn, la_fn)
        self.assertAlmostEqual(alpha[-1], 1.0)
        self.assertGreaterEqual(alpha.min(), 0.0)
        self.assertLessEqual(alpha.max(), 1.0)

    def test_unstable_step_warning(self):
        """Coarse Δt with fast kinetics ⇒ diag.is_stable == False."""
        ea = 50_000.0
        log_A = 12.0
        ea_fn, la_fn = _const_kinetics(ea, log_A)
        # 60-second steps — way too coarse for the chosen k.
        t_s = np.array([0.0, 60.0, 120.0, 180.0])
        T = np.array([200.0, 200.0, 200.0, 200.0])
        _, diag = compute_alpha(t_s, T, ea_fn, la_fn)
        self.assertFalse(diag.is_stable)
        self.assertIsNotNone(diag.warning_message())

    def test_zero_temperature_no_reaction(self):
        """Below ~0 K kinetics rate is essentially zero ⇒ α stays at a0."""
        ea_fn, la_fn = _const_kinetics(100_000.0, 8.0)
        t_s = np.linspace(0, 60, 61)
        T = np.full_like(t_s, -200.0)  # near absolute zero
        alpha, _ = compute_alpha(t_s, T, ea_fn, la_fn)
        np.testing.assert_array_almost_equal(alpha, np.zeros_like(alpha))


class TestTrim(unittest.TestCase):
    def test_trims_after_both_below_cutoff(self):
        """Should NOT trim while only one of (surf, core) is below cutoff."""
        t = np.arange(10, dtype=float)
        T_surf = np.array([20., 60., 100., 120., 100., 60., 40., 30., 25., 20.])
        T_core = np.array([20., 50., 90., 110., 100., 80., 60., 50., 40., 30.])
        # cutoff=45: surf goes <45 at idx 6; core goes <45 at idx 8.
        # Both below at idx 8 → trim from there.
        t_t, _, _, n_drop = trim_after_cooling(t, T_surf, T_core, cutoff_c=45.0)
        self.assertEqual(len(t_t), 8)
        self.assertEqual(n_drop, 2)

    def test_no_trim_if_never_below(self):
        t = np.arange(5, dtype=float)
        T = np.array([100., 110., 120., 110., 100.])
        t_t, _, _, n_drop = trim_after_cooling(t, T, T, cutoff_c=45.0)
        self.assertEqual(len(t_t), 5)
        self.assertEqual(n_drop, 0)


class TestParseNeo(unittest.TestCase):
    def test_parses_kinetics_neo_format(self):
        content = (
            "Project: test\n"
            "Export: Friedman\n\n"
            "Conversion\tEa / (kJ/mol)\t±Error\n"
            "--------\n"
            "0.10\t75,5\t1,2\n"
            "0.20\t80.1\t1.5\n"
            "0.30\t82.0\t1.8\n"
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8",
        ) as fh:
            fh.write(content)
            path = fh.name
        try:
            alphas, vals = _parse_neo(path)
            np.testing.assert_array_almost_equal(alphas, [0.1, 0.2, 0.3])
            np.testing.assert_array_almost_equal(vals, [75.5, 80.1, 82.0])
        finally:
            os.unlink(path)


class TestLoadKineticsUnits(unittest.TestCase):
    def test_kJ_to_J_conversion(self):
        """Mean Ea < 1000 ⇒ flagged as kJ/mol and scaled to J/mol."""
        body = "Conv\tEa\n--------\n"
        ea_lines = "\n".join(f"{a:.2f}\t{75 + a*10:.3f}" for a in np.arange(0.1, 1.0, 0.1))
        la_lines = "\n".join(f"{a:.2f}\t{8.0 + a:.3f}" for a in np.arange(0.1, 1.0, 0.1))
        with tempfile.TemporaryDirectory() as tmp:
            ea_path = os.path.join(tmp, "ea.txt")
            la_path = os.path.join(tmp, "la.txt")
            with open(ea_path, "w") as f:
                f.write(body + ea_lines + "\n")
            with open(la_path, "w") as f:
                f.write(body + la_lines + "\n")
            kin = load_kinetics(ea_path, la_path)
        self.assertTrue(kin.ea_unit_converted)
        # Original mean was ~80 kJ/mol; after conversion expect ~80_000 J/mol.
        self.assertAlmostEqual(np.mean(kin.ea), 80_000.0, delta=20_000.0)


if __name__ == "__main__":
    unittest.main()
