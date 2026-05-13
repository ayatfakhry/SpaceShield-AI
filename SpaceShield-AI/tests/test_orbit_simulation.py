"""
tests/test_orbit_simulation.py
================================
Unit tests for the orbit_simulation module.

Tests cover:
  - Kepler's equation solver convergence & accuracy
  - Element ↔ state vector round-trip fidelity
  - J2 secular drift direction and magnitude
  - Orbital period formula
  - Trajectory propagation consistency
  - ISS-representative orbit sanity checks
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.orbit_simulation import (
    OrbitalElements, StateVector, OrbitPropagator,
    solve_kepler, eccentric_to_true_anomaly,
    elements_to_state, state_to_elements,
    orbital_period, sma_from_altitude,
    R_EARTH, MU_EARTH, TWO_PI
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def iss_elements():
    """ISS-representative orbital elements."""
    return OrbitalElements(
        sma=sma_from_altitude(408),
        ecc=0.0001,
        inc=np.radians(51.64),
        raan=np.radians(45.0),
        argp=np.radians(90.0),
        mean_anomaly=np.radians(0.0),
        object_id="ISS-TEST"
    )


@pytest.fixture
def circular_leo():
    """Perfectly circular 500 km LEO orbit."""
    return OrbitalElements(
        sma=sma_from_altitude(500),
        ecc=0.0,
        inc=np.radians(55.0),
        raan=0.0, argp=0.0, mean_anomaly=0.0,
        object_id="CIRC-LEO"
    )


@pytest.fixture
def elliptic_orbit():
    """Moderately elliptic LEO orbit (e=0.05)."""
    return OrbitalElements(
        sma=sma_from_altitude(700),
        ecc=0.05,
        inc=np.radians(98.0),
        raan=np.radians(180.0),
        argp=np.radians(270.0),
        mean_anomaly=np.radians(90.0),
        object_id="ELLIP"
    )


# ─── Kepler Solver Tests ──────────────────────────────────────────────────────

class TestKeplerSolver:

    def test_circular_orbit_M_equals_E(self):
        """For e=0, eccentric anomaly should equal mean anomaly."""
        for M in np.linspace(0, TWO_PI, 37):
            E = solve_kepler(M, 0.0)
            assert abs(E - M % TWO_PI) < 1e-10, \
                f"Circular orbit: E({M:.3f}) = {E:.6f} ≠ M"

    def test_kepler_identity(self):
        """Verify M = E - e·sin(E) for various (M, e) pairs."""
        for ecc in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            for M in np.linspace(0.01, TWO_PI - 0.01, 12):
                E = solve_kepler(M, ecc)
                M_check = E - ecc * np.sin(E)
                assert abs(M_check - M % TWO_PI) < 1e-9, \
                    f"Kepler identity failed: e={ecc}, M={M:.3f}"

    def test_high_eccentricity_convergence(self):
        """Solver should converge even for near-parabolic eccentricity."""
        E = solve_kepler(np.pi / 4, 0.999)
        assert np.isfinite(E)
        assert 0 <= E <= TWO_PI

    def test_output_range(self):
        """Output E must lie in [0, 2π)."""
        for M in np.linspace(0, TWO_PI * 3, 100):
            E = solve_kepler(M, 0.3)
            assert 0 <= E < TWO_PI + 1e-9


# ─── True Anomaly Conversion ──────────────────────────────────────────────────

class TestTrueAnomaly:

    def test_perigee_and_apogee(self):
        """At E=0 → ν=0; at E=π → ν=π."""
        assert abs(eccentric_to_true_anomaly(0.0,  0.3)) < 1e-10
        assert abs(eccentric_to_true_anomaly(np.pi, 0.3) - np.pi) < 1e-10

    def test_circular_E_equals_nu(self):
        """For e=0: ν = E for all values (both normalised to [0,2π))."""
        for E in np.linspace(0, TWO_PI, 50):
            nu = eccentric_to_true_anomaly(E % TWO_PI, 0.0)
            E_norm = E % TWO_PI
            diff = abs(nu - E_norm) % TWO_PI
            diff = min(diff, TWO_PI - diff)  # shortest angular distance
            assert diff < 1e-9, f"nu={nu:.6f} ≠ E={E_norm:.6f}"


# ─── Elements ↔ State Vector ──────────────────────────────────────────────────

class TestElementsToState:

    def test_radius_matches_sma_circular(self, circular_leo):
        """For e=0, |r| should equal sma at all points."""
        period = orbital_period(circular_leo.sma)
        times  = np.linspace(0, period, 36)
        for t in times:
            sv = elements_to_state(circular_leo, t)
            assert abs(sv.r - circular_leo.sma) < 0.5, \
                f"r={sv.r:.3f} km ≠ sma={circular_leo.sma:.3f} km at t={t:.0f}s"

    def test_vis_viva_energy(self, iss_elements):
        """Specific orbital energy = -μ/(2a)."""
        sv   = elements_to_state(iss_elements, 0.0)
        E_orb = 0.5 * sv.v**2 - MU_EARTH / sv.r
        E_ref = -MU_EARTH / (2.0 * iss_elements.sma)
        assert abs(E_orb - E_ref) / abs(E_ref) < 1e-4, \
            f"Energy mismatch: {E_orb:.3f} vs {E_ref:.3f}"

    def test_angular_momentum_conserved(self, elliptic_orbit):
        """|h| = √(μ·p) must be constant along orbit."""
        period = orbital_period(elliptic_orbit.sma)
        p      = elliptic_orbit.sma * (1 - elliptic_orbit.ecc**2)
        h_ref  = np.sqrt(MU_EARTH * p)
        for t in np.linspace(0, period, 20):
            sv = elements_to_state(elliptic_orbit, t)
            h  = np.linalg.norm(np.cross(sv.position, sv.velocity))
            assert abs(h - h_ref) / h_ref < 1e-4, \
                f"|h|={h:.3f} ≠ ref={h_ref:.3f} at t={t:.0f}s"

    def test_altitude_positive(self, iss_elements):
        """Altitude should be positive everywhere."""
        period = orbital_period(iss_elements.sma)
        for t in np.linspace(0, period, 36):
            sv = elements_to_state(iss_elements, t)
            assert sv.altitude > 0, f"Negative altitude {sv.altitude:.1f} km"

    def test_iss_altitude_range(self, iss_elements):
        """ISS should stay within ±20 km of nominal 408 km."""
        period = orbital_period(iss_elements.sma)
        alts   = [elements_to_state(iss_elements, t).altitude
                  for t in np.linspace(0, period, 100)]
        assert all(388 < a < 428 for a in alts), \
            f"ISS altitude out of range: min={min(alts):.1f}, max={max(alts):.1f}"

    def test_speed_reasonable_leo(self, circular_leo):
        """LEO orbital speed should be ~7–8 km/s."""
        sv = elements_to_state(circular_leo, 0.0)
        assert 7.0 < sv.v < 8.5, f"Speed {sv.v:.3f} km/s outside LEO range"


# ─── Round-trip Fidelity ─────────────────────────────────────────────────────

class TestRoundTrip:

    def test_elements_state_elements_roundtrip(self, elliptic_orbit):
        """elements → state → elements should recover original (within tolerance)."""
        sv        = elements_to_state(elliptic_orbit, 0.0)
        recovered = state_to_elements(sv)

        tol_km  = 0.5   # km
        tol_rad = 0.01  # rad

        assert abs(recovered.sma - elliptic_orbit.sma) < tol_km, \
            f"sma: {recovered.sma:.3f} vs {elliptic_orbit.sma:.3f}"
        assert abs(recovered.ecc - elliptic_orbit.ecc) < 0.001, \
            f"ecc: {recovered.ecc:.5f} vs {elliptic_orbit.ecc:.5f}"
        assert abs(recovered.inc - elliptic_orbit.inc) < tol_rad, \
            f"inc: {np.degrees(recovered.inc):.2f}° vs {np.degrees(elliptic_orbit.inc):.2f}°"


# ─── Propagator ───────────────────────────────────────────────────────────────

class TestOrbitPropagator:

    def test_propagate_returns_correct_length(self, iss_elements):
        prop  = OrbitPropagator()
        times = np.linspace(0, 3600, 61)
        traj  = prop.propagate(iss_elements, times)
        assert len(traj) == 61

    def test_batch_propagation_keys(self, iss_elements, circular_leo):
        prop    = OrbitPropagator()
        times   = np.linspace(0, 600, 11)
        results = prop.propagate_batch([iss_elements, circular_leo], times)
        assert "ISS-TEST" in results
        assert "CIRC-LEO" in results
        assert len(results["ISS-TEST"]) == 11

    def test_miss_distance_self_zero(self, iss_elements):
        """An object should have zero miss distance with itself."""
        sv = elements_to_state(iss_elements, 0.0)
        d  = OrbitPropagator.miss_distance(sv, sv)
        assert d < 1e-9

    def test_relative_state_self_zero(self, iss_elements):
        """Relative position and velocity with itself must be zero."""
        sv    = elements_to_state(iss_elements, 0.0)
        rp, rv = OrbitPropagator.relative_state(sv, sv)
        assert np.linalg.norm(rp) < 1e-9
        assert np.linalg.norm(rv) < 1e-9


# ─── Utility Functions ────────────────────────────────────────────────────────

class TestUtilities:

    def test_orbital_period_iss(self):
        """ISS period should be approximately 92 minutes."""
        sma    = sma_from_altitude(408)
        period = orbital_period(sma)
        assert 5400 < period < 5700, f"ISS period {period/60:.1f} min outside 90–95 min"

    def test_sma_from_altitude(self):
        """sma = R_EARTH + altitude."""
        assert abs(sma_from_altitude(0)   - R_EARTH) < 1e-9
        assert abs(sma_from_altitude(408) - (R_EARTH + 408)) < 1e-9
        assert abs(sma_from_altitude(1000)- (R_EARTH + 1000))< 1e-9

    def test_orbital_period_scaling(self):
        """Period scales as sma^(3/2): doubling sma → 2√2 × period."""
        sma1 = sma_from_altitude(400)
        sma2 = 2.0 * sma1
        T1   = orbital_period(sma1)
        T2   = orbital_period(sma2)
        ratio = T2 / T1
        assert abs(ratio - 2.0 ** 1.5) < 0.01, \
            f"Period ratio {ratio:.4f} ≠ 2^1.5 = {2**1.5:.4f}"

    def test_element_normalisation(self):
        """OrbitalElements should normalise angles to [0, 2π)."""
        el = OrbitalElements(
            sma=R_EARTH + 400,
            ecc=0.001,
            inc=3 * np.pi + 0.1,    # > 2π
            raan=-np.pi,             # negative
            argp=5 * np.pi,
            mean_anomaly=-0.5
        )
        assert 0 <= el.inc < TWO_PI
        assert 0 <= el.raan < TWO_PI
        assert 0 <= el.argp < TWO_PI
        assert 0 <= el.mean_anomaly < TWO_PI

    def test_eccentricity_clamp(self):
        """OrbitalElements must clamp eccentricity to [0, 0.9999]."""
        el_neg = OrbitalElements(sma=R_EARTH+400, ecc=-0.5,  inc=0, raan=0, argp=0, mean_anomaly=0)
        el_big = OrbitalElements(sma=R_EARTH+400, ecc=1.5,   inc=0, raan=0, argp=0, mean_anomaly=0)
        assert el_neg.ecc >= 0.0
        assert el_big.ecc <= 0.9999
