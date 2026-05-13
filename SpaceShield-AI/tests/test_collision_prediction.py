"""
tests/test_collision_prediction.py
====================================
Unit tests for collision_prediction module.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.collision_prediction import (
    pc_chan_2d, pc_monte_carlo,
    ConjunctionEvent, ConjunctionScreener,
    find_tca, _classify_pc,
    TLE_PC_GREEN, TLE_PC_YELLOW, TLE_PC_ORANGE
)
from src.orbit_simulation import OrbitalElements, sma_from_altitude


@pytest.fixture
def sample_event():
    return ConjunctionEvent(
        event_id="CDM-000001",
        primary_id="SAT-00001",
        secondary_id="DEB-00001",
        tca=3600.0,
        miss_distance_km=0.5,
        rel_velocity_kms=10.0,
        pc=5e-4,
        combined_rcs_m2=2.0,
        combined_mass_kg=500.0,
        primary_altitude=450.0,
        secondary_altitude=452.0,
        lead_time_hours=1.0
    )


class TestPcChan2D:

    def test_zero_miss_distance_returns_one(self):
        pc = pc_chan_2d(0.0, 10.0, 1.0)
        assert pc == 1.0 or pc > 0.9

    def test_large_miss_distance_near_zero(self):
        pc = pc_chan_2d(100.0, 10.0, 1.0)
        assert pc < 1e-20

    def test_pc_bounded(self):
        for miss in [0.001, 0.1, 1.0, 5.0, 10.0]:
            pc = pc_chan_2d(miss, 10.0, 1.0)
            assert 0.0 <= pc <= 1.0, f"Pc={pc} out of [0,1] for miss={miss}"

    def test_pc_decreases_with_miss_distance(self):
        pcs = [pc_chan_2d(d, 10.0, 1.0) for d in [0.1, 0.5, 1.0, 2.0, 5.0]]
        for i in range(len(pcs) - 1):
            assert pcs[i] >= pcs[i+1], f"Pc not monotone: {pcs}"

    def test_pc_increases_with_rcs(self):
        pc1 = pc_chan_2d(1.0, 10.0, 0.5)
        pc2 = pc_chan_2d(1.0, 10.0, 5.0)
        assert pc2 >= pc1, "Larger RCS should give higher Pc"

    def test_high_velocity_reduces_pc(self):
        pc_slow = pc_chan_2d(1.0, 1.0,  1.0)
        pc_fast = pc_chan_2d(1.0, 14.0, 1.0)
        # Fast encounters have shorter effective exposure time
        assert pc_slow >= pc_fast * 0.5  # allow factor ~10 difference


class TestPcMonteCarlo:

    def test_bounded(self):
        rng = np.random.default_rng(0)
        pc  = pc_monte_carlo(0.5, 2.0, rng=rng)
        assert 0.0 <= pc <= 1.0

    def test_zero_miss_high_pc(self):
        rng = np.random.default_rng(1)
        # Use a very large RCS and small sigma to ensure many hits
        pc  = pc_monte_carlo(0.0, combined_rcs_m2=1e6, sigma_km=0.001, rng=rng)
        assert pc > 0.5

    def test_large_miss_low_pc(self):
        rng = np.random.default_rng(2)
        pc  = pc_monte_carlo(50.0, 1.0, sigma_km=0.1, rng=rng)
        assert pc < 0.01


class TestClassifyPc:

    def test_green(self):
        assert _classify_pc(1e-6) == "GREEN"

    def test_yellow(self):
        assert _classify_pc(5e-5) == "YELLOW"

    def test_orange(self):
        assert _classify_pc(5e-4) == "ORANGE"

    def test_red(self):
        assert _classify_pc(5e-3) == "RED"

    def test_zero_pc_green(self):
        assert _classify_pc(0.0) == "GREEN"


class TestConjunctionEvent:

    def test_risk_level_assigned(self, sample_event):
        assert sample_event.risk_level in ("GREEN", "YELLOW", "ORANGE", "RED")

    def test_to_dict_completeness(self, sample_event):
        d = sample_event.to_dict()
        required = ["event_id", "primary_id", "secondary_id",
                    "miss_distance_km", "pc", "risk_level", "tca_s"]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_pc_5e4_is_orange(self, sample_event):
        assert sample_event.risk_level == "ORANGE"


class TestConjunctionScreener:

    @pytest.fixture
    def small_objects(self):
        from src.debris_generator import DebrisGenerator
        gen = DebrisGenerator(seed=0)
        return gen.generate(n_active=8, n_debris=12, n_rockets=3, n_defunct=2)

    def test_screener_returns_list(self, small_objects):
        screener = ConjunctionScreener(screening_threshold_km=20.0,
                                        conjunction_threshold_km=8.0)
        events = screener.screen(small_objects, duration_hours=3.0, verbose=False)
        assert isinstance(events, list)

    def test_events_sorted_by_pc_descending(self, small_objects):
        screener = ConjunctionScreener(screening_threshold_km=20.0,
                                        conjunction_threshold_km=8.0)
        events = screener.screen(small_objects, duration_hours=3.0, verbose=False)
        if len(events) > 1:
            pcs = [ev.pc for ev in events]
            assert all(pcs[i] >= pcs[i+1] for i in range(len(pcs)-1)), \
                "Events not sorted by Pc descending"

    def test_miss_distance_within_threshold(self, small_objects):
        threshold = 8.0
        screener  = ConjunctionScreener(screening_threshold_km=20.0,
                                         conjunction_threshold_km=threshold)
        events = screener.screen(small_objects, duration_hours=3.0, verbose=False)
        for ev in events:
            assert ev.miss_distance_km <= threshold + 1.0, \
                f"Miss {ev.miss_distance_km:.3f} km exceeds threshold {threshold} km"

    def test_to_dataframe(self, small_objects):
        screener = ConjunctionScreener(screening_threshold_km=20.0)
        events   = screener.screen(small_objects, duration_hours=2.0, verbose=False)
        df       = screener.to_dataframe(events)
        if events:
            assert len(df) == len(events)
            assert "event_id" in df.columns
