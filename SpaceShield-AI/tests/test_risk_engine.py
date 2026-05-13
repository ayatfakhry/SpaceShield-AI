"""
tests/test_risk_engine.py
===========================
Unit tests for the composite risk scoring engine.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.risk_engine import (
    RiskEngine, RiskScore, score_to_band,
    score_pc, score_miss_distance, score_velocity,
    score_lead_time, score_mass, score_rcs,
    score_altitude, score_asset_priority,
    RISK_WEIGHTS
)
from src.collision_prediction import ConjunctionEvent


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_event(event_id="CDM-001", pc=1e-4, miss=1.0, vel=10.0,
               lead=24.0, mass=500.0, rcs=2.0, alt=500.0):
    return ConjunctionEvent(
        event_id=event_id, primary_id="SAT-A", secondary_id="DEB-B",
        tca=lead * 3600, miss_distance_km=miss, rel_velocity_kms=vel,
        pc=pc, combined_rcs_m2=rcs, combined_mass_kg=mass,
        primary_altitude=alt, secondary_altitude=alt + 2.0,
        lead_time_hours=lead
    )


@pytest.fixture
def engine():
    return RiskEngine()


@pytest.fixture
def moderate_event():
    return make_event(pc=5e-5, miss=2.0, lead=24.0)


@pytest.fixture
def critical_event():
    return make_event(event_id="CDM-CRIT", pc=1e-2, miss=0.05, vel=14.0,
                      lead=2.0, mass=2000.0, rcs=10.0)


# ─── Individual Factor Scorers ────────────────────────────────────────────────

class TestIndividualScorers:

    @pytest.mark.parametrize("pc,expected_min,expected_max", [
        (0.0,    0,   1),
        (1e-8,   0,  10),
        (1e-5,  30,  60),
        (1e-3,  80, 100),
        (1.0,  100, 100),
    ])
    def test_score_pc_range(self, pc, expected_min, expected_max):
        s = score_pc(pc)
        assert expected_min <= s <= expected_max, \
            f"score_pc({pc:.0e}) = {s:.1f}, expected [{expected_min},{expected_max}]"

    def test_score_pc_monotone(self):
        pcs    = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        scores = [score_pc(p) for p in pcs]
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))

    def test_score_miss_zero_is_100(self):
        assert score_miss_distance(0.0) == 100.0

    def test_score_miss_at_threshold_is_zero(self):
        from src.risk_engine import CONJUNCTION_THRESHOLD_KM
        assert score_miss_distance(CONJUNCTION_THRESHOLD_KM) == pytest.approx(0.0)

    def test_score_miss_monotone_decreasing(self):
        dists  = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
        scores = [score_miss_distance(d) for d in dists]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_score_velocity_bounded(self):
        for v in [0.0, 1.0, 5.0, 10.0, 15.0, 20.0]:
            s = score_velocity(v)
            assert 0 <= s <= 100

    def test_score_lead_time_monotone(self):
        times  = [0.5, 2.0, 6.0, 12.0, 24.0, 48.0, 72.0]
        scores = [score_lead_time(t) for t in times]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_score_lead_time_short_high(self):
        assert score_lead_time(0.5) > 90

    def test_score_lead_time_long_low(self):
        assert score_lead_time(72.0) < 5

    def test_score_mass_bounded(self):
        for m in [0.01, 1, 100, 1000, 10000]:
            assert 0 <= score_mass(m) <= 100

    def test_score_rcs_bounded(self):
        for r in [0.001, 0.1, 1.0, 10.0, 50.0]:
            assert 0 <= score_rcs(r) <= 100

    def test_score_altitude_bounded(self):
        for alt in [200, 400, 550, 800, 1200, 2000]:
            assert 0 <= score_altitude(alt) <= 100

    def test_score_asset_priority(self):
        assert score_asset_priority(True)  > score_asset_priority(False)
        assert score_asset_priority(True)  == 80.0
        assert score_asset_priority(False) == 20.0

    def test_weights_sum_to_one(self):
        total = sum(RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9


class TestRiskEngine:

    def test_score_bounded(self, engine, moderate_event):
        rs = engine.score_event(moderate_event)
        assert 0 <= rs.total_score <= 100

    def test_critical_scores_higher_than_moderate(self, engine, moderate_event, critical_event):
        rs_mod  = engine.score_event(moderate_event)
        rs_crit = engine.score_event(critical_event)
        assert rs_crit.total_score > rs_mod.total_score, \
            f"Critical ({rs_crit.total_score:.1f}) ≤ Moderate ({rs_mod.total_score:.1f})"

    def test_risk_band_assigned(self, engine, moderate_event):
        rs = engine.score_event(moderate_event)
        bands = [b for _, _, b, _ in
                 [(0,10,"MINIMAL","A"),(10,30,"LOW","B"),(30,50,"MODERATE","C"),
                  (50,70,"HIGH","D"),(70,90,"SEVERE","E"),(90,100,"CATASTROPHIC","F")]]
        assert rs.risk_band in bands

    def test_risk_grade_assigned(self, engine, moderate_event):
        rs = engine.score_event(moderate_event)
        assert rs.risk_grade in ("A","B","C","D","E","F")

    def test_score_events_sorted_descending(self, engine):
        events = [make_event(f"CDM-{i}", pc=p, miss=m)
                  for i, (p, m) in enumerate([
                      (1e-2, 0.05), (1e-4, 1.0), (1e-7, 5.0)
                  ])]
        scores = engine.score_events(events)
        vals   = [s.total_score for s in scores]
        assert all(vals[i] >= vals[i+1] for i in range(len(vals)-1))

    def test_active_satellite_scores_higher(self, engine):
        ev = make_event()
        rs_active   = engine.score_event(ev, primary_active=True)
        rs_inactive = engine.score_event(ev, primary_active=False)
        assert rs_active.total_score >= rs_inactive.total_score

    def test_summary_stats_keys(self, engine):
        events = [make_event(f"CDM-{i}", pc=1e-5) for i in range(5)]
        scores = engine.score_events(events)
        stats  = engine.summary_stats(scores)
        for key in ["n_events", "mean_score", "max_score", "min_score"]:
            assert key in stats

    def test_component_scores_in_dict(self, engine, moderate_event):
        rs = engine.score_event(moderate_event)
        for key in RISK_WEIGHTS.keys():
            assert key in rs.component_scores

    def test_to_dataframe(self, engine):
        events = [make_event(f"CDM-{i}") for i in range(3)]
        scores = engine.score_events(events)
        df     = engine.to_dataframe(scores)
        assert len(df) == 3
        assert "risk_score" in df.columns
        assert "risk_band"  in df.columns

    def test_score_to_band(self):
        assert score_to_band(5)[0]  == "MINIMAL"
        assert score_to_band(20)[0] == "LOW"
        assert score_to_band(40)[0] == "MODERATE"
        assert score_to_band(60)[0] == "HIGH"
        assert score_to_band(80)[0] == "SEVERE"
        assert score_to_band(95)[0] == "CATASTROPHIC"
