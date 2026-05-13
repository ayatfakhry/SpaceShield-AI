"""
tests/test_maneuver_recommender.py
====================================
Unit tests for the maneuver recommendation system.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.maneuver_recommender import (
    ManeuverRecommender, ManeuverRecommendation,
    tsiolkovsky_fuel, minimum_dv_for_pc_reduction,
    MANEUVER_TYPES
)
from src.collision_prediction import ConjunctionEvent
from src.risk_engine import RiskEngine, RiskScore


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_event(event_id="CDM-001", pc=1e-4, miss=1.0, vel=10.0,
               lead=24.0, mass=500.0, rcs=2.0):
    return ConjunctionEvent(
        event_id=event_id, primary_id="SAT-A", secondary_id="DEB-B",
        tca=lead * 3600, miss_distance_km=miss, rel_velocity_kms=vel,
        pc=pc, combined_rcs_m2=rcs, combined_mass_kg=mass,
        primary_altitude=500.0, secondary_altitude=502.0,
        lead_time_hours=lead
    )


def make_risk_score(event_id="CDM-001", score=50.0, band="MODERATE"):
    return RiskScore(
        event_id=event_id, primary_id="SAT-A", secondary_id="DEB-B",
        total_score=score, risk_band=band, risk_grade="C",
        component_scores={}, threat_level="MEDIUM"
    )


@pytest.fixture
def recommender():
    return ManeuverRecommender()


# ─── Tsiolkovsky Fuel ─────────────────────────────────────────────────────────

class TestTsiolkovskyFuel:

    def test_zero_dv_zero_fuel(self):
        assert tsiolkovsky_fuel(0.0, 300.0) == pytest.approx(0.0)

    def test_fuel_positive(self):
        fuel = tsiolkovsky_fuel(1.0, 300.0)
        assert fuel > 0

    def test_fuel_less_than_mass(self):
        fuel = tsiolkovsky_fuel(10.0, 300.0)
        assert fuel < 300.0

    def test_larger_dv_more_fuel(self):
        f1 = tsiolkovsky_fuel(1.0, 300.0)
        f2 = tsiolkovsky_fuel(5.0, 300.0)
        assert f2 > f1

    def test_higher_isp_less_fuel(self):
        f_low  = tsiolkovsky_fuel(2.0, 300.0, isp_s=200)
        f_high = tsiolkovsky_fuel(2.0, 300.0, isp_s=450)
        assert f_high < f_low


# ─── Minimum Dv Estimation ────────────────────────────────────────────────────

class TestMinimumDv:

    def test_returns_positive(self):
        dv = minimum_dv_for_pc_reduction(0.5, 10.0, 2.0, target_pc=1e-5)
        assert dv > 0

    def test_close_approach_requires_more_dv(self):
        dv_close = minimum_dv_for_pc_reduction(0.01, 10.0, 2.0, target_pc=1e-5)
        dv_far   = minimum_dv_for_pc_reduction(4.0,  10.0, 2.0, target_pc=1e-5)
        assert dv_close >= dv_far  # very close requires at least as much dv

    def test_large_rcs_requires_more_dv(self):
        dv_small = minimum_dv_for_pc_reduction(1.0, 10.0, 0.1, target_pc=1e-5)
        dv_large = minimum_dv_for_pc_reduction(1.0, 10.0, 20.0, target_pc=1e-5)
        assert dv_large >= dv_small


# ─── ManeuverRecommender ──────────────────────────────────────────────────────

class TestManeuverRecommender:

    def test_no_action_for_low_pc(self, recommender):
        ev = make_event(pc=1e-8, miss=8.0, lead=48.0)
        rs = make_risk_score(score=5.0, band="MINIMAL")
        rec = recommender.recommend_single(ev, rs)
        assert rec.maneuver_type in ("NO_ACTION", "MONITOR")
        assert rec.delta_v_ms == 0.0

    def test_action_for_high_pc(self, recommender):
        ev = make_event(pc=1e-2, miss=0.1, lead=12.0)
        rs = make_risk_score(score=85.0, band="SEVERE")
        rec = recommender.recommend_single(ev, rs)
        assert rec.maneuver_type not in ("NO_ACTION", "MONITOR")
        assert rec.delta_v_ms > 0

    def test_emergency_for_short_lead(self, recommender):
        ev = make_event(pc=1e-3, miss=0.5, lead=1.5)
        rs = make_risk_score(score=75.0, band="SEVERE")
        rec = recommender.recommend_single(ev, rs)
        assert rec.maneuver_type == "EMERGENCY"
        assert rec.urgent

    def test_recommendation_has_valid_type(self, recommender):
        for pc, miss, lead in [(1e-8, 9.0, 72.0), (1e-4, 1.0, 24.0),
                                (1e-2, 0.1, 6.0), (5e-3, 0.3, 2.0)]:
            ev  = make_event(pc=pc, miss=miss, lead=lead)
            rs  = RiskEngine().score_event(ev)
            rec = recommender.recommend_single(ev, rs)
            assert rec.maneuver_type in MANEUVER_TYPES, \
                f"Unknown maneuver type: {rec.maneuver_type}"

    def test_dv_vector_shape(self, recommender):
        ev  = make_event(pc=1e-3, miss=0.5, lead=12.0)
        rs  = make_risk_score(score=70.0, band="SEVERE")
        rec = recommender.recommend_single(ev, rs)
        assert rec.delta_v_vec.shape == (3,)

    def test_confidence_in_range(self, recommender):
        ev  = make_event(pc=1e-4, miss=1.0, lead=24.0)
        rs  = make_risk_score(score=50.0)
        rec = recommender.recommend_single(ev, rs)
        assert 0.0 <= rec.confidence <= 1.0

    def test_fuel_cost_non_negative(self, recommender):
        ev  = make_event(pc=1e-3, miss=0.5, lead=12.0)
        rs  = make_risk_score(score=70.0, band="SEVERE")
        rec = recommender.recommend_single(ev, rs)
        assert rec.fuel_cost_kg >= 0.0

    def test_new_pc_lower_than_original(self, recommender):
        """After maneuver, projected Pc should be lower."""
        ev  = make_event(pc=5e-4, miss=0.8, lead=10.0)
        rs  = make_risk_score(score=65.0, band="HIGH")
        rec = recommender.recommend_single(ev, rs)
        if rec.maneuver_type not in ("NO_ACTION", "MONITOR"):
            assert rec.new_pc_estimate <= ev.pc

    def test_recommend_list(self, recommender):
        events = [make_event(f"CDM-{i}", pc=1e-4, lead=24.0) for i in range(5)]
        scores = [make_risk_score(f"CDM-{i}", score=40.0) for i in range(5)]
        recs   = recommender.recommend(events, scores)
        assert len(recs) == 5

    def test_urgent_sorted_first(self, recommender):
        ev_urgent = make_event("CDM-U", pc=5e-3, miss=0.3, lead=1.0)
        ev_normal = make_event("CDM-N", pc=1e-4, miss=2.0, lead=36.0)
        engine    = RiskEngine()
        scores    = engine.score_events([ev_urgent, ev_normal])
        recs      = recommender.recommend([ev_urgent, ev_normal], scores)
        urgent_recs = [r for r in recs if r.urgent]
        if urgent_recs:
            urgent_idx  = next(i for i, r in enumerate(recs) if r.urgent)
            normal_idx  = next(i for i, r in enumerate(recs) if not r.urgent)
            assert urgent_idx < normal_idx, "Urgent recommendations should appear first"

    def test_to_dataframe(self, recommender):
        events = [make_event(f"CDM-{i}", pc=5e-4, miss=0.5, lead=10.0)
                  for i in range(3)]
        scores = [make_risk_score(f"CDM-{i}", score=65.0, band="HIGH")
                  for i in range(3)]
        recs   = recommender.recommend(events, scores)
        df     = recommender.to_dataframe(recs)
        assert len(df) == 3
        for col in ["event_id", "maneuver_type", "delta_v_ms", "fuel_cost_kg"]:
            assert col in df.columns

    def test_rationale_non_empty(self, recommender):
        ev  = make_event(pc=1e-4, miss=1.0, lead=24.0)
        rs  = make_risk_score(score=50.0)
        rec = recommender.recommend_single(ev, rs)
        assert len(rec.rationale) > 10
