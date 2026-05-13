"""
maneuver_recommender.py
========================
Autonomous maneuver recommendation system for SpaceShield AI.

Given a conjunction event and risk score, computes the minimum delta-v
maneuver required to reduce collision probability below the green threshold,
and selects the optimal maneuver strategy:

  1. In-track Boost        – Increase along-track velocity (phase shift)
  2. In-track Brake        – Decrease along-track velocity
  3. Radial Raise          – Increase orbital altitude
  4. Radial Lower          – Decrease orbital altitude
  5. Combined Maneuver     – Two-burn out-of-plane (avoid dense region)
  6. No Action Required    – Risk below threshold

Reference:
  Alfano, S. (2005). Relating Position Uncertainty to Maximum Conjunction
    Probability. Journal of the Astronautical Sciences, 53(2).
  Bombardelli, C. & Hernando-Ayuso, J. (2015). Optimal Impulsive Collision
    Avoidance in Low Earth Orbit. JGCD, 38(8).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.collision_prediction import ConjunctionEvent, TLE_PC_GREEN, TLE_PC_YELLOW
from src.risk_engine import RiskScore


# ─── Maneuver Types ───────────────────────────────────────────────────────────

MANEUVER_TYPES = {
    "NO_ACTION":        "No action required — Pc below threshold",
    "MONITOR":          "Monitor only — re-evaluate at TCA-24h",
    "IN_TRACK_BOOST":   "In-track boost: increase along-track velocity",
    "IN_TRACK_BRAKE":   "In-track brake: decrease along-track velocity",
    "RADIAL_RAISE":     "Radial raise: increase orbital altitude",
    "RADIAL_LOWER":     "Radial lower: decrease orbital altitude",
    "COMBINED":         "Combined two-burn maneuver for maximum separation",
    "EMERGENCY":        "Emergency evasive burn — immediate execution required",
}

# Fuel efficiency preference order (lowest Δv cost first)
MANEUVER_PRIORITY = [
    "IN_TRACK_BOOST",
    "IN_TRACK_BRAKE",
    "RADIAL_RAISE",
    "RADIAL_LOWER",
    "COMBINED",
    "EMERGENCY",
]


# ─── Maneuver Recommendation Record ──────────────────────────────────────────

@dataclass
class ManeuverRecommendation:
    """
    Complete maneuver recommendation for a conjunction event.

    Attributes
    ----------
    event_id         : Conjunction event identifier
    primary_id       : Satellite to maneuver
    maneuver_type    : Type of maneuver recommended
    delta_v_ms       : Required Δv magnitude [m/s]
    delta_v_vec      : Δv vector in RTN frame [m/s]  (Radial, In-Track, Normal)
    execution_epoch  : Recommended burn time [s before TCA]
    fuel_cost_kg     : Estimated propellant mass [kg]  (Tsiolkovsky)
    new_pc_estimate  : Estimated Pc after maneuver
    reduction_factor : Pc reduction factor
    maneuver_window  : Execution window [hours before TCA]
    confidence       : Recommendation confidence [0–1]
    rationale        : Human-readable decision rationale
    urgent           : Whether maneuver must be executed within 4 hours
    """
    event_id:         str
    primary_id:       str
    maneuver_type:    str
    delta_v_ms:       float         # m/s
    delta_v_vec:      np.ndarray    # RTN [m/s]
    execution_epoch:  float         # s before TCA
    fuel_cost_kg:     float         # kg
    new_pc_estimate:  float
    reduction_factor: float
    maneuver_window:  Tuple[float, float]  # (earliest, latest) hours before TCA
    confidence:       float
    rationale:        str
    urgent:           bool = False

    def to_dict(self) -> dict:
        return {
            "event_id":         self.event_id,
            "primary_id":       self.primary_id,
            "maneuver_type":    self.maneuver_type,
            "description":      MANEUVER_TYPES.get(self.maneuver_type, ""),
            "delta_v_ms":       round(self.delta_v_ms, 4),
            "dv_radial_ms":     round(float(self.delta_v_vec[0]), 4),
            "dv_intrack_ms":    round(float(self.delta_v_vec[1]), 4),
            "dv_normal_ms":     round(float(self.delta_v_vec[2]), 4),
            "execution_before_tca_s": round(self.execution_epoch, 1),
            "fuel_cost_kg":     round(self.fuel_cost_kg, 4),
            "new_pc":           f"{self.new_pc_estimate:.4e}",
            "reduction_factor": round(self.reduction_factor, 2),
            "window_earliest_h":round(self.maneuver_window[0], 2),
            "window_latest_h":  round(self.maneuver_window[1], 2),
            "confidence":       round(self.confidence, 3),
            "urgent":           self.urgent,
            "rationale":        self.rationale,
        }


# ─── Physics Helpers ──────────────────────────────────────────────────────────

def tsiolkovsky_fuel(dv_ms: float, m_wet_kg: float,
                     isp_s: float = 220.0) -> float:
    """
    Tsiolkovsky rocket equation: fuel mass for a given Δv.

    Parameters
    ----------
    dv_ms    : Required Δv [m/s]
    m_wet_kg : Spacecraft wet mass [kg]
    isp_s    : Specific impulse [s]  (default: hydrazine monoprop)

    Returns
    -------
    Fuel mass [kg]
    """
    g0 = 9.80665  # m/s²
    m_dry = m_wet_kg / np.exp(dv_ms / (isp_s * g0))
    return float(max(m_wet_kg - m_dry, 0.0))


def minimum_dv_for_pc_reduction(miss_km: float, rel_vel_kms: float,
                                 combined_rcs_m2: float,
                                 target_pc: float,
                                 sigma_km: float = 0.2) -> float:
    """
    Estimate the minimum Δv [m/s] needed to reduce Pc to target_pc.

    Uses an analytical approximation based on the 2D encounter model:
    Increasing miss distance requires shifting the orbital phase by ~Δv·T
    where T is the time to TCA.

    Returns Δv in m/s.
    """
    from src.collision_prediction import pc_chan_2d

    # Binary search for required miss distance
    target_miss_lo = miss_km
    target_miss_hi = 100.0   # 100 km should always be safe

    for _ in range(50):
        target_miss_mid = (target_miss_lo + target_miss_hi) / 2.0
        pc_test = pc_chan_2d(target_miss_mid, rel_vel_kms, combined_rcs_m2)
        if pc_test <= target_pc:
            target_miss_hi = target_miss_mid
        else:
            target_miss_lo = target_miss_mid

    required_miss = (target_miss_lo + target_miss_hi) / 2.0
    delta_miss_km = max(required_miss - miss_km, 0.0)

    # Δv to achieve delta_miss: in-track burn shifts orbit by ~Δv·P/(2π)
    # where P is ~90 min for LEO → very roughly Δv ≈ delta_miss/T * v_orb/2
    # Use empirical scaling: ~0.5 m/s per km of miss distance increase
    dv_per_km = 0.5   # m/s per km (conservative estimate)
    dv_ms = delta_miss_km * dv_per_km

    return float(max(dv_ms, 0.01))   # Minimum 1 cm/s


def _select_maneuver_type(event: ConjunctionEvent, risk_score: RiskScore,
                          lead_hours: float) -> str:
    """Rule-based selection of maneuver strategy."""
    if event.pc < TLE_PC_GREEN:
        return "NO_ACTION"
    if event.pc < TLE_PC_YELLOW and risk_score.total_score < 30:
        return "MONITOR"

    if lead_hours < 4.0:
        return "EMERGENCY"
    elif risk_score.total_score >= 70:
        return "COMBINED"
    else:
        # Prefer in-track for fuel efficiency
        # Boost if object is approaching from behind; brake otherwise
        # Without full geometry: use a heuristic based on relative velocity
        return "IN_TRACK_BOOST" if risk_score.total_score < 50 else "RADIAL_RAISE"


def _compute_dv_vector(maneuver_type: str, dv_ms: float) -> np.ndarray:
    """
    Compute Δv vector in RTN frame for the given maneuver type.
    RTN: Radial (outward), In-Track (velocity direction), Normal (h×r).
    """
    vectors = {
        "IN_TRACK_BOOST":  np.array([0.0,    dv_ms,  0.0]),
        "IN_TRACK_BRAKE":  np.array([0.0,   -dv_ms,  0.0]),
        "RADIAL_RAISE":    np.array([dv_ms,  0.0,    0.0]),
        "RADIAL_LOWER":    np.array([-dv_ms, 0.0,    0.0]),
        "COMBINED":        np.array([dv_ms * 0.5, dv_ms * 0.5 * np.sqrt(2.0), dv_ms * 0.3]),
        "EMERGENCY":       np.array([0.0,    dv_ms,  dv_ms * 0.2]),
        "NO_ACTION":       np.array([0.0,    0.0,    0.0]),
        "MONITOR":         np.array([0.0,    0.0,    0.0]),
    }
    vec = vectors.get(maneuver_type, np.array([0.0, dv_ms, 0.0]))
    # Normalise to correct total magnitude
    mag = float(np.linalg.norm(vec))
    if mag > 0 and maneuver_type not in ("NO_ACTION", "MONITOR"):
        vec = vec / mag * dv_ms
    return vec


def _execution_window(lead_hours: float, maneuver_type: str) -> Tuple[float, float]:
    """Return (earliest, latest) hours before TCA for maneuver execution."""
    if maneuver_type in ("NO_ACTION", "MONITOR"):
        return (0.0, 0.0)
    if maneuver_type == "EMERGENCY":
        return (max(0.1, lead_hours - 0.5), lead_hours)
    # Optimal execution: 24–48 h before TCA for in-track maneuvers
    earliest = min(48.0, lead_hours)
    latest   = max(4.0, lead_hours * 0.3)
    return (latest, earliest)


def _build_rationale(event: ConjunctionEvent, risk_score: RiskScore,
                     maneuver_type: str, dv_ms: float,
                     new_pc: float) -> str:
    """Generate a human-readable rationale string."""
    lines = [
        f"Conjunction {event.event_id}: {event.primary_id} ↔ {event.secondary_id}",
        f"Current Pc={event.pc:.2e}  Miss={event.miss_distance_km:.3f}km  "
        f"TCA in {event.lead_time_hours:.1f}h",
        f"Risk score={risk_score.total_score:.1f}/100  Band={risk_score.risk_band}",
        f"Recommended: {MANEUVER_TYPES.get(maneuver_type, maneuver_type)}",
        f"Required Δv={dv_ms:.3f} m/s  → projected Pc={new_pc:.2e}",
    ]
    if maneuver_type == "NO_ACTION":
        lines.append("Action: None required at this time.")
    elif maneuver_type == "MONITOR":
        lines.append("Action: Continue tracking. Re-assess at TCA-24h.")
    elif maneuver_type == "EMERGENCY":
        lines.append("⚠ ACTION REQUIRED IMMEDIATELY — execute burn ASAP.")
    else:
        lines.append(f"Action: Execute {maneuver_type} maneuver within window.")
    return " | ".join(lines)


# ─── Recommender ─────────────────────────────────────────────────────────────

class ManeuverRecommender:
    """
    Generates autonomous maneuver recommendations for conjunction events.

    Usage
    -----
    >>> recommender = ManeuverRecommender()
    >>> recs = recommender.recommend(events, risk_scores, active_objects)
    """

    def __init__(self,
                 target_pc: float = TLE_PC_GREEN,
                 default_mass_kg: float = 300.0,
                 isp_s: float = 220.0):
        self.target_pc       = target_pc
        self.default_mass_kg = default_mass_kg
        self.isp_s           = isp_s

    def recommend_single(self,
                         event: ConjunctionEvent,
                         risk_score: RiskScore,
                         satellite_mass_kg: float = None) -> ManeuverRecommendation:
        """Generate a maneuver recommendation for a single event."""
        mass = satellite_mass_kg or self.default_mass_kg

        maneuver_type = _select_maneuver_type(event, risk_score,
                                               event.lead_time_hours)

        if maneuver_type in ("NO_ACTION", "MONITOR"):
            dv_ms = 0.0
            new_pc = event.pc
            reduction = 1.0
            confidence = 0.95
        else:
            dv_ms = minimum_dv_for_pc_reduction(
                miss_km=event.miss_distance_km,
                rel_vel_kms=event.rel_velocity_kms,
                combined_rcs_m2=event.combined_rcs_m2,
                target_pc=self.target_pc
            )
            # Add safety margin (10–30%)
            safety = 1.15 if risk_score.total_score < 60 else 1.30
            dv_ms *= safety

            # Estimate post-maneuver Pc (rough: Pc decreases as exp of dv)
            from src.collision_prediction import pc_chan_2d
            new_miss = event.miss_distance_km + dv_ms * 2.0  # ~2 km per m/s
            new_pc = pc_chan_2d(new_miss, event.rel_velocity_kms, event.combined_rcs_m2)
            reduction = event.pc / max(new_pc, 1e-15)
            confidence = min(0.95, 0.70 + 0.01 * min(event.lead_time_hours, 24))

        dv_vec   = _compute_dv_vector(maneuver_type, dv_ms)
        fuel_kg  = tsiolkovsky_fuel(dv_ms, mass, self.isp_s)
        window   = _execution_window(event.lead_time_hours, maneuver_type)
        rationale = _build_rationale(event, risk_score, maneuver_type,
                                      dv_ms, new_pc)

        # Execution epoch: midpoint of optimal window
        exec_s_before_tca = (window[0] + window[1]) / 2.0 * 3600.0 if dv_ms > 0 else 0.0

        return ManeuverRecommendation(
            event_id=event.event_id,
            primary_id=event.primary_id,
            maneuver_type=maneuver_type,
            delta_v_ms=dv_ms,
            delta_v_vec=dv_vec,
            execution_epoch=exec_s_before_tca,
            fuel_cost_kg=fuel_kg,
            new_pc_estimate=new_pc,
            reduction_factor=reduction,
            maneuver_window=window,
            confidence=confidence,
            rationale=rationale,
            urgent=(event.lead_time_hours < 4.0 and maneuver_type not in
                    ("NO_ACTION", "MONITOR"))
        )

    def recommend(self,
                  events: List[ConjunctionEvent],
                  risk_scores: List[RiskScore],
                  object_masses: dict = None) -> List[ManeuverRecommendation]:
        """
        Generate recommendations for a list of conjunction events.

        Parameters
        ----------
        events       : List of ConjunctionEvent
        risk_scores  : Corresponding RiskScore objects (same order)
        object_masses: dict mapping object_id → mass_kg

        Returns
        -------
        List of ManeuverRecommendation, sorted by urgency then risk score
        """
        if object_masses is None:
            object_masses = {}

        score_map = {s.event_id: s for s in risk_scores}

        recs = []
        for event in events:
            rs = score_map.get(event.event_id)
            if rs is None:
                continue
            mass = object_masses.get(event.primary_id, self.default_mass_kg)
            rec = self.recommend_single(event, rs, satellite_mass_kg=mass)
            recs.append(rec)

        # Sort: urgent first, then by risk score (descending)
        recs.sort(key=lambda r: (-int(r.urgent),
                                  -score_map.get(r.event_id, RiskScore(
                                      "", "", "", 0, "", "", {}, "")).total_score))
        return recs

    def to_dataframe(self, recs: List[ManeuverRecommendation]) -> pd.DataFrame:
        if not recs:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in recs])

    def save(self, recs: List[ManeuverRecommendation],
             filepath: str = "results/maneuver_recommendations.csv") -> pd.DataFrame:
        df = self.to_dataframe(recs)
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"[ManeuverRecommender] Saved → {filepath}  ({len(df)} recs)")
        return df

    def print_summary(self, recs: List[ManeuverRecommendation]) -> None:
        """Print a human-readable recommendation summary."""
        print("\n" + "=" * 70)
        print("  MANEUVER RECOMMENDATION SUMMARY")
        print("=" * 70)
        urgent = [r for r in recs if r.urgent]
        if urgent:
            print(f"  ⚠ URGENT MANEUVERS: {len(urgent)}")
        for rec in recs[:10]:   # Show top 10
            print(f"\n  [{rec.maneuver_type:16s}] Event {rec.event_id}")
            print(f"    Primary : {rec.primary_id}")
            print(f"    Δv      : {rec.delta_v_ms:.3f} m/s")
            print(f"    New Pc  : {rec.new_pc_estimate:.2e}")
            print(f"    Window  : {rec.maneuver_window[0]:.1f}–{rec.maneuver_window[1]:.1f} h before TCA")
            if rec.urgent:
                print("    *** URGENT — Execute immediately ***")
        print("=" * 70)


if __name__ == "__main__":
    from src.debris_generator import DebrisGenerator
    from src.collision_prediction import ConjunctionScreener
    from src.risk_engine import RiskEngine

    gen = DebrisGenerator(seed=7)
    objs = gen.generate(n_active=10, n_debris=20, n_rockets=5, n_defunct=5)
    active_ids = {o.object_id for o in objs if o.active}

    screener = ConjunctionScreener(screening_threshold_km=20.0)
    events   = screener.screen(objs, duration_hours=6.0, verbose=False)

    if events:
        engine = RiskEngine()
        scores = engine.score_events(events, active_ids=active_ids)

        rec_engine = ManeuverRecommender()
        recs = rec_engine.recommend(events, scores)
        rec_engine.print_summary(recs)
    else:
        print("No events — increase population or threshold for demo.")

    print("\nManeuver recommender module: OK ✓")
