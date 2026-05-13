"""
risk_engine.py
==============
Composite risk scoring engine for SpaceShield AI.

Combines multiple risk factors into a single normalised Risk Score [0–100]
using a weighted multi-criteria decision model.

Risk factors:
  - Collision probability (Pc)
  - Miss distance relative to threshold
  - Relative approach velocity (kinetic energy proxy)
  - Lead time to TCA (urgency)
  - Object mass (damage potential)
  - Radar cross-section (size of hazard)
  - Altitude region (congestion)
  - Asset priority (active satellite vs. debris)

Outputs:
  - risk_score  [0–100]  : Normalised composite risk
  - risk_grade  : A–F letter grade
  - risk_band   : MINIMAL / LOW / MODERATE / HIGH / SEVERE / CATASTROPHIC
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

from src.collision_prediction import ConjunctionEvent, CONJUNCTION_THRESHOLD_KM


# ─── Risk Bands ───────────────────────────────────────────────────────────────

RISK_BANDS = [
    (90, 100, "CATASTROPHIC", "F"),
    (70,  90, "SEVERE",       "E"),
    (50,  70, "HIGH",         "D"),
    (30,  50, "MODERATE",     "C"),
    (10,  30, "LOW",          "B"),
    ( 0,  10, "MINIMAL",      "A"),
]


def score_to_band(score: float) -> tuple:
    """Return (band_name, letter_grade) for a given score."""
    for lo, hi, band, grade in RISK_BANDS:
        if lo <= score <= hi:
            return band, grade
    return "CATASTROPHIC", "F"


# ─── Individual Factor Scorers ────────────────────────────────────────────────

def score_pc(pc: float) -> float:
    """
    Map collision probability to [0, 100].
    Uses log-linear scaling between 1e-8 and 1e-2.
    """
    if pc <= 0:
        return 0.0
    if pc >= 1e-2:
        return 100.0
    log_pc = np.log10(pc)
    # Map [-8, -2] → [0, 100]
    score = (log_pc - (-8.0)) / ((-2.0) - (-8.0)) * 100.0
    return float(np.clip(score, 0.0, 100.0))


def score_miss_distance(miss_km: float,
                        threshold_km: float = CONJUNCTION_THRESHOLD_KM) -> float:
    """
    Map miss distance to [0, 100].
    Very close approach → high score. Scales quadratically.
    """
    if miss_km <= 0:
        return 100.0
    if miss_km >= threshold_km:
        return 0.0
    ratio = 1.0 - (miss_km / threshold_km)
    return float(np.clip(ratio**2 * 100.0, 0.0, 100.0))


def score_velocity(rel_vel_kms: float) -> float:
    """
    Map relative velocity to [0, 100].
    High velocity → high kinetic energy → high risk.
    Typical LEO range: 0–15 km/s.
    """
    return float(np.clip((rel_vel_kms / 15.0) ** 0.7 * 100.0, 0.0, 100.0))


def score_lead_time(lead_hours: float) -> float:
    """
    Map lead time to [0, 100].
    Very short lead time → insufficient reaction window → high score.
    24 h is considered minimum comfortable; 72 h is nominal; < 4 h is critical.
    """
    if lead_hours <= 0.5:
        return 100.0
    if lead_hours >= 72.0:
        return 0.0
    # Exponential urgency: high score for short lead times
    return float(np.clip(100.0 * np.exp(-lead_hours / 10.0), 0.0, 100.0))


def score_mass(combined_mass_kg: float) -> float:
    """
    Map combined mass to [0, 100].
    Heavier objects cause more damage on collision.
    Reference: ~10,000 kg upper bound for tracked objects.
    """
    return float(np.clip(np.log10(max(combined_mass_kg, 0.1)) / 4.0 * 100.0, 0.0, 100.0))


def score_rcs(combined_rcs_m2: float) -> float:
    """
    Map combined RCS to [0, 100].
    Larger cross-section → higher collision probability.
    """
    return float(np.clip(np.log10(max(combined_rcs_m2, 0.001)) / 2.0 * 100.0, 0.0, 100.0))


def score_altitude(altitude_km: float) -> float:
    """
    Map altitude to congestion score [0, 100].
    Most congested bands: 550–650 km (Starlink), 400–500 km (ISS),
    800–1000 km (high-inclination), 1200 km (Van Allen).
    """
    # Hotspot bands with peak congestion
    hotspots = [450, 560, 630, 780, 900, 1200]
    sigma = 60.0  # km half-width

    score = 0.0
    for h in hotspots:
        score += np.exp(-0.5 * ((altitude_km - h) / sigma)**2)

    return float(np.clip(score / len(hotspots) * 200.0, 0.0, 100.0))


def score_asset_priority(primary_active: bool) -> float:
    """
    Penalise risk when the primary is an operational satellite (protectable asset).
    Active asset threatened → 80 points; debris-on-debris → 20 points.
    """
    return 80.0 if primary_active else 20.0


# ─── Weights ──────────────────────────────────────────────────────────────────

RISK_WEIGHTS = {
    "pc":             0.35,   # Dominant factor
    "miss_distance":  0.20,
    "velocity":       0.10,
    "lead_time":      0.15,
    "mass":           0.08,
    "rcs":            0.05,
    "altitude":       0.04,
    "asset_priority": 0.03,
}

assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"


# ─── Risk Score Record ────────────────────────────────────────────────────────

@dataclass
class RiskScore:
    """Composite risk score for a single conjunction event."""
    event_id:       str
    primary_id:     str
    secondary_id:   str
    total_score:    float          # [0, 100]
    risk_band:      str
    risk_grade:     str
    component_scores: dict         # Individual factor scores
    threat_level:   str            # From ML classifier (if available)

    def to_dict(self) -> dict:
        d = {
            "event_id":    self.event_id,
            "primary_id":  self.primary_id,
            "secondary_id":self.secondary_id,
            "risk_score":  round(self.total_score, 2),
            "risk_band":   self.risk_band,
            "risk_grade":  self.risk_grade,
            "threat_level":self.threat_level,
        }
        d.update({f"score_{k}": round(v, 2)
                  for k, v in self.component_scores.items()})
        return d


# ─── Risk Engine ──────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Computes composite risk scores for conjunction events.

    Usage
    -----
    >>> engine = RiskEngine()
    >>> scores = engine.score_events(events, active_ids=active_ids)
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or RISK_WEIGHTS.copy()
        # Normalise
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score_event(self,
                    event: ConjunctionEvent,
                    primary_active: bool = False,
                    threat_level: str = "UNKNOWN") -> RiskScore:
        """
        Compute composite risk score for a single conjunction event.

        Parameters
        ----------
        event          : ConjunctionEvent
        primary_active : Is the primary object an active satellite?
        threat_level   : ML classifier output (optional, for reporting)

        Returns
        -------
        RiskScore
        """
        components = {
            "pc":             score_pc(event.pc),
            "miss_distance":  score_miss_distance(event.miss_distance_km),
            "velocity":       score_velocity(event.rel_velocity_kms),
            "lead_time":      score_lead_time(event.lead_time_hours),
            "mass":           score_mass(event.combined_mass_kg),
            "rcs":            score_rcs(event.combined_rcs_m2),
            "altitude":       score_altitude(event.primary_altitude),
            "asset_priority": score_asset_priority(primary_active),
        }

        total = sum(self.weights[k] * v for k, v in components.items())
        total = float(np.clip(total, 0.0, 100.0))
        band, grade = score_to_band(total)

        return RiskScore(
            event_id=event.event_id,
            primary_id=event.primary_id,
            secondary_id=event.secondary_id,
            total_score=total,
            risk_band=band,
            risk_grade=grade,
            component_scores=components,
            threat_level=threat_level,
        )

    def score_events(self,
                     events: List[ConjunctionEvent],
                     active_ids: set = None,
                     threat_labels: list = None) -> List[RiskScore]:
        """
        Score a list of conjunction events.

        Parameters
        ----------
        events        : List of ConjunctionEvent
        active_ids    : Set of object_ids that are active satellites
        threat_labels : Optional list of ML threat labels (same order as events)

        Returns
        -------
        List of RiskScore, sorted by total_score descending
        """
        if active_ids is None:
            active_ids = set()
        if threat_labels is None:
            threat_labels = ["UNKNOWN"] * len(events)

        scores = [
            self.score_event(
                ev,
                primary_active=(ev.primary_id in active_ids),
                threat_level=tl
            )
            for ev, tl in zip(events, threat_labels)
        ]

        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores

    def to_dataframe(self, scores: List[RiskScore]) -> pd.DataFrame:
        """Convert list of RiskScore to DataFrame."""
        if not scores:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in scores])

    def summary_stats(self, scores: List[RiskScore]) -> dict:
        """Aggregate risk statistics across all scored events."""
        if not scores:
            return {}
        vals = [s.total_score for s in scores]
        band_counts: Dict[str, int] = {}
        for s in scores:
            band_counts[s.risk_band] = band_counts.get(s.risk_band, 0) + 1

        return {
            "n_events":    len(scores),
            "mean_score":  round(float(np.mean(vals)), 2),
            "max_score":   round(float(np.max(vals)),  2),
            "min_score":   round(float(np.min(vals)),  2),
            "std_score":   round(float(np.std(vals)),  2),
            "band_distribution": band_counts,
        }

    def save(self, scores: List[RiskScore],
             filepath: str = "results/risk_scores.csv") -> pd.DataFrame:
        """Save risk scores to CSV."""
        df = self.to_dataframe(scores)
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"[RiskEngine] Scores saved → {filepath}  ({len(df)} records)")
        return df


if __name__ == "__main__":
    # Minimal self-test
    from src.debris_generator import DebrisGenerator
    from src.collision_prediction import ConjunctionScreener

    gen = DebrisGenerator(seed=99)
    objs = gen.generate(n_active=10, n_debris=15, n_rockets=5, n_defunct=5)
    active_ids = {o.object_id for o in objs if o.active}

    screener = ConjunctionScreener(screening_threshold_km=20.0)
    events = screener.screen(objs, duration_hours=6.0, verbose=False)

    if events:
        engine = RiskEngine()
        scores = engine.score_events(events, active_ids=active_ids)
        stats  = engine.summary_stats(scores)
        print(f"\nRisk summary: {stats}")
        print(f"Top event: {scores[0].event_id}  score={scores[0].total_score:.1f}  "
              f"band={scores[0].risk_band}")
    else:
        print("No conjunction events detected in test run.")

    print("Risk engine module: OK ✓")
