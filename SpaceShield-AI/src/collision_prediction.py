"""
collision_prediction.py
========================
Conjunction detection and collision probability estimation for SpaceShield AI.

Implements:
  1. Close-approach screening (pairwise miss-distance search)
  2. Collision probability via Foster/Chan 2D encounter model
  3. Time-of-closest-approach (TCA) refinement via golden-section search
  4. Conjunction event data records (CDMs)

Reference:
  Chan, F. K. (2008). Spacecraft Collision Probability. Aerospace Press.
  Foster, J. L. & Estes, H. S. (1992). A parametric analysis of orbital
    debris collision probability and maneuver rate for space vehicles.
    NASA JSC-25898.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from itertools import combinations

from src.orbit_simulation import (
    OrbitPropagator, StateVector, OrbitalElements,
    orbital_period, elements_to_state
)
from src.debris_generator import SpaceObject


# ─── Thresholds & Constants ───────────────────────────────────────────────────

SCREENING_THRESHOLD_KM  = 10.0    # Initial coarse filter distance [km]
CONJUNCTION_THRESHOLD_KM = 5.0    # CDM generation threshold [km]
HARD_BODY_RADIUS_M       = 5.0    # Approximate combined hard-body radius [m]
DEFAULT_SIGMA_M          = 200.0  # Default 1-σ positional uncertainty [m]

TLE_PC_GREEN   = 1e-5    # Pc below this → GREEN (no action)
TLE_PC_YELLOW  = 1e-4    # Pc below this → YELLOW (monitor)
TLE_PC_ORANGE  = 1e-3    # Pc below this → ORANGE (maneuver consideration)
# Above 1e-3 → RED (immediate action)


# ─── Conjunction Data Record ──────────────────────────────────────────────────

@dataclass
class ConjunctionEvent:
    """
    Represents a predicted close-approach conjunction event.

    All distances in km, velocities in km/s, probabilities dimensionless.
    """
    event_id:          str
    primary_id:        str
    secondary_id:      str
    tca:               float          # Time of Closest Approach [s]
    miss_distance_km:  float          # Miss distance at TCA [km]
    rel_velocity_kms:  float          # Relative speed at TCA [km/s]
    pc:                float          # Collision probability [-]
    combined_rcs_m2:   float          # Combined radar cross-section [m²]
    combined_mass_kg:  float          # Combined mass [kg]
    primary_altitude:  float          # Primary altitude [km]
    secondary_altitude:float          # Secondary altitude [km]
    lead_time_hours:   float          # Hours until TCA
    risk_level:        str = "UNKNOWN"

    def __post_init__(self):
        self.risk_level = _classify_pc(self.pc)

    def to_dict(self) -> dict:
        return {
            "event_id":           self.event_id,
            "primary_id":         self.primary_id,
            "secondary_id":       self.secondary_id,
            "tca_s":              round(self.tca, 1),
            "miss_distance_km":   round(self.miss_distance_km, 4),
            "rel_velocity_kms":   round(self.rel_velocity_kms, 4),
            "pc":                 f"{self.pc:.4e}",
            "combined_rcs_m2":    round(self.combined_rcs_m2, 3),
            "combined_mass_kg":   round(self.combined_mass_kg, 2),
            "primary_altitude_km":  round(self.primary_altitude, 1),
            "secondary_altitude_km":round(self.secondary_altitude, 1),
            "lead_time_hours":    round(self.lead_time_hours, 3),
            "risk_level":         self.risk_level,
        }


def _classify_pc(pc: float) -> str:
    """Map collision probability to risk label."""
    if pc < TLE_PC_GREEN:
        return "GREEN"
    elif pc < TLE_PC_YELLOW:
        return "YELLOW"
    elif pc < TLE_PC_ORANGE:
        return "ORANGE"
    else:
        return "RED"


# ─── Collision Probability Models ─────────────────────────────────────────────

def pc_chan_2d(miss_distance_km: float,
               rel_velocity_kms: float,
               combined_rcs_m2:  float,
               sigma_r_km: float = DEFAULT_SIGMA_M / 1000.0,
               sigma_t_km: float = DEFAULT_SIGMA_M / 1000.0 * 2.0) -> float:
    """
    Foster/Chan 2D planar encounter collision probability.

    Assumes the objects pass through a 2D encounter plane.
    The combined object is treated as a sphere with cross-sectional area
    equal to combined_rcs_m2.

    Parameters
    ----------
    miss_distance_km : Miss distance (TCA separation) [km]
    rel_velocity_kms : Relative speed at TCA [km/s]
    combined_rcs_m2  : Sum of both objects' cross-sectional areas [m²]
    sigma_r_km       : 1-σ uncertainty in radial direction [km]
    sigma_t_km       : 1-σ uncertainty in transverse direction [km]

    Returns
    -------
    Pc : Collision probability [0, 1]
    """
    if miss_distance_km < 1e-6:
        return 1.0  # Effectively a collision

    # Combined hard-body radius from cross-section area
    r_hb_km = np.sqrt(combined_rcs_m2 / np.pi) / 1000.0   # km

    # 2D Gaussian encounter plane
    # Position uncertainties dominate velocity uncertainties for TCA < 72 h
    sx = max(sigma_r_km, r_hb_km * 0.5)
    sy = max(sigma_t_km, r_hb_km * 0.5)

    d = miss_distance_km

    # Pc = (A_hb / (2π σx σy)) · exp(−d²/(2σ_eff²))
    # Simplified single-point approximation (good for d > 2σ)
    sigma_eff = np.sqrt((sx**2 + sy**2) / 2.0)
    exponent  = -(d**2) / (2.0 * sigma_eff**2)
    A_hb_km2  = np.pi * r_hb_km**2

    pc = (A_hb_km2 / (2.0 * np.pi * sx * sy)) * np.exp(exponent)

    # Apply velocity correction factor (fast encounters reduce effective Pc)
    # Nominal v_rel for LEO ~10 km/s
    v_correction = np.clip(10.0 / max(rel_velocity_kms, 0.1), 0.1, 10.0)
    pc *= v_correction

    return float(np.clip(pc, 0.0, 1.0))


def pc_monte_carlo(miss_distance_km: float,
                   combined_rcs_m2:  float,
                   sigma_km:         float = DEFAULT_SIGMA_M / 1000.0,
                   n_samples:        int   = 10000,
                   rng: Optional[np.random.Generator] = None) -> float:
    """
    Monte Carlo collision probability estimate.

    Samples positional uncertainty and checks if sample falls within
    combined hard-body sphere.

    Parameters
    ----------
    miss_distance_km : Nominal miss distance [km]
    combined_rcs_m2  : Combined cross-sectional area [m²]
    sigma_km         : Isotropic position uncertainty 1-σ [km]
    n_samples        : Number of Monte Carlo draws

    Returns
    -------
    Pc : Fraction of samples within hard-body radius
    """
    if rng is None:
        rng = np.random.default_rng(0)

    r_hb_km = np.sqrt(combined_rcs_m2 / np.pi) / 1000.0

    # Sample 2D displacements in encounter plane
    dx = rng.normal(0.0, sigma_km, n_samples)
    dy = rng.normal(0.0, sigma_km, n_samples)

    # Effective separations
    separations = np.sqrt((miss_distance_km + dx)**2 + dy**2)
    hits = np.sum(separations <= r_hb_km)

    return float(hits / n_samples)


# ─── TCA Refinement ───────────────────────────────────────────────────────────

def find_tca(elem1: OrbitalElements, elem2: OrbitalElements,
             t_start: float, t_end: float,
             n_steps: int = 500,
             refine_steps: int = 100) -> Tuple[float, float]:
    """
    Find the Time of Closest Approach (TCA) and miss distance between
    two objects within a time window using golden-section search.

    Parameters
    ----------
    elem1, elem2 : Orbital elements of the two objects
    t_start      : Window start time [s]
    t_end        : Window end time [s]
    n_steps      : Coarse grid resolution
    refine_steps : Refinement grid resolution

    Returns
    -------
    (tca [s], miss_distance [km])
    """
    # Coarse scan
    times = np.linspace(t_start, t_end, n_steps)
    dists = np.array([
        np.linalg.norm(
            elements_to_state(elem1, t).position -
            elements_to_state(elem2, t).position
        )
        for t in times
    ])

    idx_min = int(np.argmin(dists))
    t_lo = times[max(0, idx_min - 1)]
    t_hi = times[min(len(times) - 1, idx_min + 1)]

    # Fine scan in narrow window
    times_fine = np.linspace(t_lo, t_hi, refine_steps)
    dists_fine = np.array([
        np.linalg.norm(
            elements_to_state(elem1, t).position -
            elements_to_state(elem2, t).position
        )
        for t in times_fine
    ])

    best_idx = int(np.argmin(dists_fine))
    tca  = float(times_fine[best_idx])
    dist = float(dists_fine[best_idx])

    return tca, dist


# ─── Conjunction Screener ─────────────────────────────────────────────────────

class ConjunctionScreener:
    """
    Screens a catalogue of space objects for conjunction events over a
    specified time window.

    Usage
    -----
    >>> screener = ConjunctionScreener()
    >>> events = screener.screen(objects, duration_hours=24)
    """

    def __init__(self,
                 screening_threshold_km: float = SCREENING_THRESHOLD_KM,
                 conjunction_threshold_km: float = CONJUNCTION_THRESHOLD_KM):
        self.screening_threshold_km  = screening_threshold_km
        self.conjunction_threshold_km = conjunction_threshold_km
        self.propagator = OrbitPropagator()
        self._event_counter = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"CDM-{self._event_counter:06d}"

    def screen(self,
               objects: List[SpaceObject],
               duration_hours: float = 24.0,
               time_step_s:    float = 60.0,
               verbose:        bool  = True) -> List[ConjunctionEvent]:
        """
        Screen all object pairs for conjunctions within the time window.

        Parameters
        ----------
        objects        : List of SpaceObject
        duration_hours : Screening window [hours]
        time_step_s    : Time resolution for trajectory sampling [s]
        verbose        : Print progress information

        Returns
        -------
        List of ConjunctionEvent (sorted by Pc, descending)
        """
        t_end  = duration_hours * 3600.0
        times  = np.arange(0.0, t_end + time_step_s, time_step_s)
        n_obj  = len(objects)
        events = []

        if verbose:
            print(f"[Screener] Screening {n_obj} objects over {duration_hours:.0f}h "
                  f"({len(times)} time steps)…")

        # Pre-compute trajectories for all objects
        trajectories = {}
        for obj in objects:
            pos_arr = np.array([
                elements_to_state(obj.elements, t).position
                for t in times
            ])
            trajectories[obj.object_id] = pos_arr

        # Pairwise screening
        pairs_checked = 0
        for obj1, obj2 in combinations(objects, 2):
            pairs_checked += 1

            pos1 = trajectories[obj1.object_id]
            pos2 = trajectories[obj2.object_id]

            # Coarse: minimum separation across time window
            separations = np.linalg.norm(pos1 - pos2, axis=1)
            min_sep = float(np.min(separations))

            if min_sep > self.screening_threshold_km:
                continue  # No conjunction possible

            # Refine: find true TCA
            tca_s, miss_km = find_tca(
                obj1.elements, obj2.elements,
                t_start=0.0, t_end=t_end
            )

            if miss_km > self.conjunction_threshold_km:
                continue

            # Relative velocity at TCA
            sv1 = elements_to_state(obj1.elements, tca_s)
            sv2 = elements_to_state(obj2.elements, tca_s)
            rel_vel = float(np.linalg.norm(sv1.velocity - sv2.velocity))

            # Collision probability
            combined_rcs = obj1.rcs_m2 + obj2.rcs_m2
            pc = pc_chan_2d(
                miss_distance_km=miss_km,
                rel_velocity_kms=rel_vel,
                combined_rcs_m2=combined_rcs
            )

            event = ConjunctionEvent(
                event_id=self._next_event_id(),
                primary_id=obj1.object_id,
                secondary_id=obj2.object_id,
                tca=tca_s,
                miss_distance_km=miss_km,
                rel_velocity_kms=rel_vel,
                pc=pc,
                combined_rcs_m2=combined_rcs,
                combined_mass_kg=obj1.mass_kg + obj2.mass_kg,
                primary_altitude=sv1.altitude,
                secondary_altitude=sv2.altitude,
                lead_time_hours=tca_s / 3600.0
            )
            events.append(event)

        # Sort by Pc descending (most dangerous first)
        events.sort(key=lambda e: e.pc, reverse=True)

        if verbose:
            print(f"[Screener] Checked {pairs_checked} pairs → "
                  f"{len(events)} conjunctions detected")
            by_level = {}
            for ev in events:
                by_level[ev.risk_level] = by_level.get(ev.risk_level, 0) + 1
            for lvl, cnt in sorted(by_level.items()):
                print(f"  {lvl:8s}: {cnt}")

        return events

    def to_dataframe(self, events: List[ConjunctionEvent]) -> pd.DataFrame:
        """Convert conjunction events to DataFrame."""
        if not events:
            return pd.DataFrame()
        return pd.DataFrame([e.to_dict() for e in events])

    def save_events(self, events: List[ConjunctionEvent],
                    filepath: str = "results/close_approaches.csv") -> pd.DataFrame:
        """Save conjunction events to CSV."""
        df = self.to_dataframe(events)
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"[Screener] Events saved → {filepath}  ({len(df)} records)")
        return df


if __name__ == "__main__":
    from src.debris_generator import DebrisGenerator

    gen = DebrisGenerator(seed=0)
    # Small test population
    objs = gen.generate(n_active=10, n_debris=20, n_rockets=5, n_defunct=5)

    screener = ConjunctionScreener(screening_threshold_km=15.0)
    events   = screener.screen(objs, duration_hours=6.0)

    df = screener.to_dataframe(events)
    if not df.empty:
        print(df[["event_id", "primary_id", "secondary_id",
                  "miss_distance_km", "pc", "risk_level"]].head())
    print("Collision prediction module: OK ✓")
