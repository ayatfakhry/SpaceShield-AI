"""
debris_generator.py
====================
Synthetic space object population generator for SpaceShield AI.

Generates statistically realistic populations of:
  - Active satellites   (LEO / MEO / GEO / SSO)
  - Rocket bodies       (discarded upper stages)
  - Fragmentation debris (explosion / collision clouds)
  - Operational payloads

Distributions are calibrated to ESA MASTER 2009 / NASA ORDEM 3.1 statistics.

Units: km, radians, kg, m²
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.orbit_simulation import OrbitalElements, R_EARTH, sma_from_altitude

# ─── Random seed for reproducibility ─────────────────────────────────────────
DEFAULT_SEED = 42


# ─── Object Type Registry ─────────────────────────────────────────────────────

OBJECT_TYPES = {
    "ACTIVE_SATELLITE": {
        "alt_range":  (300, 1200),   # km
        "inc_modes":  [0.53, 0.97, 1.70],  # rad  (30°, 55°, 98°)
        "ecc_mean":   0.001,
        "mass_range": (2, 5000),     # kg
        "rcs_range":  (0.01, 20.0),  # m²  (radar cross-section)
        "color":      "#00FF88",
        "priority":   "HIGH",        # asset priority
    },
    "ROCKET_BODY": {
        "alt_range":  (400, 2000),
        "inc_modes":  [0.53, 0.87, 1.57],
        "ecc_mean":   0.003,
        "mass_range": (500, 9000),
        "rcs_range":  (0.5, 30.0),
        "color":      "#FF8800",
        "priority":   "MEDIUM",
    },
    "DEBRIS": {
        "alt_range":  (200, 2000),
        "inc_modes":  [0.53, 0.97, 1.65],
        "ecc_mean":   0.010,
        "mass_range": (0.001, 200),
        "rcs_range":  (0.001, 1.0),
        "color":      "#FF3333",
        "priority":   "LOW",
    },
    "DEFUNCT_SATELLITE": {
        "alt_range":  (500, 1800),
        "inc_modes":  [0.53, 0.87, 1.70],
        "ecc_mean":   0.005,
        "mass_range": (100, 3000),
        "rcs_range":  (0.05, 15.0),
        "color":      "#FFFF00",
        "priority":   "MEDIUM",
    },
}


# ─── Space Object Dataclass ───────────────────────────────────────────────────

@dataclass
class SpaceObject:
    """
    Represents a tracked space object (satellite, debris, rocket body).

    Attributes
    ----------
    object_id    : NORAD-style identifier string
    object_type  : One of OBJECT_TYPES keys
    elements     : Orbital elements
    mass_kg      : Mass [kg]
    rcs_m2       : Radar cross-section [m²]
    diameter_m   : Effective diameter [m]  (sphere-equivalent)
    active       : Whether object is an active, manoeuvrable satellite
    operator     : Organization responsible (None for debris)
    launch_year  : Approximate launch year
    """
    object_id:   str
    object_type: str
    elements:    OrbitalElements
    mass_kg:     float
    rcs_m2:      float
    diameter_m:  float
    active:      bool
    operator:    Optional[str] = None
    launch_year: Optional[int] = None

    @property
    def altitude_km(self) -> float:
        return self.elements.sma - R_EARTH

    @property
    def priority(self) -> str:
        return OBJECT_TYPES.get(self.object_type, {}).get("priority", "LOW")

    def to_dict(self) -> dict:
        return {
            "object_id":    self.object_id,
            "object_type":  self.object_type,
            "altitude_km":  round(self.altitude_km, 2),
            "sma_km":       round(self.elements.sma, 3),
            "eccentricity": round(self.elements.ecc, 6),
            "inclination_deg": round(np.degrees(self.elements.inc), 2),
            "raan_deg":     round(np.degrees(self.elements.raan), 2),
            "argp_deg":     round(np.degrees(self.elements.argp), 2),
            "mean_anomaly_deg": round(np.degrees(self.elements.mean_anomaly), 2),
            "mass_kg":      round(self.mass_kg, 3),
            "rcs_m2":       round(self.rcs_m2, 4),
            "diameter_m":   round(self.diameter_m, 3),
            "active":       self.active,
            "operator":     self.operator,
            "launch_year":  self.launch_year,
            "priority":     self.priority,
        }


# ─── Generator ────────────────────────────────────────────────────────────────

class DebrisGenerator:
    """
    Generates synthetic space object catalogues for LEO / MEO populations.

    Usage
    -----
    >>> gen = DebrisGenerator(seed=42)
    >>> objects = gen.generate(n_active=50, n_debris=150, n_rockets=20)
    >>> df = gen.to_dataframe(objects)
    """

    def __init__(self, seed: int = DEFAULT_SEED):
        self.rng = np.random.default_rng(seed)
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter:05d}"

    # ── Single object samplers ────────────────────────────────────────────────

    def _sample_inclination(self, modes: List[float]) -> float:
        """Sample inclination from a mixture of Gaussian modes [rad]."""
        mode = self.rng.choice(modes)
        return float(np.clip(self.rng.normal(mode, 0.15), 0.0, np.pi))

    def _sample_elements(self, obj_type: str) -> OrbitalElements:
        cfg = OBJECT_TYPES[obj_type]
        alt_lo, alt_hi = cfg["alt_range"]

        # Altitude: log-uniform to reproduce power-law density profile
        log_alt = self.rng.uniform(np.log(alt_lo), np.log(alt_hi))
        alt = np.exp(log_alt)
        sma = sma_from_altitude(alt)

        # Eccentricity: truncated exponential (most LEO objects ~circular)
        ecc = float(np.clip(self.rng.exponential(cfg["ecc_mean"]), 0.0, 0.05))

        inc  = self._sample_inclination(cfg["inc_modes"])
        raan = self.rng.uniform(0, 2 * np.pi)
        argp = self.rng.uniform(0, 2 * np.pi)
        M    = self.rng.uniform(0, 2 * np.pi)

        return OrbitalElements(sma=sma, ecc=ecc, inc=inc, raan=raan,
                                argp=argp, mean_anomaly=M)

    def _sample_mass(self, obj_type: str) -> float:
        lo, hi = OBJECT_TYPES[obj_type]["mass_range"]
        return float(self.rng.uniform(lo, hi))

    def _sample_rcs(self, obj_type: str, mass_kg: float) -> Tuple[float, float]:
        """Return (rcs_m², diameter_m).  RCS loosely correlated with mass."""
        lo, hi = OBJECT_TYPES[obj_type]["rcs_range"]
        # Scale RCS with √mass (larger objects tend to be heavier)
        base_rcs = lo + (hi - lo) * (mass_kg / OBJECT_TYPES[obj_type]["mass_range"][1]) ** 0.5
        rcs = float(np.clip(self.rng.lognormal(np.log(base_rcs), 0.3), lo, hi * 2))
        diameter = float(2.0 * np.sqrt(rcs / np.pi))  # sphere-equivalent
        return rcs, diameter

    # ── Population generators ─────────────────────────────────────────────────

    def generate_active_satellites(self, n: int) -> List[SpaceObject]:
        """Generate n active, manoeuvrable satellites."""
        operators = ["SpaceX", "ESA", "NASA", "ISRO", "JAXA",
                     "OneWeb", "Planet", "Spire", "Telesat", "CNSA"]
        objects = []
        for _ in range(n):
            oid = self._next_id("SAT-")
            elems = self._sample_elements("ACTIVE_SATELLITE")
            elems.object_id = oid
            mass = self._sample_mass("ACTIVE_SATELLITE")
            rcs, diam = self._sample_rcs("ACTIVE_SATELLITE", mass)
            launch_year = int(self.rng.integers(2015, 2025))
            op = self.rng.choice(operators)
            objects.append(SpaceObject(
                object_id=oid, object_type="ACTIVE_SATELLITE",
                elements=elems, mass_kg=mass, rcs_m2=rcs, diameter_m=diam,
                active=True, operator=str(op), launch_year=launch_year))
        return objects

    def generate_rocket_bodies(self, n: int) -> List[SpaceObject]:
        """Generate n discarded rocket upper stages."""
        objects = []
        for _ in range(n):
            oid = self._next_id("RKT-")
            elems = self._sample_elements("ROCKET_BODY")
            elems.object_id = oid
            mass = self._sample_mass("ROCKET_BODY")
            rcs, diam = self._sample_rcs("ROCKET_BODY", mass)
            launch_year = int(self.rng.integers(1998, 2024))
            objects.append(SpaceObject(
                object_id=oid, object_type="ROCKET_BODY",
                elements=elems, mass_kg=mass, rcs_m2=rcs, diameter_m=diam,
                active=False, launch_year=launch_year))
        return objects

    def generate_debris(self, n: int) -> List[SpaceObject]:
        """Generate n fragmentation debris objects."""
        objects = []
        for _ in range(n):
            oid = self._next_id("DEB-")
            elems = self._sample_elements("DEBRIS")
            elems.object_id = oid
            mass = self._sample_mass("DEBRIS")
            rcs, diam = self._sample_rcs("DEBRIS", mass)
            objects.append(SpaceObject(
                object_id=oid, object_type="DEBRIS",
                elements=elems, mass_kg=mass, rcs_m2=rcs, diameter_m=diam,
                active=False))
        return objects

    def generate_defunct_satellites(self, n: int) -> List[SpaceObject]:
        """Generate n defunct / non-manoeuvrable satellites."""
        objects = []
        for _ in range(n):
            oid = self._next_id("DEF-")
            elems = self._sample_elements("DEFUNCT_SATELLITE")
            elems.object_id = oid
            mass = self._sample_mass("DEFUNCT_SATELLITE")
            rcs, diam = self._sample_rcs("DEFUNCT_SATELLITE", mass)
            launch_year = int(self.rng.integers(2000, 2023))
            objects.append(SpaceObject(
                object_id=oid, object_type="DEFUNCT_SATELLITE",
                elements=elems, mass_kg=mass, rcs_m2=rcs, diameter_m=diam,
                active=False, launch_year=launch_year))
        return objects

    def generate(self,
                 n_active:   int = 50,
                 n_debris:   int = 150,
                 n_rockets:  int = 20,
                 n_defunct:  int = 30) -> List[SpaceObject]:
        """
        Generate a full mixed-population catalogue.

        Parameters
        ----------
        n_active  : Number of active satellites
        n_debris  : Number of debris objects
        n_rockets : Number of rocket bodies
        n_defunct : Number of defunct satellites

        Returns
        -------
        Combined list of SpaceObject instances
        """
        catalogue = (
            self.generate_active_satellites(n_active) +
            self.generate_rocket_bodies(n_rockets) +
            self.generate_debris(n_debris) +
            self.generate_defunct_satellites(n_defunct)
        )
        print(f"[DebrisGenerator] Generated {len(catalogue)} space objects:")
        print(f"  Active satellites  : {n_active}")
        print(f"  Rocket bodies      : {n_rockets}")
        print(f"  Debris fragments   : {n_debris}")
        print(f"  Defunct satellites : {n_defunct}")
        return catalogue

    def generate_fragmentation_cloud(self,
                                     parent: SpaceObject,
                                     n_fragments: int = 50) -> List[SpaceObject]:
        """
        Simulate a debris cloud from a fragmentation event (explosion/collision).
        Fragments are given orbital elements close to the parent orbit with
        random velocity perturbations.

        Parameters
        ----------
        parent      : Parent SpaceObject at breakup
        n_fragments : Number of fragments to generate

        Returns
        -------
        List of SpaceObject (debris type)
        """
        fragments = []
        for i in range(n_fragments):
            oid = self._next_id("FRAG-")
            # Perturb parent elements slightly
            dv_mag = self.rng.uniform(0.001, 0.5)  # km/s delta-v from breakup
            delta_sma  = self.rng.normal(0, dv_mag * 20)
            delta_ecc  = self.rng.normal(0, 0.005)
            delta_inc  = self.rng.normal(0, np.radians(1.0))
            delta_raan = self.rng.normal(0, np.radians(2.0))
            delta_argp = self.rng.normal(0, np.radians(5.0))

            new_sma = max(parent.elements.sma + delta_sma, R_EARTH + 120)
            new_ecc = float(np.clip(parent.elements.ecc + delta_ecc, 0.0, 0.3))

            elems = OrbitalElements(
                sma=new_sma,
                ecc=new_ecc,
                inc=parent.elements.inc  + delta_inc,
                raan=parent.elements.raan + delta_raan,
                argp=parent.elements.argp + delta_argp,
                mean_anomaly=self.rng.uniform(0, 2 * np.pi),
                object_id=oid
            )

            mass = float(self.rng.uniform(0.001, min(parent.mass_kg * 0.1, 50)))
            rcs  = float(self.rng.uniform(0.001, 0.5))
            diam = float(2.0 * np.sqrt(rcs / np.pi))

            fragments.append(SpaceObject(
                object_id=oid, object_type="DEBRIS",
                elements=elems, mass_kg=mass, rcs_m2=rcs, diameter_m=diam,
                active=False))

        return fragments

    # ── Export utilities ──────────────────────────────────────────────────────

    def to_dataframe(self, objects: List[SpaceObject]) -> pd.DataFrame:
        """Convert list of SpaceObjects to a flat Pandas DataFrame."""
        return pd.DataFrame([obj.to_dict() for obj in objects])

    def save_catalogue(self, objects: List[SpaceObject],
                       filepath: str = "data/catalogue.csv") -> pd.DataFrame:
        """Save object catalogue to CSV and return DataFrame."""
        df = self.to_dataframe(objects)
        df.to_csv(filepath, index=False)
        print(f"[DebrisGenerator] Catalogue saved → {filepath}  ({len(df)} objects)")
        return df


if __name__ == "__main__":
    gen = DebrisGenerator(seed=42)
    cat = gen.generate(n_active=30, n_debris=80, n_rockets=10, n_defunct=15)
    df  = gen.to_dataframe(cat)
    print(f"\nSample:\n{df.head()}")
    print(f"\nAltitude stats [km]:\n{df['altitude_km'].describe()}")
    print("Debris generator module: OK ✓")
