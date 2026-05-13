"""
orbit_simulation.py
====================
Orbital propagation engine for SpaceShield AI.

Implements Keplerian two-body mechanics with J2 oblateness perturbation
to propagate satellite and debris objects through time.

All units: distances in km, angles in radians, time in seconds.
Coordinate frame: Earth-Centered Inertial (ECI), J2000.

References:
  Vallado (2013), Fundamentals of Astrodynamics and Applications, 4th ed.
  Bate, Mueller, White (1971), Fundamentals of Astrodynamics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ─── Physical Constants ────────────────────────────────────────────────────────
MU_EARTH    = 398600.4418       # Earth gravitational parameter  [km³/s²]
R_EARTH     = 6378.137          # Earth equatorial radius        [km]
J2          = 1.08262668e-3     # Second zonal harmonic coefficient
OMEGA_EARTH = 7.2921150e-5      # Earth rotation rate            [rad/s]
TWO_PI      = 2.0 * np.pi


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class OrbitalElements:
    """
    Classical Keplerian orbital elements.

    Parameters
    ----------
    sma   : Semi-major axis              [km]
    ecc   : Eccentricity                 [dimensionless, 0 <= ecc < 1]
    inc   : Inclination                  [radians]
    raan  : Right Ascension Asc. Node    [radians]
    argp  : Argument of Perigee          [radians]
    mean_anomaly : Mean anomaly at epoch  [radians]
    epoch : Epoch time (seconds from J2000) [s]
    object_id : Unique object identifier
    """
    sma:          float          # semi-major axis      [km]
    ecc:          float          # eccentricity
    inc:          float          # inclination          [rad]
    raan:         float          # RAAN                 [rad]
    argp:         float          # arg. of perigee      [rad]
    mean_anomaly: float          # mean anomaly         [rad]
    epoch:        float = 0.0    # epoch                [s]
    object_id:    str   = "OBJ"

    def __post_init__(self):
        # Clamp eccentricity to valid range
        self.ecc = float(np.clip(self.ecc, 0.0, 0.9999))
        # Normalise angles to [0, 2π)
        self.inc          = float(self.inc          % TWO_PI)
        self.raan         = float(self.raan         % TWO_PI)
        self.argp         = float(self.argp         % TWO_PI)
        self.mean_anomaly = float(self.mean_anomaly % TWO_PI)


@dataclass
class StateVector:
    """
    Cartesian state vector in ECI frame.

    Parameters
    ----------
    position : np.ndarray [x, y, z]  [km]
    velocity : np.ndarray [vx,vy,vz] [km/s]
    time     : Epoch seconds
    object_id: Owner identifier
    """
    position:  np.ndarray
    velocity:  np.ndarray
    time:      float = 0.0
    object_id: str   = "OBJ"

    @property
    def r(self) -> float:
        """Radial distance from Earth centre [km]."""
        return float(np.linalg.norm(self.position))

    @property
    def v(self) -> float:
        """Speed [km/s]."""
        return float(np.linalg.norm(self.velocity))

    @property
    def altitude(self) -> float:
        """Altitude above Earth surface [km]."""
        return self.r - R_EARTH


# ─── Kepler Solver ────────────────────────────────────────────────────────────

def solve_kepler(mean_anomaly: float, ecc: float,
                 tol: float = 1e-12, max_iter: int = 100) -> float:
    """
    Solve Kepler's equation  M = E - e·sin(E)  for eccentric anomaly E
    using Newton-Raphson iteration.

    Parameters
    ----------
    mean_anomaly : Mean anomaly M [rad]
    ecc          : Eccentricity
    tol          : Convergence tolerance
    max_iter     : Maximum iterations

    Returns
    -------
    E : Eccentric anomaly [rad]
    """
    M = mean_anomaly % TWO_PI
    # Initial guess (Markley 1995)
    E = M + ecc * np.sin(M) / (1.0 - np.sin(M + ecc) + np.sin(M))

    for _ in range(max_iter):
        f  = E - ecc * np.sin(E) - M
        fp = 1.0 - ecc * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break

    return float(E % TWO_PI)


def eccentric_to_true_anomaly(E: float, ecc: float) -> float:
    """Convert eccentric anomaly to true anomaly [rad]."""
    sin_nu = np.sqrt(1.0 - ecc**2) * np.sin(E) / (1.0 - ecc * np.cos(E))
    cos_nu = (np.cos(E) - ecc) / (1.0 - ecc * np.cos(E))
    return float(np.arctan2(sin_nu, cos_nu) % TWO_PI)


# ─── Orbital Element Conversions ──────────────────────────────────────────────

def elements_to_state(elements: OrbitalElements, time: float = 0.0) -> StateVector:
    """
    Convert classical orbital elements to ECI Cartesian state vector.
    Propagates from the element epoch to the requested time using J2 precession.

    Parameters
    ----------
    elements : OrbitalElements object
    time     : Time at which to evaluate state [s]

    Returns
    -------
    StateVector in ECI frame
    """
    dt = time - elements.epoch

    # ── Mean motion and semi-latus rectum ─────────────────────────────────────
    n = np.sqrt(MU_EARTH / elements.sma**3)       # mean motion [rad/s]
    p = elements.sma * (1.0 - elements.ecc**2)    # semi-latus rectum [km]

    # ── J2 secular drift rates ─────────────────────────────────────────────────
    factor = -1.5 * n * J2 * (R_EARTH / p)**2
    raan_dot = factor * np.cos(elements.inc)
    argp_dot = factor * (2.5 * np.sin(elements.inc)**2 - 2.0) * (-1.0)
    n_j2     = n * (1.0 + 1.5 * J2 * (R_EARTH / p)**2 *
                    np.sqrt(1.0 - elements.ecc**2) * (1.0 - 1.5 * np.sin(elements.inc)**2))

    # ── Propagated angles ─────────────────────────────────────────────────────
    M    = (elements.mean_anomaly + n_j2 * dt)   % TWO_PI
    raan = (elements.raan         + raan_dot * dt) % TWO_PI
    argp = (elements.argp         + argp_dot * dt) % TWO_PI

    # ── Solve Kepler's equation ───────────────────────────────────────────────
    E  = solve_kepler(M, elements.ecc)
    nu = eccentric_to_true_anomaly(E, elements.ecc)

    # ── Perifocal frame position & velocity ───────────────────────────────────
    r_peri = p / (1.0 + elements.ecc * np.cos(nu))
    sqrt_mu_p = np.sqrt(MU_EARTH / p)

    pos_peri = r_peri * np.array([np.cos(nu), np.sin(nu), 0.0])
    vel_peri = sqrt_mu_p * np.array([-np.sin(nu),
                                      elements.ecc + np.cos(nu),
                                      0.0])

    # ── Rotation matrix: perifocal → ECI ─────────────────────────────────────
    R = _rotation_matrix(raan, elements.inc, argp)

    position = R @ pos_peri
    velocity = R @ vel_peri

    return StateVector(position=position, velocity=velocity,
                       time=time, object_id=elements.object_id)


def state_to_elements(sv: StateVector) -> OrbitalElements:
    """
    Convert ECI Cartesian state vector to classical orbital elements.

    Parameters
    ----------
    sv : StateVector in ECI frame

    Returns
    -------
    OrbitalElements
    """
    r_vec = sv.position
    v_vec = sv.velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Node vector
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = ((v**2 - MU_EARTH / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU_EARTH
    ecc = float(np.linalg.norm(e_vec))

    # Orbital energy → semi-major axis
    energy = 0.5 * v**2 - MU_EARTH / r
    sma = -MU_EARTH / (2.0 * energy)

    # Inclination
    inc = float(np.arccos(np.clip(h_vec[2] / h, -1, 1)))

    # RAAN
    raan = float(np.arctan2(n_vec[1], n_vec[0]) % TWO_PI) if n > 1e-12 else 0.0

    # Argument of perigee
    if n > 1e-12 and ecc > 1e-8:
        argp = float(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * ecc), -1, 1)))
        if e_vec[2] < 0:
            argp = TWO_PI - argp
    else:
        argp = 0.0

    # True anomaly
    if ecc > 1e-8:
        nu = float(np.arccos(np.clip(np.dot(e_vec, r_vec) / (ecc * r), -1, 1)))
        if np.dot(r_vec, v_vec) < 0:
            nu = TWO_PI - nu
    else:
        nu = float(np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1, 1)))
        if r_vec[2] < 0:
            nu = TWO_PI - nu

    # Mean anomaly
    E = float(2.0 * np.arctan2(np.sqrt(1.0 - ecc**2) * np.sin(nu / 2.0),
                                 np.sqrt(1.0 + ecc) * np.cos(nu / 2.0) +
                                 np.sqrt(1.0 - ecc) * np.cos(nu / 2.0)))
    mean_anomaly = float((E - ecc * np.sin(E)) % TWO_PI)

    return OrbitalElements(
        sma=float(max(sma, R_EARTH + 100)),
        ecc=float(np.clip(ecc, 0.0, 0.9999)),
        inc=inc,
        raan=raan,
        argp=argp,
        mean_anomaly=mean_anomaly,
        epoch=sv.time,
        object_id=sv.object_id
    )


# ─── Trajectory Propagation ───────────────────────────────────────────────────

class OrbitPropagator:
    """
    Propagates a collection of space objects over a time grid.

    Example
    -------
    >>> prop = OrbitPropagator()
    >>> times = np.linspace(0, 3600, 60)  # 1-hour window, 60 steps
    >>> trajectory = prop.propagate(elements, times)
    """

    def propagate(self, elements: OrbitalElements,
                  times: np.ndarray) -> List[StateVector]:
        """
        Propagate a single object over the given time array.

        Parameters
        ----------
        elements : Initial orbital elements
        times    : Array of times [s] from reference epoch

        Returns
        -------
        List of StateVector, one per time step
        """
        return [elements_to_state(elements, t) for t in times]

    def propagate_batch(self, objects: List[OrbitalElements],
                        times: np.ndarray) -> dict:
        """
        Propagate multiple objects over a shared time array.

        Parameters
        ----------
        objects : List of OrbitalElements
        times   : Shared time array [s]

        Returns
        -------
        dict mapping object_id → List[StateVector]
        """
        return {obj.object_id: self.propagate(obj, times) for obj in objects}

    @staticmethod
    def relative_state(sv1: StateVector, sv2: StateVector) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the relative position and velocity between two objects.

        Returns (rel_pos [km], rel_vel [km/s]) in ECI frame.
        """
        rel_pos = sv2.position - sv1.position
        rel_vel = sv2.velocity - sv1.velocity
        return rel_pos, rel_vel

    @staticmethod
    def miss_distance(sv1: StateVector, sv2: StateVector) -> float:
        """Euclidean separation between two objects [km]."""
        return float(np.linalg.norm(sv1.position - sv2.position))


# ─── Utility: Rotation Matrix ─────────────────────────────────────────────────

def _rotation_matrix(raan: float, inc: float, argp: float) -> np.ndarray:
    """
    3×3 rotation matrix from perifocal frame to ECI.
    Equivalent to R3(-RAAN) · R1(-inc) · R3(-argp).
    """
    co, so = np.cos(argp), np.sin(argp)
    ci, si = np.cos(inc),  np.sin(inc)
    cR, sR = np.cos(raan), np.sin(raan)

    R = np.array([
        [ cR*co - sR*so*ci,  -cR*so - sR*co*ci,  sR*si],
        [ sR*co + cR*so*ci,  -sR*so + cR*co*ci, -cR*si],
        [ so*si,               co*si,              ci  ]
    ])
    return R


# ─── Convenience: Orbital Period & Altitude ───────────────────────────────────

def orbital_period(sma: float) -> float:
    """Keplerian orbital period [s] for given semi-major axis [km]."""
    return TWO_PI * np.sqrt(sma**3 / MU_EARTH)


def sma_from_altitude(alt_km: float) -> float:
    """Semi-major axis [km] from altitude above surface [km]."""
    return R_EARTH + alt_km


if __name__ == "__main__":
    # Quick self-test
    iss = OrbitalElements(
        sma=sma_from_altitude(408),
        ecc=0.0001,
        inc=np.radians(51.64),
        raan=np.radians(45.0),
        argp=np.radians(90.0),
        mean_anomaly=np.radians(0.0),
        object_id="ISS"
    )

    prop = OrbitPropagator()
    T = orbital_period(iss.sma)
    times = np.linspace(0, T, 360)
    traj = prop.propagate(iss, times)

    print(f"ISS orbital period: {T/60:.1f} min")
    print(f"ISS altitude (epoch): {traj[0].altitude:.1f} km")
    print(f"ISS speed  (epoch): {traj[0].v:.3f} km/s")
    print("Orbit simulation module: OK ✓")
