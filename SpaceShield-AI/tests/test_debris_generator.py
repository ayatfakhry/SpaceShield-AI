"""
tests/test_debris_generator.py
================================
Unit tests for the debris_generator module.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from src.debris_generator import DebrisGenerator, SpaceObject, OBJECT_TYPES
from src.orbit_simulation import R_EARTH


@pytest.fixture
def gen():
    return DebrisGenerator(seed=42)


@pytest.fixture
def small_catalogue(gen):
    return gen.generate(n_active=10, n_debris=20, n_rockets=5, n_defunct=5)


class TestDebrisGenerator:

    def test_generate_count(self, gen):
        objs = gen.generate(n_active=10, n_debris=20, n_rockets=5, n_defunct=5)
        assert len(objs) == 40

    def test_object_ids_unique(self, small_catalogue):
        ids = [o.object_id for o in small_catalogue]
        assert len(ids) == len(set(ids)), "Duplicate object IDs detected"

    def test_active_satellite_count(self, gen):
        sats = gen.generate_active_satellites(15)
        assert len(sats) == 15
        assert all(o.object_type == "ACTIVE_SATELLITE" for o in sats)
        assert all(o.active for o in sats)

    def test_debris_not_active(self, gen):
        debs = gen.generate_debris(10)
        assert all(not o.active for o in debs)

    def test_altitudes_in_range(self, small_catalogue):
        for obj in small_catalogue:
            cfg = OBJECT_TYPES[obj.object_type]
            lo, hi = cfg["alt_range"]
            alt = obj.altitude_km
            assert lo - 50 <= alt <= hi + 200, \
                f"{obj.object_id} altitude {alt:.1f} km out of range [{lo},{hi}]"

    def test_eccentricity_bounds(self, small_catalogue):
        for obj in small_catalogue:
            assert 0.0 <= obj.elements.ecc < 1.0

    def test_rcs_positive(self, small_catalogue):
        for obj in small_catalogue:
            assert obj.rcs_m2 > 0
            assert obj.diameter_m > 0

    def test_mass_positive(self, small_catalogue):
        for obj in small_catalogue:
            assert obj.mass_kg > 0

    def test_to_dataframe(self, small_catalogue, gen):
        df = gen.to_dataframe(small_catalogue)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(small_catalogue)
        required_cols = ["object_id", "object_type", "altitude_km",
                         "eccentricity", "inclination_deg", "mass_kg", "active"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_fragmentation_cloud(self, gen, small_catalogue):
        parent = small_catalogue[0]
        frags  = gen.generate_fragmentation_cloud(parent, n_fragments=30)
        assert len(frags) == 30
        assert all(f.object_type == "DEBRIS" for f in frags)
        # All fragments should have positive altitudes
        assert all(f.altitude_km > 100 for f in frags)

    def test_reproducibility(self):
        """Same seed produces identical catalogues."""
        g1 = DebrisGenerator(seed=7)
        g2 = DebrisGenerator(seed=7)
        o1 = g1.generate(n_active=5, n_debris=10, n_rockets=2, n_defunct=3)
        o2 = g2.generate(n_active=5, n_debris=10, n_rockets=2, n_defunct=3)
        ids1 = [o.object_id for o in o1]
        ids2 = [o.object_id for o in o2]
        assert ids1 == ids2, "Different seeds produce identical results (wrong)"

    def test_different_seeds_differ(self):
        """Different seeds produce different catalogues."""
        g1 = DebrisGenerator(seed=1)
        g2 = DebrisGenerator(seed=2)
        o1 = g1.generate(n_active=5, n_debris=10, n_rockets=2, n_defunct=3)
        o2 = g2.generate(n_active=5, n_debris=10, n_rockets=2, n_defunct=3)
        alts1 = [round(o.altitude_km) for o in o1]
        alts2 = [round(o.altitude_km) for o in o2]
        assert alts1 != alts2

    def test_save_catalogue(self, gen, small_catalogue, tmp_path):
        path = str(tmp_path / "cat.csv")
        df   = gen.save_catalogue(small_catalogue, filepath=path)
        assert os.path.exists(path)
        df2 = pd.read_csv(path)
        assert len(df2) == len(small_catalogue)
