"""
tests/test_threat_classifier.py
=================================
Unit tests for the ML threat classifier.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from src.threat_classifier import (
    ThreatClassifier, THREAT_LEVELS,
    generate_training_data, extract_features
)
from src.collision_prediction import ConjunctionEvent


@pytest.fixture(scope="module")
def trained_clf():
    clf = ThreatClassifier(model_type="random_forest")
    clf.train(n_samples=2000, verbose=False)
    return clf


@pytest.fixture
def sample_events():
    return [
        ConjunctionEvent(
            event_id=f"CDM-{i:06d}",
            primary_id="SAT-00001",
            secondary_id=f"DEB-{i:05d}",
            tca=float(i * 3600),
            miss_distance_km=miss,
            rel_velocity_kms=10.0,
            pc=pc,
            combined_rcs_m2=1.5,
            combined_mass_kg=400.0,
            primary_altitude=500.0,
            secondary_altitude=502.0,
            lead_time_hours=float(i + 1)
        )
        for i, (miss, pc) in enumerate([
            (0.05, 1e-2),   # CRITICAL — very close, high Pc
            (0.5,  1e-3),   # HIGH
            (2.0,  1e-5),   # MEDIUM
            (8.0,  1e-8),   # LOW
        ])
    ]


class TestTrainingData:

    def test_generate_shape(self):
        X, y = generate_training_data(n_samples=500)
        assert len(X) == 500
        assert len(y) == 500

    def test_label_range(self):
        _, y = generate_training_data(n_samples=300)
        assert set(y).issubset({0, 1, 2, 3})

    def test_all_classes_present(self):
        _, y = generate_training_data(n_samples=2000, seed=42)
        assert len(set(y)) == 4, f"Only {len(set(y))} classes found"

    def test_feature_names(self):
        X, _ = generate_training_data(n_samples=100)
        for col in ThreatClassifier.FEATURE_NAMES:
            assert col in X.columns, f"Missing feature: {col}"

    def test_no_nan_in_features(self):
        X, _ = generate_training_data(n_samples=500)
        assert not X.isnull().any().any(), "NaN values in training features"


class TestExtractFeatures:

    def test_output_shape(self, sample_events):
        X = extract_features(sample_events)
        assert X.shape == (len(sample_events), len(ThreatClassifier.FEATURE_NAMES))

    def test_log_pc_negative(self, sample_events):
        X = extract_features(sample_events)
        assert (X["log_pc"] <= 0).all(), "log_pc should be ≤ 0"

    def test_no_nan(self, sample_events):
        X = extract_features(sample_events)
        assert not X.isnull().any().any()


class TestThreatClassifier:

    def test_train_returns_metrics(self, trained_clf):
        m = trained_clf.metrics
        assert "accuracy" in m
        assert "f1_weighted" in m
        assert 0.5 <= m["accuracy"] <= 1.0

    def test_predict_returns_valid_labels(self, trained_clf, sample_events):
        labels = trained_clf.predict(sample_events)
        assert len(labels) == len(sample_events)
        assert all(l in THREAT_LEVELS for l in labels)

    def test_predict_proba_shape(self, trained_clf, sample_events):
        proba = trained_clf.predict_proba(sample_events)
        assert proba.shape == (len(sample_events), 4)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_untrained_raises(self):
        clf = ThreatClassifier()
        with pytest.raises(RuntimeError):
            clf.predict([])

    def test_feature_importance_available(self, trained_clf):
        fi = trained_clf.feature_importance()
        assert fi is not None
        assert len(fi) == len(ThreatClassifier.FEATURE_NAMES)
        assert (fi >= 0).all()
        assert abs(fi.sum() - 1.0) < 0.01

    def test_save_and_load(self, trained_clf, tmp_path):
        path = str(tmp_path / "model.pkl")
        trained_clf.save(filepath=path)
        assert os.path.exists(path)

        clf2 = ThreatClassifier()
        clf2.load(filepath=path)
        assert clf2.is_trained

    def test_gradient_boost_trains(self):
        clf = ThreatClassifier(model_type="gradient_boost")
        m   = clf.train(n_samples=500, verbose=False)
        assert m["accuracy"] > 0.4

    def test_high_pc_classified_critical_or_high(self, trained_clf):
        """Very high Pc event should be classified HIGH or CRITICAL."""
        critical_event = ConjunctionEvent(
            event_id="CDM-CRIT",
            primary_id="SAT-A", secondary_id="DEB-B",
            tca=3600.0, miss_distance_km=0.02,
            rel_velocity_kms=12.0, pc=0.05,
            combined_rcs_m2=5.0, combined_mass_kg=1000.0,
            primary_altitude=450.0, secondary_altitude=451.0,
            lead_time_hours=1.0
        )
        label = trained_clf.predict([critical_event])[0]
        assert label in ("HIGH", "CRITICAL"), \
            f"Expected HIGH/CRITICAL for pc=0.05, got {label}"

    def test_low_pc_classified_low_or_medium(self, trained_clf):
        """Very low Pc event should be LOW or MEDIUM."""
        safe_event = ConjunctionEvent(
            event_id="CDM-SAFE",
            primary_id="SAT-A", secondary_id="DEB-B",
            tca=72.0 * 3600, miss_distance_km=9.0,
            rel_velocity_kms=5.0, pc=1e-9,
            combined_rcs_m2=0.1, combined_mass_kg=10.0,
            primary_altitude=600.0, secondary_altitude=601.0,
            lead_time_hours=72.0
        )
        label = trained_clf.predict([safe_event])[0]
        assert label in ("LOW", "MEDIUM"), \
            f"Expected LOW/MEDIUM for pc=1e-9, got {label}"
