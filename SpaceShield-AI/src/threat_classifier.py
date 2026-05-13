"""
threat_classifier.py
=====================
Machine-learning-based threat level classifier for SpaceShield AI.

Trains a Random Forest ensemble on synthetic conjunction data and
classifies each conjunction event into one of four threat levels:
  LOW | MEDIUM | HIGH | CRITICAL

Features engineered from Conjunction Data Messages (CDMs).

Also supports export of the trained model for reproducible inference.
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
import joblib

from src.collision_prediction import ConjunctionEvent, TLE_PC_GREEN, TLE_PC_YELLOW, TLE_PC_ORANGE

warnings.filterwarnings("ignore", category=UserWarning)


# ─── Threat Levels ────────────────────────────────────────────────────────────

THREAT_LEVELS  = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
THREAT_NUMERIC = {lvl: i for i, lvl in enumerate(THREAT_LEVELS)}


# ─── Feature Engineering ──────────────────────────────────────────────────────

def extract_features(events: List[ConjunctionEvent]) -> pd.DataFrame:
    """
    Extract ML features from a list of conjunction events.

    Feature vector (12 features):
    ─────────────────────────────
    1.  miss_distance_km         – Miss distance at TCA
    2.  rel_velocity_kms         – Relative speed at TCA
    3.  log_pc                   – log10 of collision probability
    4.  combined_rcs_m2          – Combined cross-section
    5.  combined_mass_kg         – Combined mass
    6.  primary_altitude_km      – Altitude of primary asset
    7.  secondary_altitude_km    – Altitude of secondary object
    8.  altitude_diff_km         – |alt1 – alt2|  (orbit proximity)
    9.  lead_time_hours          – Time to TCA
    10. inv_lead_time            – 1/lead_time  (urgency signal)
    11. kinetic_energy_proxy     – 0.5·m·v²  (damage potential)
    12. miss_vel_ratio           – miss_dist / rel_velocity  (geometry)
    """
    rows = []
    for ev in events:
        log_pc = np.log10(max(ev.pc, 1e-12))
        ke_proxy = 0.5 * ev.combined_mass_kg * ev.rel_velocity_kms**2
        inv_lead = 1.0 / max(ev.lead_time_hours, 0.1)
        miss_vel = ev.miss_distance_km / max(ev.rel_velocity_kms, 0.01)
        alt_diff = abs(ev.primary_altitude - ev.secondary_altitude)

        rows.append({
            "miss_distance_km":    ev.miss_distance_km,
            "rel_velocity_kms":    ev.rel_velocity_kms,
            "log_pc":              log_pc,
            "combined_rcs_m2":     ev.combined_rcs_m2,
            "combined_mass_kg":    ev.combined_mass_kg,
            "primary_altitude_km": ev.primary_altitude,
            "secondary_altitude_km": ev.secondary_altitude,
            "altitude_diff_km":    alt_diff,
            "lead_time_hours":     ev.lead_time_hours,
            "inv_lead_time":       inv_lead,
            "kinetic_energy_proxy": ke_proxy,
            "miss_vel_ratio":      miss_vel,
        })
    return pd.DataFrame(rows)


def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from a CDM DataFrame (already-saved events).
    Handles string pc values like '1.23e-04'.
    """
    df = df.copy()
    if df["pc"].dtype == object:
        df["pc"] = df["pc"].astype(float)

    df["log_pc"]          = np.log10(df["pc"].clip(lower=1e-12))
    df["inv_lead_time"]   = 1.0 / df["lead_time_hours"].clip(lower=0.1)
    df["kinetic_energy_proxy"] = (
        0.5 * df["combined_mass_kg"] * df["rel_velocity_kms"]**2
    )
    df["miss_vel_ratio"]  = (
        df["miss_distance_km"] / df["rel_velocity_kms"].clip(lower=0.01)
    )
    df["altitude_diff_km"] = (
        df["primary_altitude_km"] - df["secondary_altitude_km"]
    ).abs()

    feature_cols = [
        "miss_distance_km", "rel_velocity_kms", "log_pc",
        "combined_rcs_m2", "combined_mass_kg",
        "primary_altitude_km", "secondary_altitude_km",
        "altitude_diff_km", "lead_time_hours",
        "inv_lead_time", "kinetic_energy_proxy", "miss_vel_ratio",
    ]
    return df[feature_cols]


# ─── Synthetic Training Data Generator ───────────────────────────────────────

def generate_training_data(n_samples: int = 5000,
                            seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic labelled conjunction dataset for training.

    Labels are assigned via a physics-based rule set reflecting real
    operational thresholds used by NASA/ESA conjunction analysis teams.

    Parameters
    ----------
    n_samples : Number of synthetic CDMs
    seed      : Random seed

    Returns
    -------
    (X: DataFrame of features, y: array of integer labels)
    """
    rng = np.random.default_rng(seed)

    # ── Sample raw parameters ────────────────────────────────────────────────
    # Miss distance: log-uniform 0.001 – 10 km
    miss_dist = np.exp(rng.uniform(np.log(0.001), np.log(10.0), n_samples))

    # Relative velocity: 0.5 – 15 km/s (LEO encounters)
    rel_vel = rng.uniform(0.5, 15.0, n_samples)

    # Combined RCS: 0.001 – 50 m²
    comb_rcs = np.exp(rng.uniform(np.log(0.001), np.log(50.0), n_samples))

    # Combined mass: 0.01 – 15000 kg
    comb_mass = np.exp(rng.uniform(np.log(0.01), np.log(15000), n_samples))

    # Altitudes
    prim_alt = rng.uniform(200, 1500, n_samples)
    sec_alt  = prim_alt + rng.normal(0, 50, n_samples)
    sec_alt  = np.clip(sec_alt, 200, 2000)

    # Lead time: 0.5 – 72 hours
    lead_time = rng.uniform(0.5, 72.0, n_samples)

    # ── Compute Pc ───────────────────────────────────────────────────────────
    from src.collision_prediction import pc_chan_2d
    pc = np.array([
        pc_chan_2d(md, rv, rcs)
        for md, rv, rcs in zip(miss_dist, rel_vel, comb_rcs)
    ])

    # ── Build feature matrix ─────────────────────────────────────────────────
    log_pc      = np.log10(np.clip(pc, 1e-12, 1.0))
    inv_lead    = 1.0 / np.clip(lead_time, 0.1, 1000)
    ke_proxy    = 0.5 * comb_mass * rel_vel**2
    miss_vel_r  = miss_dist / np.clip(rel_vel, 0.01, 100)
    alt_diff    = np.abs(prim_alt - sec_alt)

    X = pd.DataFrame({
        "miss_distance_km":    miss_dist,
        "rel_velocity_kms":    rel_vel,
        "log_pc":              log_pc,
        "combined_rcs_m2":     comb_rcs,
        "combined_mass_kg":    comb_mass,
        "primary_altitude_km": prim_alt,
        "secondary_altitude_km": sec_alt,
        "altitude_diff_km":    alt_diff,
        "lead_time_hours":     lead_time,
        "inv_lead_time":       inv_lead,
        "kinetic_energy_proxy":ke_proxy,
        "miss_vel_ratio":      miss_vel_r,
    })

    # ── Assign labels via rule set ───────────────────────────────────────────
    labels = _assign_labels(miss_dist, pc, lead_time, comb_rcs, rel_vel, rng)

    return X, labels


def _assign_labels(miss_dist, pc, lead_time, comb_rcs,
                   rel_vel, rng) -> np.ndarray:
    """
    Physics-based multi-factor label assignment.
    Mimics NASA/ESA operational threat classification.
    """
    n = len(miss_dist)
    labels = np.zeros(n, dtype=int)  # 0=LOW default

    for i in range(n):
        md = miss_dist[i]
        p  = pc[i]
        lt = lead_time[i]
        rv = rel_vel[i]
        rcs = comb_rcs[i]

        # CRITICAL: very high Pc OR very close + fast + large object
        if p >= TLE_PC_ORANGE:
            labels[i] = 3   # CRITICAL
        elif p >= TLE_PC_YELLOW:
            labels[i] = 2   # HIGH
        elif p >= TLE_PC_GREEN:
            labels[i] = 1   # MEDIUM
        else:
            labels[i] = 0   # LOW

        # Upgrades based on context
        if md < 0.1:                          # < 100 m miss
            labels[i] = max(labels[i], 3)
        elif md < 0.5 and rv > 10.0:          # fast + close
            labels[i] = max(labels[i], 2)
        if lt < 2.0 and labels[i] >= 1:       # urgent lead time
            labels[i] = min(labels[i] + 1, 3)
        if rcs > 20.0 and labels[i] >= 1:     # large debris
            labels[i] = min(labels[i] + 1, 3)

        # Small random noise to prevent perfectly separable classes
        if rng.random() < 0.05:
            labels[i] = int(np.clip(labels[i] + rng.choice([-1, 1]), 0, 3))

    return labels


# ─── Threat Classifier ────────────────────────────────────────────────────────

class ThreatClassifier:
    """
    Random Forest-based threat level classifier for conjunction events.

    Workflow
    --------
    1. train()         – Train on synthetic data (or custom dataset)
    2. predict()       – Classify new conjunction events
    3. evaluate()      – Print classification metrics
    4. save() / load() – Persist trained model

    Example
    -------
    >>> clf = ThreatClassifier()
    >>> clf.train(n_samples=5000)
    >>> predictions = clf.predict(events)
    """

    FEATURE_NAMES = [
        "miss_distance_km", "rel_velocity_kms", "log_pc",
        "combined_rcs_m2", "combined_mass_kg",
        "primary_altitude_km", "secondary_altitude_km",
        "altitude_diff_km", "lead_time_hours",
        "inv_lead_time", "kinetic_energy_proxy", "miss_vel_ratio",
    ]

    def __init__(self, model_type: str = "random_forest"):
        """
        Parameters
        ----------
        model_type : 'random_forest' | 'gradient_boost' | 'svm'
        """
        self.model_type = model_type
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(THREAT_LEVELS)
        self.is_trained = False
        self._metrics: dict = {}

    def _build_pipeline(self) -> Pipeline:
        if self.model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == "gradient_boost":
            clf = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                random_state=42
            )
        elif self.model_type == "svm":
            clf = SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf)
        ])

    def train(self, n_samples: int = 6000,
              X_custom: Optional[pd.DataFrame] = None,
              y_custom: Optional[np.ndarray]   = None,
              test_size: float = 0.2,
              verbose: bool = True) -> dict:
        """
        Train the classifier on synthetic (or custom) data.

        Parameters
        ----------
        n_samples : Synthetic training samples (ignored if X_custom given)
        X_custom  : Optional custom feature DataFrame
        y_custom  : Optional custom label array (integer-coded)
        test_size : Fraction held out for evaluation
        verbose   : Print training metrics

        Returns
        -------
        Metrics dictionary
        """
        if X_custom is not None and y_custom is not None:
            X, y = X_custom, y_custom
        else:
            X, y = generate_training_data(n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # ── Evaluation ───────────────────────────────────────────────────────
        y_pred = self.pipeline.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")

        # Cross-val
        cv_scores = cross_val_score(self.pipeline, X_train, y_train,
                                    cv=5, scoring="f1_weighted")

        self._metrics = {
            "accuracy":  round(acc,  4),
            "f1_weighted": round(f1, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std":  round(cv_scores.std(),  4),
            "n_train":    len(X_train),
            "n_test":     len(X_test),
        }

        if verbose:
            print(f"\n[ThreatClassifier] Training complete ({self.model_type})")
            print(f"  Accuracy       : {acc:.4f}")
            print(f"  F1 (weighted)  : {f1:.4f}")
            print(f"  CV F1 (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print("\n  Classification Report:")
            target_names = [THREAT_LEVELS[i] for i in sorted(np.unique(y_test))]
            print(classification_report(y_test, y_pred,
                                        target_names=target_names, zero_division=0))

        return self._metrics

    def predict(self, events: List[ConjunctionEvent]) -> List[str]:
        """
        Predict threat levels for a list of conjunction events.

        Returns
        -------
        List of strings: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
        """
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        X = extract_features(events)
        y_pred = self.pipeline.predict(X)
        return [THREAT_LEVELS[int(y)] for y in y_pred]

    def predict_proba(self, events: List[ConjunctionEvent]) -> np.ndarray:
        """Return class probability matrix (n_events × 4)."""
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained.")
        X = extract_features(events)
        return self.pipeline.predict_proba(X)

    def feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances for Random Forest / Gradient Boost."""
        if not self.is_trained:
            return None
        try:
            clf = self.pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                return pd.Series(
                    clf.feature_importances_,
                    index=self.FEATURE_NAMES
                ).sort_values(ascending=False)
        except Exception:
            pass
        return None

    def evaluate(self, events: List[ConjunctionEvent],
                 true_labels: List[str]) -> dict:
        """Evaluate against known ground-truth labels."""
        y_pred = self.predict(events)
        y_true_int = [THREAT_NUMERIC[l] for l in true_labels]
        y_pred_int = [THREAT_NUMERIC[l] for l in y_pred]
        acc = accuracy_score(y_true_int, y_pred_int)
        f1  = f1_score(y_true_int, y_pred_int, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1": f1}

    def save(self, filepath: str = "results/threat_classifier.pkl") -> None:
        """Persist trained pipeline to disk."""
        if not self.is_trained:
            raise RuntimeError("Nothing to save — model not trained.")
        joblib.dump(self.pipeline, filepath)
        print(f"[ThreatClassifier] Model saved → {filepath}")

    def load(self, filepath: str = "results/threat_classifier.pkl") -> None:
        """Load a previously saved pipeline."""
        self.pipeline = joblib.load(filepath)
        self.is_trained = True
        print(f"[ThreatClassifier] Model loaded ← {filepath}")

    @property
    def metrics(self) -> dict:
        return self._metrics


if __name__ == "__main__":
    clf = ThreatClassifier(model_type="random_forest")
    metrics = clf.train(n_samples=3000, verbose=True)

    fi = clf.feature_importance()
    if fi is not None:
        print("\nTop-5 feature importances:")
        print(fi.head())
    print("\nThreat classifier module: OK ✓")
