"""
scripts/run_simulation.py
==========================
Standalone command-line simulation runner for SpaceShield AI.

Provides fine-grained control over every pipeline stage with
additional options not exposed in main.py.

Usage examples
--------------
# Default run (200 objects, 24h window)
python scripts/run_simulation.py

# Large-scale run with custom output
python scripts/run_simulation.py --objects 500 --duration 48 --output results/run_500

# Debris-only population (stress test screener)
python scripts/run_simulation.py --active 10 --debris 300 --rockets 0 --defunct 0

# Skip plots, just produce CSVs
python scripts/run_simulation.py --no-plots --objects 100 --duration 6

# Reproduce a specific run
python scripts/run_simulation.py --seed 1337 --objects 250

# Show fragmentation cloud scenario
python scripts/run_simulation.py --fragmentation-event --objects 150
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.debris_generator import DebrisGenerator
from src.collision_prediction import ConjunctionScreener
from src.threat_classifier import ThreatClassifier
from src.risk_engine import RiskEngine
from src.maneuver_recommender import ManeuverRecommender
from src.visualization import (
    orbit_plot_3d, risk_heatmap, threat_distribution,
    collision_probability_histogram, risk_score_distribution,
    maneuver_summary_chart, conjunction_timeline,
    feature_importance_plot, telemetry_dashboard
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SpaceShield AI — Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Population
    pop = p.add_argument_group("Population")
    pop.add_argument("--objects",  type=int, default=200,
                     help="Total objects (overrides individual counts)")
    pop.add_argument("--active",   type=int, default=None,
                     help="Active satellites (overrides --objects split)")
    pop.add_argument("--debris",   type=int, default=None,
                     help="Debris fragments")
    pop.add_argument("--rockets",  type=int, default=None,
                     help="Rocket bodies")
    pop.add_argument("--defunct",  type=int, default=None,
                     help="Defunct satellites")

    # Simulation
    sim = p.add_argument_group("Simulation")
    sim.add_argument("--duration",   type=float, default=24.0,
                     help="Screening window [hours]")
    sim.add_argument("--threshold",  type=float, default=12.0,
                     help="Coarse screening threshold [km]")
    sim.add_argument("--conj-threshold", type=float, default=6.0,
                     help="Conjunction CDM threshold [km]")
    sim.add_argument("--seed",       type=int,   default=42,
                     help="Random seed")

    # ML
    ml = p.add_argument_group("Machine Learning")
    ml.add_argument("--train-samples", type=int, default=8000,
                    help="Synthetic training samples")
    ml.add_argument("--model",         type=str,  default="random_forest",
                    choices=["random_forest", "gradient_boost", "svm"],
                    help="Classifier type")

    # Output
    out = p.add_argument_group("Output")
    out.add_argument("--output",    type=str, default="results",
                     help="Output directory")
    out.add_argument("--no-plots",  action="store_true",
                     help="Skip all plot generation")
    out.add_argument("--verbose",   action="store_true", default=True,
                     help="Verbose output")

    # Scenarios
    scn = p.add_argument_group("Scenarios")
    scn.add_argument("--fragmentation-event", action="store_true",
                     help="Add a simulated fragmentation cloud")

    return p.parse_args()


# ─── Population helper ────────────────────────────────────────────────────────

def resolve_population(args):
    """Resolve final population counts from CLI arguments."""
    if all(x is not None for x in [args.active, args.debris,
                                    args.rockets, args.defunct]):
        return args.active, args.debris, args.rockets, args.defunct

    n = args.objects
    n_active  = max(int(n * 0.25), 5)
    n_debris  = max(int(n * 0.50), 10)
    n_rockets = max(int(n * 0.12), 3)
    n_defunct = max(n - n_active - n_debris - n_rockets, 0)

    return (
        args.active   if args.active   is not None else n_active,
        args.debris   if args.debris   is not None else n_debris,
        args.rockets  if args.rockets  is not None else n_rockets,
        args.defunct  if args.defunct  is not None else n_defunct,
    )


# ─── Main simulation ──────────────────────────────────────────────────────────

def run(args):
    t0 = time.time()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    n_active, n_debris, n_rockets, n_defunct = resolve_population(args)
    total = n_active + n_debris + n_rockets + n_defunct

    print("=" * 65)
    print("  🛰  SpaceShield AI  —  Simulation Runner")
    print("=" * 65)
    print(f"  Objects : {total}  (active={n_active}, debris={n_debris}, "
          f"rockets={n_rockets}, defunct={n_defunct})")
    print(f"  Window  : {args.duration:.0f}h  |  Seed: {args.seed}")
    print(f"  Model   : {args.model}  |  Train samples: {args.train_samples}")
    print("=" * 65)

    # ── Step 1: Generate catalogue ────────────────────────────────────────────
    print("\n[1/7] Generating space object catalogue…")
    gen = DebrisGenerator(seed=args.seed)
    objects = gen.generate(
        n_active=n_active, n_debris=n_debris,
        n_rockets=n_rockets, n_defunct=n_defunct
    )

    # Optional: fragmentation event
    if args.fragmentation_event:
        print("      Adding fragmentation cloud scenario…")
        parent = next((o for o in objects if o.object_type == "DEFUNCT_SATELLITE"), objects[0])
        fragments = gen.generate_fragmentation_cloud(parent, n_fragments=40)
        objects.extend(fragments)
        print(f"      Added {len(fragments)} fragments from {parent.object_id}")

    gen.save_catalogue(objects, filepath="data/catalogue.csv")
    active_ids = {o.object_id for o in objects if o.active}
    obj_masses = {o.object_id: o.mass_kg for o in objects}

    # ── Step 2: Conjunction screening ─────────────────────────────────────────
    print(f"\n[2/7] Screening conjunctions ({args.duration:.0f}h window)…")
    screener = ConjunctionScreener(
        screening_threshold_km=args.threshold,
        conjunction_threshold_km=args.conj_threshold
    )
    events = screener.screen(objects, duration_hours=args.duration, verbose=True)
    screener.save_events(events, filepath=os.path.join(args.output, "close_approaches.csv"))

    # ── Step 3: ML threat classification ─────────────────────────────────────
    print(f"\n[3/7] Training ML threat classifier ({args.model})…")
    clf = ThreatClassifier(model_type=args.model)
    metrics = clf.train(n_samples=args.train_samples, verbose=True)
    clf.save(filepath=os.path.join(args.output, "threat_classifier.pkl"))

    ml_labels = clf.predict(events) if events else []
    fi = clf.feature_importance()

    # ── Step 4: Risk scoring ──────────────────────────────────────────────────
    print("\n[4/7] Computing composite risk scores…")
    engine = RiskEngine()
    scores = engine.score_events(events, active_ids=active_ids,
                                  threat_labels=ml_labels or None)
    engine.save(scores, filepath=os.path.join(args.output, "risk_scores.csv"))
    stats = engine.summary_stats(scores)
    print(f"      Mean={stats.get('mean_score',0):.1f}  "
          f"Max={stats.get('max_score',0):.1f}  "
          f"Bands={stats.get('band_distribution',{})}")

    # ── Step 5: Maneuver recommendations ─────────────────────────────────────
    print("\n[5/7] Generating maneuver recommendations…")
    recommender = ManeuverRecommender()
    recs = recommender.recommend(events, scores, object_masses=obj_masses)
    recommender.save(recs, filepath=os.path.join(args.output, "maneuver_recommendations.csv"))
    recommender.print_summary(recs)

    # ── Step 6: Visualizations ────────────────────────────────────────────────
    if not args.no_plots:
        print("\n[6/7] Generating visualizations…")
        _generate_all_plots(objects, events, scores, recs, fi, args.output, ml_labels)
    else:
        print("\n[6/7] Visualizations skipped (--no-plots).")

    # ── Step 7: Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_final_summary(objects, events, scores, recs, metrics, stats, elapsed, args.output)

    return {
        "objects":  len(objects),
        "events":   len(events),
        "scores":   len(scores),
        "recs":     len(recs),
        "elapsed":  elapsed,
    }


def _generate_all_plots(objects, events, scores, recs, fi, output_dir, ml_labels):
    """Generate all visualization outputs."""
    plots = [
        ("orbit_plot.png",              lambda: orbit_plot_3d(objects)),
        ("risk_heatmap.png",            lambda: risk_heatmap(objects, events)),
        ("threat_distribution.png",     lambda: threat_distribution(events, ml_labels or None)),
        ("pc_histogram.png",            lambda: collision_probability_histogram(events)),
        ("risk_score_distribution.png", lambda: risk_score_distribution(scores)),
        ("maneuver_summary.png",        lambda: maneuver_summary_chart(recs)),
        ("conjunction_timeline.png",    lambda: conjunction_timeline(events, scores)),
        ("telemetry_dashboard.png",     lambda: telemetry_dashboard(objects, events, scores, recs)),
    ]
    if fi is not None:
        plots.append(("feature_importance.png", lambda: feature_importance_plot(fi)))

    for fname, fn in plots:
        fpath = os.path.join(output_dir, fname)
        try:
            import src.visualization as viz
            # Override filepath in each call
            fn.__globals__.get("filepath", None)
            # Call with explicit path
            _call_plot(fn, fpath, fname)
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")


def _call_plot(fn, filepath, name):
    """Call plot function, redirecting output filepath."""
    import src.visualization as viz
    # Map function → viz function name for filepath injection
    map_ = {
        "orbit_plot_3d":                     lambda: viz.orbit_plot_3d(fn.__self__ if hasattr(fn, '__self__') else None),
    }
    # Simpler: just call with the right filepath kwarg by inspecting source
    import inspect
    sig = inspect.signature(fn)
    if "filepath" in sig.parameters:
        fn(filepath=filepath)
    else:
        fn()
    print(f"  → {filepath}")


def _print_final_summary(objects, events, scores, recs, metrics, stats, elapsed, output_dir):
    n_urgent = sum(1 for r in recs if r.urgent)
    n_red    = sum(1 for e in events if e.risk_level == "RED")

    print("\n" + "=" * 65)
    print("  ✅  SIMULATION COMPLETE")
    print("=" * 65)
    print(f"  Tracked objects     : {len(objects)}")
    print(f"  Conjunctions        : {len(events)}  (RED: {n_red})")
    print(f"  ML accuracy         : {metrics.get('accuracy', 0):.4f}")
    print(f"  Avg risk score      : {stats.get('mean_score', 0):.1f}/100")
    print(f"  Maneuver recs       : {len(recs)}  (urgent: {n_urgent})")
    print(f"  Total runtime       : {elapsed:.1f}s")
    print(f"  Output              : {output_dir}/")
    print("=" * 65)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
