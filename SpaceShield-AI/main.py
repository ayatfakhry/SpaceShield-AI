"""
main.py
========
SpaceShield AI — Full Autonomous Pipeline Orchestrator

Runs the complete detection → classification → recommendation pipeline:

  Step 1 : Generate synthetic space object catalogue
  Step 2 : Propagate orbits & screen for conjunctions
  Step 3 : Estimate collision probabilities (Chan 2D model)
  Step 4 : ML threat classification (Random Forest)
  Step 5 : Composite risk scoring
  Step 6 : Autonomous maneuver recommendations
  Step 7 : Generate all visualizations
  Step 8 : Write mission report

Usage:
  python main.py
  python main.py --objects 300 --duration 24 --seed 42
"""

import os
import sys
import time
import argparse
import numpy as np

# ── Ensure src is importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

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


def parse_args():
    p = argparse.ArgumentParser(description="SpaceShield AI Pipeline")
    p.add_argument("--objects",  type=int,   default=200,  help="Total space objects")
    p.add_argument("--duration", type=float, default=24.0, help="Screening window [hours]")
    p.add_argument("--seed",     type=int,   default=42,   help="Random seed")
    p.add_argument("--output",   type=str,   default="results", help="Output directory")
    p.add_argument("--no-plots", action="store_true", help="Skip visualization")
    return p.parse_args()


def banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         🛰   S P A C E S H I E L D   A I                       ║
║    Autonomous Space Threat Detection & Decision Support          ║
║                     Research Edition                             ║
╚══════════════════════════════════════════════════════════════════╝
""")


def step_header(n: int, title: str):
    print(f"\n{'─'*60}")
    print(f"  STEP {n}  ▶  {title}")
    print(f"{'─'*60}")


def run_pipeline(n_objects:    int   = 200,
                 duration_h:   float = 24.0,
                 seed:         int   = 42,
                 output_dir:   str   = "results",
                 skip_plots:   bool  = False) -> dict:
    """
    Execute the full SpaceShield AI pipeline.

    Parameters
    ----------
    n_objects   : Total number of space objects to simulate
    duration_h  : Conjunction screening window [hours]
    seed        : Random seed for reproducibility
    output_dir  : Directory for output files
    skip_plots  : If True, skip visualization generation

    Returns
    -------
    Dictionary with summary statistics and file paths
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    banner()
    print(f"  Config: {n_objects} objects | {duration_h:.0f}h window | "
          f"seed={seed} | output={output_dir}/")

    # ── Proportional breakdown ────────────────────────────────────────────────
    n_active  = max(int(n_objects * 0.25), 5)
    n_debris  = max(int(n_objects * 0.50), 10)
    n_rockets = max(int(n_objects * 0.12), 3)
    n_defunct = n_objects - n_active - n_debris - n_rockets

    results = {}

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1: Generate Catalogue
    # ════════════════════════════════════════════════════════════════════════
    step_header(1, "Space Object Catalogue Generation")
    gen = DebrisGenerator(seed=seed)
    objects = gen.generate(
        n_active=n_active, n_debris=n_debris,
        n_rockets=n_rockets, n_defunct=n_defunct
    )
    active_ids  = {o.object_id for o in objects if o.active}
    obj_masses  = {o.object_id: o.mass_kg for o in objects}

    cat_path = os.path.join("data", "catalogue.csv")
    df_cat = gen.save_catalogue(objects, filepath=cat_path)
    results["catalogue_path"] = cat_path
    results["n_objects"] = len(objects)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 & 3: Conjunction Screening + Pc Estimation
    # ════════════════════════════════════════════════════════════════════════
    step_header(2, "Conjunction Screening & Collision Probability Estimation")
    screener = ConjunctionScreener(
        screening_threshold_km=12.0,
        conjunction_threshold_km=6.0
    )
    events = screener.screen(objects, duration_hours=duration_h, verbose=True)

    events_path = os.path.join(output_dir, "close_approaches.csv")
    df_events = screener.save_events(events, filepath=events_path)
    results["n_conjunctions"] = len(events)
    results["events_path"]    = events_path

    by_risk = {}
    for ev in events:
        by_risk[ev.risk_level] = by_risk.get(ev.risk_level, 0) + 1
    results["events_by_risk"] = by_risk
    print(f"\n  Conjunction summary: {by_risk}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4: ML Threat Classification
    # ════════════════════════════════════════════════════════════════════════
    step_header(4, "ML Threat Classification (Random Forest)")
    clf = ThreatClassifier(model_type="random_forest")
    train_metrics = clf.train(n_samples=8000, verbose=True)
    results["clf_accuracy"] = train_metrics.get("accuracy")
    results["clf_f1"]       = train_metrics.get("f1_weighted")

    ml_labels = []
    if events:
        ml_labels = clf.predict(events)
        print(f"\n  ML predictions: {dict(zip(*np.unique(ml_labels, return_counts=True)))}")

    # Save classifier
    clf_path = os.path.join(output_dir, "threat_classifier.pkl")
    clf.save(filepath=clf_path)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 5: Risk Scoring
    # ════════════════════════════════════════════════════════════════════════
    step_header(5, "Composite Risk Scoring")
    engine = RiskEngine()
    scores = engine.score_events(events, active_ids=active_ids,
                                  threat_labels=ml_labels if ml_labels else None)

    scores_path = os.path.join(output_dir, "risk_scores.csv")
    engine.save(scores, filepath=scores_path)
    stats = engine.summary_stats(scores)
    results["risk_stats"]   = stats
    results["scores_path"]  = scores_path
    print(f"\n  Risk stats: {stats}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 6: Maneuver Recommendations
    # ════════════════════════════════════════════════════════════════════════
    step_header(6, "Autonomous Maneuver Recommendations")
    recommender = ManeuverRecommender(target_pc=1e-5)
    recs = recommender.recommend(events, scores, object_masses=obj_masses)

    recs_path = os.path.join(output_dir, "maneuver_recommendations.csv")
    recommender.save(recs, filepath=recs_path)
    recommender.print_summary(recs)

    n_urgent = sum(1 for r in recs if r.urgent)
    n_active_recs = sum(1 for r in recs if r.delta_v_ms > 0)
    results["n_recommendations"] = len(recs)
    results["n_urgent"]          = n_urgent
    results["n_active_maneuvers"]= n_active_recs
    results["recs_path"]         = recs_path

    # ════════════════════════════════════════════════════════════════════════
    # STEP 7: Visualizations
    # ════════════════════════════════════════════════════════════════════════
    if not skip_plots:
        step_header(7, "Visualization Suite")

        fi = clf.feature_importance()

        plot_tasks = [
            ("Orbit 3D plot",       lambda: orbit_plot_3d(
                objects, filepath=os.path.join(output_dir, "orbit_plot.png"))),
            ("Risk heatmap",        lambda: risk_heatmap(
                objects, events,
                filepath=os.path.join(output_dir, "risk_heatmap.png"))),
            ("Threat distribution", lambda: threat_distribution(
                events, ml_labels if ml_labels else None,
                filepath=os.path.join(output_dir, "threat_distribution.png"))),
            ("Pc histogram",        lambda: collision_probability_histogram(
                events, filepath=os.path.join(output_dir, "pc_histogram.png"))),
            ("Risk score dist.",    lambda: risk_score_distribution(
                scores, filepath=os.path.join(output_dir, "risk_score_distribution.png"))),
            ("Maneuver chart",      lambda: maneuver_summary_chart(
                recs, filepath=os.path.join(output_dir, "maneuver_summary.png"))),
            ("Conjunction timeline",lambda: conjunction_timeline(
                events, scores,
                filepath=os.path.join(output_dir, "conjunction_timeline.png"))),
            ("Feature importance",  lambda: feature_importance_plot(
                fi, filepath=os.path.join(output_dir, "feature_importance.png"))
                if fi is not None else None),
            ("Telemetry dashboard", lambda: telemetry_dashboard(
                objects, events, scores, recs,
                filepath=os.path.join(output_dir, "telemetry_dashboard.png"))),
        ]

        plot_paths = []
        for name, fn in plot_tasks:
            try:
                print(f"  Generating: {name}…")
                fn()
                plot_paths.append(name)
            except Exception as exc:
                print(f"  [WARN] {name} failed: {exc}")

        results["plots_generated"] = len(plot_paths)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 8: Mission Report
    # ════════════════════════════════════════════════════════════════════════
    step_header(8, "Mission Report Generation")
    report_path = os.path.join(output_dir, "simulation_report.txt")
    elapsed = time.time() - t0

    _write_report(
        filepath=report_path,
        objects=objects, events=events, scores=scores, recs=recs,
        train_metrics=train_metrics, stats=stats,
        elapsed=elapsed, seed=seed, duration_h=duration_h,
        output_dir=output_dir
    )
    results["report_path"] = report_path

    # ════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("  ✅  SpaceShield AI pipeline complete")
    print(f"{'═'*60}")
    print(f"  Objects tracked    : {len(objects)}")
    print(f"  Conjunctions found : {len(events)}")
    print(f"  ML accuracy        : {train_metrics.get('accuracy', 0):.4f}")
    print(f"  Risk scores        : {len(scores)}")
    print(f"  Maneuver recs      : {len(recs)}  (urgent: {n_urgent})")
    print(f"  Elapsed time       : {elapsed:.1f}s")
    print(f"  Output directory   : {output_dir}/")
    print(f"{'═'*60}\n")

    return results


def _write_report(filepath, objects, events, scores, recs,
                   train_metrics, stats, elapsed,
                   seed, duration_h, output_dir):
    """Write a comprehensive plain-text mission report."""

    by_type = {}
    for o in objects:
        by_type[o.object_type] = by_type.get(o.object_type, 0) + 1

    by_risk = {}
    for ev in events:
        by_risk[ev.risk_level] = by_risk.get(ev.risk_level, 0) + 1

    top5_events = events[:5] if events else []
    top5_recs   = [r for r in recs if r.delta_v_ms > 0][:5]
    n_urgent    = sum(1 for r in recs if r.urgent)

    lines = [
        "=" * 70,
        "  SPACESHIELD AI — SIMULATION MISSION REPORT",
        "=" * 70,
        "",
        f"  Pipeline Configuration",
        f"  ─────────────────────",
        f"  Random seed        : {seed}",
        f"  Screening window   : {duration_h:.0f} hours",
        f"  Output directory   : {output_dir}/",
        f"  Wall-clock time    : {elapsed:.1f} seconds",
        "",
        f"  Space Object Catalogue ({len(objects)} objects)",
        f"  ──────────────────────────────────────────",
    ]
    for obj_type, count in sorted(by_type.items()):
        lines.append(f"    {obj_type:<25}: {count}")

    lines += [
        "",
        f"  Conjunction Analysis ({len(events)} events)",
        f"  ────────────────────────────────────────",
    ]
    for lvl, count in sorted(by_risk.items()):
        lines.append(f"    {lvl:<10}: {count}")

    if top5_events:
        lines += ["", "  Top-5 Highest-Risk Events:", "  " + "─" * 55]
        for ev in top5_events:
            lines.append(
                f"  {ev.event_id}  {ev.primary_id} ↔ {ev.secondary_id}"
                f"  Pc={ev.pc:.2e}  miss={ev.miss_distance_km:.3f}km"
                f"  [{ev.risk_level}]"
            )

    lines += [
        "",
        f"  ML Threat Classifier Performance",
        f"  ─────────────────────────────────",
        f"    Model type       : Random Forest",
        f"    Accuracy         : {train_metrics.get('accuracy', 0):.4f}",
        f"    F1 (weighted)    : {train_metrics.get('f1_weighted', 0):.4f}",
        f"    CV F1 (5-fold)   : {train_metrics.get('cv_f1_mean', 0):.4f}"
                                f" ± {train_metrics.get('cv_f1_std', 0):.4f}",
        "",
        f"  Composite Risk Scores ({len(scores)} events)",
        f"  ──────────────────────────────────────────",
        f"    Mean score       : {stats.get('mean_score', 0):.2f}",
        f"    Max score        : {stats.get('max_score', 0):.2f}",
        f"    Min score        : {stats.get('min_score', 0):.2f}",
        f"    Std deviation    : {stats.get('std_score', 0):.2f}",
    ]
    band_dist = stats.get("band_distribution", {})
    if band_dist:
        lines.append(f"    Band distribution:")
        for band, cnt in sorted(band_dist.items()):
            lines.append(f"      {band:<15}: {cnt}")

    lines += [
        "",
        f"  Maneuver Recommendations ({len(recs)} total, {n_urgent} urgent)",
        f"  ────────────────────────────────────────────────────────────────",
    ]
    if top5_recs:
        for rec in top5_recs:
            lines.append(
                f"  {rec.event_id}  {rec.primary_id}  {rec.maneuver_type}"
                f"  Δv={rec.delta_v_ms:.3f}m/s"
                f"  fuel={rec.fuel_cost_kg:.4f}kg"
                + ("  *** URGENT ***" if rec.urgent else "")
            )

    lines += [
        "",
        f"  Output Files",
        f"  ────────────",
        f"    data/catalogue.csv",
        f"    {output_dir}/close_approaches.csv",
        f"    {output_dir}/risk_scores.csv",
        f"    {output_dir}/maneuver_recommendations.csv",
        f"    {output_dir}/threat_classifier.pkl",
        f"    {output_dir}/orbit_plot.png",
        f"    {output_dir}/risk_heatmap.png",
        f"    {output_dir}/threat_distribution.png",
        f"    {output_dir}/pc_histogram.png",
        f"    {output_dir}/risk_score_distribution.png",
        f"    {output_dir}/maneuver_summary.png",
        f"    {output_dir}/conjunction_timeline.png",
        f"    {output_dir}/feature_importance.png",
        f"    {output_dir}/telemetry_dashboard.png",
        "",
        "=" * 70,
        "  SpaceShield AI — End of Report",
        "=" * 70,
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [Report] Saved → {filepath}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        n_objects=args.objects,
        duration_h=args.duration,
        seed=args.seed,
        output_dir=args.output,
        skip_plots=args.no_plots
    )
