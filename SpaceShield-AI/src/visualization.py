"""
visualization.py
=================
Visualization suite for SpaceShield AI.

Generates:
  1.  orbit_plot_3d()           – 3D ECI orbit visualization
  2.  orbit_ground_tracks()     – 2D ground track map
  3.  risk_heatmap()            – Altitude × inclination risk heatmap
  4.  threat_distribution()     – ML threat classification bar chart
  5.  collision_probability_histogram() – Pc distribution
  6.  risk_score_distribution() – Composite risk scores
  7.  maneuver_summary_chart()  – Delta-v bar chart by maneuver type
  8.  conjunction_timeline()    – Time-to-TCA vs. risk score scatter
  9.  feature_importance_plot() – ML feature importances
  10. telemetry_dashboard()     – 6-panel mission dashboard
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
import os

from src.orbit_simulation import R_EARTH, orbital_period, elements_to_state
from src.debris_generator import SpaceObject, OBJECT_TYPES
from src.collision_prediction import ConjunctionEvent
from src.risk_engine import RiskScore, RISK_BANDS
from src.maneuver_recommender import ManeuverRecommendation


# ─── Style constants ──────────────────────────────────────────────────────────
BG_DARK   = "#0a0e1a"
BG_PANEL  = "#111827"
ACCENT    = "#00d4ff"
ACCENT2   = "#ff4757"
ACCENT3   = "#2ed573"
ACCENT4   = "#ffa502"
TEXT_COL  = "#e8eaf6"
GRID_COL  = "#1e2a3a"

BAND_COLORS = {
    "MINIMAL":      "#2ed573",
    "LOW":          "#7bed9f",
    "MODERATE":     "#ffa502",
    "HIGH":         "#ff6b35",
    "SEVERE":       "#ff4757",
    "CATASTROPHIC": "#8b0000",
}

THREAT_COLORS = {
    "LOW":      "#2ed573",
    "MEDIUM":   "#ffa502",
    "HIGH":     "#ff6b35",
    "CRITICAL": "#ff4757",
    "GREEN":    "#2ed573",
    "YELLOW":   "#ffa502",
    "ORANGE":   "#ff6b35",
    "RED":      "#ff4757",
    "UNKNOWN":  "#888888",
}

OBJTYPE_COLORS = {
    "ACTIVE_SATELLITE":  "#00d4ff",
    "ROCKET_BODY":       "#ffa502",
    "DEBRIS":            "#ff4757",
    "DEFUNCT_SATELLITE": "#f1c40f",
}


def _apply_dark_style(fig, axes=None):
    """Apply consistent dark space theme to figure and axes."""
    fig.patch.set_facecolor(BG_DARK)
    if axes is None:
        return
    ax_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in np.array(ax_list).flat:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_color(TEXT_COL)
            ax.tick_params(axis="z", colors=TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [VIZ] Saved → {path}")


# ─── 1. 3D Orbit Plot ─────────────────────────────────────────────────────────

def orbit_plot_3d(objects: List[SpaceObject],
                  n_points: int = 180,
                  max_objects: int = 80,
                  filepath: str = "results/orbit_plot.png") -> None:
    """
    3D ECI scatter of orbital positions with Earth sphere.

    Parameters
    ----------
    objects    : List of SpaceObject
    n_points   : Points per orbit trajectory
    max_objects: Subsample for performance
    filepath   : Output file path
    """
    fig = plt.figure(figsize=(14, 11))
    ax  = fig.add_subplot(111, projection="3d")
    _apply_dark_style(fig)
    ax.set_facecolor(BG_DARK)

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    xe = R_EARTH * np.outer(np.cos(u), np.sin(v))
    ye = R_EARTH * np.outer(np.sin(u), np.sin(v))
    ze = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xe, ye, ze, color="#1a3a5c", alpha=0.6, linewidth=0, zorder=0)
    # Grid lines on Earth
    ax.plot_wireframe(xe, ye, ze, color="#234060", alpha=0.15, linewidth=0.3)

    # Sample objects
    sample = objects[:max_objects] if len(objects) > max_objects else objects

    legend_handles = {}
    for obj in sample:
        period = orbital_period(obj.elements.sma)
        times  = np.linspace(0, period, n_points)
        pos    = np.array([elements_to_state(obj.elements, t).position for t in times])

        color = OBJTYPE_COLORS.get(obj.object_type, "#888888")
        alpha = 0.8 if obj.active else 0.35
        lw    = 1.2 if obj.active else 0.5

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=color, alpha=alpha, linewidth=lw, zorder=1)

        if obj.object_type not in legend_handles:
            legend_handles[obj.object_type] = mpatches.Patch(
                color=color, label=obj.object_type.replace("_", " ").title())

    ax.legend(handles=list(legend_handles.values()),
              loc="upper left", framealpha=0.3, labelcolor=TEXT_COL,
              facecolor=BG_PANEL, edgecolor=GRID_COL, fontsize=8)

    ax.set_xlabel("X [km]", labelpad=8)
    ax.set_ylabel("Y [km]", labelpad=8)
    ax.set_zlabel("Z [km]", labelpad=8)
    ax.set_title("SpaceShield AI  —  Tracked Object Orbits (ECI Frame)",
                 color=TEXT_COL, fontsize=13, fontweight="bold", pad=15)

    # Axis pane colour
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COL)

    ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.grid(True, color=GRID_COL, linewidth=0.3)

    _save(fig, filepath)


# ─── 2. Risk Heatmap ─────────────────────────────────────────────────────────

def risk_heatmap(objects: List[SpaceObject],
                 events: List[ConjunctionEvent],
                 filepath: str = "results/risk_heatmap.png") -> None:
    """
    2D heatmap: altitude band × inclination bin, coloured by conjunction count.
    """
    alt_bins = np.arange(200, 2100, 100)
    inc_bins = np.arange(0, 181, 10)

    grid = np.zeros((len(alt_bins) - 1, len(inc_bins) - 1))

    # Build fast lookup: object_id → (alt, inc)
    obj_lookup = {o.object_id: (o.altitude_km, np.degrees(o.elements.inc))
                  for o in objects}

    for ev in events:
        alt, inc = obj_lookup.get(ev.primary_id, (None, None))
        if alt is None:
            continue
        ai = int(np.searchsorted(alt_bins, alt) - 1)
        ii = int(np.searchsorted(inc_bins, inc) - 1)
        if 0 <= ai < grid.shape[0] and 0 <= ii < grid.shape[1]:
            grid[ai, ii] += np.log10(max(ev.pc, 1e-12)) + 12  # +12 to make positive

    fig, ax = plt.subplots(figsize=(14, 7))
    _apply_dark_style(fig, ax)

    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[inc_bins[0], inc_bins[-1],
                            alt_bins[0], alt_bins[-1]],
                   cmap="inferno", interpolation="bilinear")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Conjunction Risk Score (log Pc scaled)", color=TEXT_COL)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    ax.set_xlabel("Inclination [°]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("SpaceShield AI  —  Collision Risk Heatmap\n"
                 "(Altitude × Inclination — brighter = higher risk)",
                 color=TEXT_COL, fontsize=12, fontweight="bold")

    # Mark ISS altitude
    ax.axhline(408, color=ACCENT, linewidth=1.2, linestyle="--", alpha=0.6)
    ax.text(2, 415, "ISS ~408 km", color=ACCENT, fontsize=7, alpha=0.8)

    _save(fig, filepath)


# ─── 3. Threat Distribution ───────────────────────────────────────────────────

def threat_distribution(events: List[ConjunctionEvent],
                         ml_labels: List[str] = None,
                         filepath: str = "results/threat_distribution.png") -> None:
    """Bar chart of threat level distribution (Pc-based and ML-based)."""
    pc_counts  = {}
    ml_counts  = {}

    for ev in events:
        pc_counts[ev.risk_level]  = pc_counts.get(ev.risk_level, 0) + 1

    if ml_labels:
        for lbl in ml_labels:
            ml_counts[lbl] = ml_counts.get(lbl, 0) + 1

    categories = ["GREEN", "YELLOW", "ORANGE", "RED"]
    ml_cats    = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_style(fig, axes)

    # Pc-based
    ax1 = axes[0]
    vals1 = [pc_counts.get(c, 0) for c in categories]
    cols1 = [THREAT_COLORS[c] for c in categories]
    bars1 = ax1.bar(categories, vals1, color=cols1, edgecolor=BG_DARK, linewidth=1.2)
    for bar, v in zip(bars1, vals1):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                     str(v), ha="center", va="bottom", color=TEXT_COL, fontsize=10)
    ax1.set_title("Conjunction Risk (Pc-based)", color=TEXT_COL, fontweight="bold")
    ax1.set_ylabel("Number of Events")

    # ML-based
    ax2 = axes[1]
    if ml_labels:
        vals2 = [ml_counts.get(c, 0) for c in ml_cats]
        cols2 = [THREAT_COLORS[c] for c in ml_cats]
        bars2 = ax2.bar(ml_cats, vals2, color=cols2, edgecolor=BG_DARK, linewidth=1.2)
        for bar, v in zip(bars2, vals2):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                         str(v), ha="center", va="bottom", color=TEXT_COL, fontsize=10)
        ax2.set_title("Threat Classification (ML Model)", color=TEXT_COL, fontweight="bold")
        ax2.set_ylabel("Number of Events")
    else:
        ax2.text(0.5, 0.5, "ML labels\nnot available",
                 ha="center", va="center", transform=ax2.transAxes,
                 color=TEXT_COL, fontsize=13)

    fig.suptitle("SpaceShield AI — Threat Distribution Analysis",
                 color=TEXT_COL, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, filepath)


# ─── 4. Collision Probability Histogram ──────────────────────────────────────

def collision_probability_histogram(events: List[ConjunctionEvent],
                                     filepath: str = "results/pc_histogram.png") -> None:
    """Log-scale histogram of Pc values across all conjunction events."""
    if not events:
        print("  [VIZ] No events to plot for Pc histogram.")
        return

    pcs = [ev.pc for ev in events if ev.pc > 0]
    if not pcs:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(fig, ax)

    log_pcs = np.log10(pcs)
    bins    = np.linspace(np.floor(min(log_pcs)) - 1, 0, 40)

    n, edges, patches = ax.hist(log_pcs, bins=bins, edgecolor=BG_DARK,
                                  linewidth=0.8, alpha=0.9)

    # Colour by threshold band
    for patch, left_edge in zip(patches, edges[:-1]):
        pc_val = 10 ** left_edge
        if pc_val < 1e-5:
            c = THREAT_COLORS["GREEN"]
        elif pc_val < 1e-4:
            c = THREAT_COLORS["YELLOW"]
        elif pc_val < 1e-3:
            c = THREAT_COLORS["ORANGE"]
        else:
            c = THREAT_COLORS["RED"]
        patch.set_facecolor(c)

    # Threshold lines
    for thresh, label, col in [
        (-5, "Green (1e-5)", THREAT_COLORS["GREEN"]),
        (-4, "Yellow (1e-4)", THREAT_COLORS["YELLOW"]),
        (-3, "Red (1e-3)",   THREAT_COLORS["RED"]),
    ]:
        ax.axvline(thresh, color=col, linestyle="--", alpha=0.8, linewidth=1.5)
        ax.text(thresh + 0.05, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 5,
                label, color=col, fontsize=7, rotation=90, va="top")

    ax.set_xlabel("log₁₀(Collision Probability)")
    ax.set_ylabel("Number of Events")
    ax.set_title("SpaceShield AI — Collision Probability Distribution",
                 color=TEXT_COL, fontsize=12, fontweight="bold")
    _save(fig, filepath)


# ─── 5. Risk Score Distribution ───────────────────────────────────────────────

def risk_score_distribution(scores: List[RiskScore],
                              filepath: str = "results/risk_score_distribution.png") -> None:
    """Histogram of composite risk scores with band overlays."""
    if not scores:
        return

    vals = [s.total_score for s in scores]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_style(fig, [ax1, ax2])

    # Histogram
    n, edges, patches = ax1.hist(vals, bins=20, edgecolor=BG_DARK, linewidth=0.8)
    for patch, left in zip(patches, edges[:-1]):
        band, _ = next(
            ((b, g) for lo, hi, b, g in RISK_BANDS if lo <= left < hi),
            ("MINIMAL", "A")
        )
        patch.set_facecolor(BAND_COLORS.get(band, "#888"))

    ax1.set_xlabel("Risk Score [0–100]")
    ax1.set_ylabel("Count")
    ax1.set_title("Risk Score Distribution", color=TEXT_COL, fontweight="bold")

    # Band pie
    band_counts = {}
    for s in scores:
        band_counts[s.risk_band] = band_counts.get(s.risk_band, 0) + 1

    if band_counts:
        labels = list(band_counts.keys())
        vals2  = list(band_counts.values())
        colors = [BAND_COLORS.get(b, "#888") for b in labels]
        wedges, texts, autotexts = ax2.pie(
            vals2, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=140, pctdistance=0.75,
            wedgeprops={"edgecolor": BG_DARK, "linewidth": 1.5}
        )
        for t in texts + autotexts:
            t.set_color(TEXT_COL)
            t.set_fontsize(8)
        ax2.set_title("Risk Band Distribution", color=TEXT_COL, fontweight="bold")

    fig.suptitle("SpaceShield AI — Composite Risk Analysis",
                 color=TEXT_COL, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, filepath)


# ─── 6. Maneuver Summary ─────────────────────────────────────────────────────

def maneuver_summary_chart(recs: List[ManeuverRecommendation],
                            filepath: str = "results/maneuver_summary.png") -> None:
    """Bar chart of delta-v requirements and maneuver type breakdown."""
    if not recs:
        return

    active_recs = [r for r in recs if r.delta_v_ms > 0]
    if not active_recs:
        print("  [VIZ] No active maneuvers to chart.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_style(fig, axes)

    # Δv bar chart (top-10)
    ax1 = axes[0]
    top = sorted(active_recs, key=lambda r: r.delta_v_ms, reverse=True)[:10]
    ids = [r.event_id for r in top]
    dvs = [r.delta_v_ms for r in top]
    cols = [ACCENT2 if r.urgent else ACCENT for r in top]
    bars = ax1.barh(ids, dvs, color=cols, edgecolor=BG_DARK)
    ax1.set_xlabel("Required Δv [m/s]")
    ax1.set_title("Top-10 Required Maneuvers", color=TEXT_COL, fontweight="bold")
    ax1.invert_yaxis()

    # Type pie
    ax2 = axes[1]
    type_counts = {}
    for r in active_recs:
        type_counts[r.maneuver_type] = type_counts.get(r.maneuver_type, 0) + 1

    labels = list(type_counts.keys())
    vals   = list(type_counts.values())
    pie_cols = [ACCENT, ACCENT4, ACCENT3, ACCENT2, "#a29bfe", "#fd79a8"]
    wedges, texts, autotexts = ax2.pie(
        vals, labels=labels, colors=pie_cols[:len(labels)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": BG_DARK, "linewidth": 1.5}
    )
    for t in texts + autotexts:
        t.set_color(TEXT_COL)
        t.set_fontsize(8)
    ax2.set_title("Maneuver Type Distribution", color=TEXT_COL, fontweight="bold")

    fig.suptitle("SpaceShield AI — Maneuver Recommendation Summary",
                 color=TEXT_COL, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, filepath)


# ─── 7. Conjunction Timeline ─────────────────────────────────────────────────

def conjunction_timeline(events: List[ConjunctionEvent],
                          scores: List[RiskScore],
                          filepath: str = "results/conjunction_timeline.png") -> None:
    """Scatter: Time-to-TCA vs. miss distance, coloured by risk score."""
    if not events:
        return

    score_map = {s.event_id: s.total_score for s in scores}

    lead_times = [ev.lead_time_hours for ev in events]
    miss_dists = [ev.miss_distance_km for ev in events]
    risk_vals  = [score_map.get(ev.event_id, 0) for ev in events]
    risk_levels = [ev.risk_level for ev in events]

    fig, ax = plt.subplots(figsize=(12, 7))
    _apply_dark_style(fig, ax)

    sc = ax.scatter(lead_times, miss_dists,
                    c=risk_vals, cmap="YlOrRd",
                    s=[60 + 200 * (v / 100) for v in risk_vals],
                    alpha=0.75, edgecolors=BG_DARK, linewidths=0.5,
                    vmin=0, vmax=100)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Composite Risk Score", color=TEXT_COL)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    # Threshold lines
    ax.axhline(1.0, color=ACCENT2, linestyle="--", alpha=0.7, linewidth=1.2,
               label="1 km threshold")
    ax.axvline(24.0, color=ACCENT, linestyle=":", alpha=0.7, linewidth=1.2,
               label="24 h warning")
    ax.axvline(4.0, color=ACCENT4, linestyle=":", alpha=0.7, linewidth=1.2,
               label="4 h critical")

    ax.set_xlabel("Lead Time to TCA [hours]")
    ax.set_ylabel("Miss Distance at TCA [km]")
    ax.set_title("SpaceShield AI — Conjunction Timeline\n"
                 "(Size and colour ∝ risk score)",
                 color=TEXT_COL, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, facecolor=BG_PANEL, labelcolor=TEXT_COL,
              edgecolor=GRID_COL)
    _save(fig, filepath)


# ─── 8. Feature Importance ────────────────────────────────────────────────────

def feature_importance_plot(importances: pd.Series,
                             filepath: str = "results/feature_importance.png") -> None:
    """Horizontal bar chart of ML feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(fig, ax)

    imp_sorted = importances.sort_values(ascending=True)
    colors = [ACCENT if v > imp_sorted.median() else ACCENT3
              for v in imp_sorted.values]
    ax.barh(imp_sorted.index, imp_sorted.values,
            color=colors, edgecolor=BG_DARK, linewidth=0.8)

    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("SpaceShield AI — ML Model Feature Importances",
                 color=TEXT_COL, fontsize=12, fontweight="bold")
    _save(fig, filepath)


# ─── 9. Telemetry Dashboard (6-panel) ────────────────────────────────────────

def telemetry_dashboard(objects: List[SpaceObject],
                         events: List[ConjunctionEvent],
                         scores: List[RiskScore],
                         recs: List[ManeuverRecommendation],
                         filepath: str = "results/telemetry_dashboard.png") -> None:
    """
    Master 6-panel telemetry dashboard:
      [0,0] Altitude distribution     [0,1] Object type breakdown
      [1,0] log Pc histogram           [1,1] Risk score gauge
      [2,0] Lead time histogram        [2,1] Top-10 delta-v
    """
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor(BG_DARK)

    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    axs = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    _apply_dark_style(fig, axs)

    # ── Panel 0: Altitude histogram ───────────────────────────────────────────
    ax = axs[0]
    alts = [o.altitude_km for o in objects]
    ax.hist(alts, bins=40, color=ACCENT, alpha=0.85, edgecolor=BG_DARK)
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Count")
    ax.set_title("Altitude Distribution", color=TEXT_COL, fontweight="bold")
    ax.axvline(550, color=ACCENT4, linestyle="--", alpha=0.6, linewidth=1)
    ax.text(555, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
            "Starlink belt", color=ACCENT4, fontsize=7)

    # ── Panel 1: Object type pie ──────────────────────────────────────────────
    ax = axs[1]
    type_counts = {}
    for o in objects:
        type_counts[o.object_type] = type_counts.get(o.object_type, 0) + 1
    labels = [k.replace("_", "\n") for k in type_counts.keys()]
    vals   = list(type_counts.values())
    cols   = [OBJTYPE_COLORS[k] for k in type_counts.keys()]
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=cols, autopct="%1.0f%%",
        startangle=90, wedgeprops={"edgecolor": BG_DARK, "linewidth": 1.2}
    )
    for t in texts + autotexts:
        t.set_color(TEXT_COL)
        t.set_fontsize(7)
    ax.set_title("Object Type Breakdown", color=TEXT_COL, fontweight="bold")

    # ── Panel 2: Pc histogram ─────────────────────────────────────────────────
    ax = axs[2]
    if events:
        pcs = [ev.pc for ev in events if ev.pc > 0]
        if pcs:
            log_pcs = np.log10(pcs)
            ax.hist(log_pcs, bins=20, color=ACCENT2, alpha=0.85, edgecolor=BG_DARK)
        ax.set_xlabel("log₁₀(Pc)")
        ax.set_ylabel("Events")
        ax.set_title(f"Collision Probability ({len(events)} events)",
                     color=TEXT_COL, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No conjunction\nevents", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_COL, fontsize=12)
        ax.set_title("Collision Probability", color=TEXT_COL, fontweight="bold")

    # ── Panel 3: Risk score gauge (KDE-style) ─────────────────────────────────
    ax = axs[3]
    if scores:
        risk_vals = [s.total_score for s in scores]
        ax.hist(risk_vals, bins=15, color=ACCENT4, alpha=0.85, edgecolor=BG_DARK)
        mean_risk = np.mean(risk_vals)
        ax.axvline(mean_risk, color=ACCENT2, linestyle="--", linewidth=1.5)
        ax.text(mean_risk + 1, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
                f"Mean={mean_risk:.1f}", color=ACCENT2, fontsize=8)
    ax.set_xlabel("Composite Risk Score")
    ax.set_ylabel("Events")
    ax.set_title("Risk Score Distribution", color=TEXT_COL, fontweight="bold")
    ax.set_xlim(0, 100)

    # ── Panel 4: Lead time histogram ──────────────────────────────────────────
    ax = axs[4]
    if events:
        lead_times = [ev.lead_time_hours for ev in events]
        ax.hist(lead_times, bins=20, color=ACCENT3, alpha=0.85, edgecolor=BG_DARK)
        ax.axvline(24, color=ACCENT, linestyle="--", alpha=0.7, linewidth=1.2)
        ax.axvline(4,  color=ACCENT2, linestyle="--", alpha=0.7, linewidth=1.2)
    ax.set_xlabel("Lead Time to TCA [h]")
    ax.set_ylabel("Events")
    ax.set_title("Lead Time Distribution", color=TEXT_COL, fontweight="bold")

    # ── Panel 5: Top delta-v bar ───────────────────────────────────────────────
    ax = axs[5]
    active_recs = [r for r in recs if r.delta_v_ms > 0]
    if active_recs:
        top = sorted(active_recs, key=lambda r: r.delta_v_ms, reverse=True)[:8]
        ax.barh([r.event_id for r in top],
                [r.delta_v_ms for r in top],
                color=[ACCENT2 if r.urgent else ACCENT for r in top],
                edgecolor=BG_DARK)
        ax.set_xlabel("Δv [m/s]")
        ax.invert_yaxis()
    ax.set_title("Top Maneuver Δv Requirements", color=TEXT_COL, fontweight="bold")

    # Title
    total_critical = sum(1 for e in events if e.risk_level == "RED")
    total_urgent   = sum(1 for r in recs if r.urgent)
    fig.suptitle(
        f"SpaceShield AI  —  Mission Telemetry Dashboard\n"
        f"Objects: {len(objects)}   Conjunctions: {len(events)}   "
        f"Critical: {total_critical}   Urgent Maneuvers: {total_urgent}",
        color=TEXT_COL, fontsize=14, fontweight="bold", y=0.98
    )

    _save(fig, filepath)


if __name__ == "__main__":
    from src.debris_generator import DebrisGenerator
    from src.collision_prediction import ConjunctionScreener
    from src.risk_engine import RiskEngine

    gen  = DebrisGenerator(seed=5)
    objs = gen.generate(n_active=30, n_debris=60, n_rockets=10, n_defunct=10)
    active_ids = {o.object_id for o in objs if o.active}

    screener = ConjunctionScreener(screening_threshold_km=15.0)
    events   = screener.screen(objs, duration_hours=6.0, verbose=True)

    engine = RiskEngine()
    scores = engine.score_events(events, active_ids=active_ids)

    import os; os.makedirs("results", exist_ok=True)

    orbit_plot_3d(objs, filepath="results/orbit_plot.png")
    risk_heatmap(objs, events, filepath="results/risk_heatmap.png")
    threat_distribution(events, filepath="results/threat_distribution.png")
    collision_probability_histogram(events, filepath="results/pc_histogram.png")
    risk_score_distribution(scores, filepath="results/risk_score_distribution.png")
    conjunction_timeline(events, scores, filepath="results/conjunction_timeline.png")
    telemetry_dashboard(objs, events, scores, [], filepath="results/telemetry_dashboard.png")

    print("Visualization module: OK ✓")
