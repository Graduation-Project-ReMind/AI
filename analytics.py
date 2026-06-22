import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter   
from typing import List, Optional

from gaze_tracker import GazeFrame, AttentionMetrics



COLORS = {
    "center":   "#00DC64",
    "left":     "#FF8C42",
    "right":    "#FF8C42",
    "up":       "#4FC3F7",
    "down":     "#4FC3F7",
    "blink":    "#B0BEC5",
    "unknown":  "#546E7A",
    # UI
    "bg":       "#0D1117",
    "panel":    "#161B22",
    "border":   "#30363D",
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
    "accent":   "#7C6FFF",
    # Score zones
    "score_hi": "#00DC64",
    "score_md": "#F0A500",
    "score_lo": "#FF5252",
}

DIR_LABELS = ["center", "left", "right", "up", "down", "blink", "unknown"]


def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["panel"],
        "axes.edgecolor":    COLORS["border"],
        "axes.labelcolor":   COLORS["text"],
        "xtick.color":       COLORS["subtext"],
        "ytick.color":       COLORS["subtext"],
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        COLORS["text"],
        "grid.color":        COLORS["border"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "font.family":       "DejaVu Sans",
        "legend.facecolor":  COLORS["panel"],
        "legend.edgecolor":  COLORS["border"],
        "legend.fontsize":   8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def plot_session_dashboard(session_data: List[GazeFrame],
                            metrics: AttentionMetrics,
                            output_path: str = "outputs/session_dashboard.png",
                            subject_name: str = "Subject") -> str:
    """Render a 6-panel analytical dashboard and save to output_path."""
    if not session_data:
        print("[Analytics] No session data – dashboard skipped.")
        return ""

    _apply_dark_theme()

    timestamps   = np.array([g.timestamp      for g in session_data])
    t            = timestamps - timestamps[0]
    scores       = np.array([g.attention_score for g in session_data])
    left_ratios  = np.array([g.left_ratio      for g in session_data])
    right_ratios = np.array([g.right_ratio     for g in session_data])
    vert_ratios  = np.array([g.vertical_ratio  for g in session_data])
    directions   = [g.gaze_direction           for g in session_data]

    duration_s = float(t[-1])
    attn_pct   = metrics.attention_percentage

    fig = plt.figure(figsize=(20, 13), facecolor=COLORS["bg"])
    fig.suptitle(
        f"Eye Gaze Tracking Dashboard  ·  {subject_name}",
        fontsize=22, fontweight="bold", color=COLORS["text"], y=0.975
    )

    gs = GridSpec(3, 3, figure=fig,
                    hspace=0.50, wspace=0.38,
                    left=0.055, right=0.975, top=0.935, bottom=0.055)

    ax_score   = fig.add_subplot(gs[0, :2])   
    ax_gdir    = fig.add_subplot(gs[1, :2])    
    ax_pie     = fig.add_subplot(gs[0, 2])    
    ax_ratio   = fig.add_subplot(gs[2, :2])   
    ax_focus   = fig.add_subplot(gs[1, 2])     
    ax_summary = fig.add_subplot(gs[2, 2])     

    score_colors = [
        COLORS["score_hi"] if s >= 70 else
        COLORS["score_md"] if s >= 40 else
        COLORS["score_lo"] for s in scores
    ]
    ax_score.scatter(t, scores, c=score_colors, s=5, alpha=0.55, zorder=3)
    smooth = _smooth(scores, 21)
    ax_score.plot(t, smooth, color=COLORS["accent"], lw=2.2, zorder=4, label="Smoothed")
    ax_score.axhline(70, color=COLORS["score_hi"], lw=1.0, ls="--", alpha=0.7, label="Good ≥70")
    ax_score.axhline(40, color=COLORS["score_lo"], lw=1.0, ls="--", alpha=0.7, label="Low < 40")
    ax_score.fill_between(t, scores, alpha=0.07, color=COLORS["accent"])
    ax_score.axhspan(70, 100, alpha=0.04, color=COLORS["score_hi"])
    ax_score.axhspan(40,  70, alpha=0.04, color=COLORS["score_md"])
    ax_score.axhspan(0,   40, alpha=0.04, color=COLORS["score_lo"])
    ax_score.set_ylim(-3, 103)
    ax_score.set_ylabel("Score (0–100)", fontsize=9)
    ax_score.set_title("Attention Score Over Time", fontweight="bold", fontsize=11)
    ax_score.legend(loc="upper right", framealpha=0.6)
    ax_score.grid(True)
    ax_score.tick_params(labelbottom=False)

    dir_y_vals = {d: i for i, d in enumerate(DIR_LABELS)}
    dir_y      = np.array([dir_y_vals.get(d, 6) for d in directions])
    for i in range(len(t) - 1):
        d = directions[i]
        c = COLORS.get(d, COLORS["unknown"])
        ax_gdir.plot(t[i:i+2], dir_y[i:i+2], color=c, lw=2.0, solid_capstyle="round")
    ax_gdir.set_yticks(list(dir_y_vals.values()))
    ax_gdir.set_yticklabels([d.capitalize() for d in DIR_LABELS], fontsize=9)
    ax_gdir.set_xlabel("Time (s)", fontsize=9)
    ax_gdir.set_title("Gaze Direction Timeline", fontweight="bold", fontsize=11)
    ax_gdir.grid(True, axis="x")

    dc          = metrics.direction_counts
    pie_labels  = [k for k in DIR_LABELS if dc.get(k, 0) > 0]
    pie_values  = [dc[k] for k in pie_labels]
    pie_colors  = [COLORS.get(k, "#888") for k in pie_labels]
    if pie_values:
        wedges, texts, autotexts = ax_pie.pie(
            pie_values, labels=pie_labels, colors=pie_colors,
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"linewidth": 1.8, "edgecolor": COLORS["bg"]},
            textprops={"color": COLORS["text"], "fontsize": 9},
        )
        for at in autotexts:
            at.set_color(COLORS["bg"])
            at.set_fontsize(8)
            at.set_fontweight("bold")
    ax_pie.set_title("Gaze Distribution", fontweight="bold", fontsize=11)

    ax_ratio.plot(t, left_ratios,  color="#4FC3F7", lw=1.1, alpha=0.75, label="Left Eye")
    ax_ratio.plot(t, right_ratios, color="#FF8C42", lw=1.1, alpha=0.75, label="Right Eye")
    ax_ratio.axhline(self_threshold(0.42), color="#888", lw=0.8, ls="--", alpha=0.6)
    ax_ratio.axhline(self_threshold(0.58), color="#888", lw=0.8, ls="--", alpha=0.6)
    ax_ratio.fill_between(t, 0.42, 0.58, alpha=0.07, color=COLORS["score_hi"], label="Center zone")
    ax_ratio.set_ylim(-0.03, 1.03)
    ax_ratio.set_xlabel("Time (s)", fontsize=9)
    ax_ratio.set_ylabel("Iris ratio (H)", fontsize=9)
    ax_ratio.set_title("Horizontal Gaze Ratio  (Pupil Position)", fontweight="bold", fontsize=11)
    ax_ratio.legend(loc="upper right", framealpha=0.6)
    ax_ratio.grid(True)

    categories = ["Focused\n(Center)", "Distracted\n(Off-center)"]
    durations  = [metrics.total_fixation_duration, metrics.total_distraction_duration]
    bcolors    = [COLORS["score_hi"], COLORS["score_lo"]]
    bars = ax_focus.bar(categories, durations, color=bcolors, width=0.48,
                        edgecolor=COLORS["bg"], linewidth=1.5)
    for bar, dur in zip(bars, durations):
        ax_focus.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{dur:.1f}s", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=COLORS["text"]
        )
    ax_focus.set_ylabel("Duration (s)", fontsize=9)
    ax_focus.set_title("Focus vs Distraction", fontweight="bold", fontsize=11)
    ax_focus.grid(True, axis="y")
    ax_focus.set_ylim(0, max(durations) * 1.22)

    ax_summary.axis("off")
    score_color = (COLORS["score_hi"] if attn_pct >= 70 else
                    COLORS["score_md"] if attn_pct >= 40 else
                    COLORS["score_lo"])
    final_score = session_data[-1].attention_score

    lines = [
        ("SESSION SUMMARY",                      COLORS["accent"],  13, True),
        (f"Duration      {duration_s:.1f} s",    COLORS["text"],    10, False),
        (f"Frames        {metrics.total_frames}", COLORS["text"],   10, False),
        (f"Blinks        {metrics.blink_count}",  COLORS["text"],   10, False),
        (f"Blink rate    {metrics.blink_rate_per_min:.1f} /min", COLORS["text"], 10, False),
        (f"Fixations     {metrics.fixation_count}", COLORS["text"], 10, False),
        (f"Avg fixation  {metrics.avg_fixation_duration:.2f} s", COLORS["text"], 10, False),
        (f"Saccades      {metrics.saccade_count}", COLORS["text"],  10, False),
        (f"Saccade rate  {metrics.saccade_rate:.2f} /s", COLORS["text"], 10, False),
        ("",                                     COLORS["text"],    6,  False),
        (f"Attention %   {attn_pct:.1f}%",       score_color,      12, True),
        (f"Final Score   {final_score:.0f} / 100", score_color,    12, True),
    ]
    y = 0.97
    for text, color, size, bold in lines:
        ax_summary.text(
            0.04, y, text,
            transform=ax_summary.transAxes,
            fontsize=size, color=color,
            fontweight="bold" if bold else "normal",
            verticalalignment="top",
            fontfamily="monospace",
        )
        y -= 0.082

    ax_summary.text(
        0.04, 0.03,
        "⚠ Behavioural indicators only.\n   Not a diagnostic tool.",
        transform=ax_summary.transAxes,
        fontsize=7.5, color=COLORS["subtext"],
        verticalalignment="bottom",
        fontstyle="italic",
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"[Analytics] Dashboard saved → {output_path}")
    return output_path



def plot_gaze_heatmap(session_data: List[GazeFrame],
                        output_path: str = "outputs/gaze_heatmap.png") -> str:
    """Render a smoothed 2-D gaze density heatmap."""
    if not session_data:
        return ""

    _apply_dark_theme()

    h_vals = np.array([g.left_ratio     for g in session_data])
    v_vals = np.array([g.vertical_ratio for g in session_data])

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=COLORS["bg"])
    fig.suptitle("Gaze Position Heatmap",
                    fontsize=17, fontweight="bold", color=COLORS["text"], y=0.97)

    heatmap, xe, ye = np.histogram2d(h_vals, v_vals, bins=50, range=[[0, 1], [0, 1]])
    heatmap = gaussian_filter(heatmap.T, sigma=1.5)    

    im = ax.imshow(
        heatmap,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="plasma",
        aspect="auto",
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="Dwell density", pad=0.02)

    ax.axhline(0.5, color="white", lw=0.8, alpha=0.35, ls="--")
    ax.axvline(0.5, color="white", lw=0.8, alpha=0.35, ls="--")

    center_rect = plt.Rectangle(
        (0.42, 0.38), 0.16, 0.24,
        linewidth=1.5, edgecolor="white", facecolor="none",
        linestyle="--", alpha=0.5
    )
    ax.add_patch(center_rect)
    ax.text(0.50, 0.645, "Center zone",
            ha="center", va="bottom", fontsize=8,
            color="white", alpha=0.65)

    ax.set_xlabel("Horizontal Ratio  (0 = Left  →  1 = Right)", fontsize=10)
    ax.set_ylabel("Vertical Ratio  (0 = Up  →  1 = Down)",      fontsize=10)
    ax.grid(False)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"[Analytics] Heatmap saved → {output_path}")
    return output_path


def export_json_report(session_data: List[GazeFrame],
                        metrics: AttentionMetrics,
                        output_path: str = "outputs/session_report.json",
                        subject_name: str = "Subject") -> str:
    """Export a structured JSON summary of the session."""
    if not session_data:
        return ""

    t0       = session_data[0].timestamp
    duration = session_data[-1].timestamp - t0
    scores   = [g.attention_score for g in session_data]

    report = {
        "subject":                subject_name,
        "session_start":          t0,
        "session_duration_s":     round(duration, 2),
        "total_frames":           metrics.total_frames,
        "blink_count":            metrics.blink_count,
        "blink_rate_per_min":     round(metrics.blink_rate_per_min, 2),
        "fixation_count":         metrics.fixation_count,
        "avg_fixation_duration_s":round(metrics.avg_fixation_duration, 3),
        "total_fixation_s":       round(metrics.total_fixation_duration, 2),
        "saccade_count":          metrics.saccade_count,
        "saccade_rate_per_s":     round(metrics.saccade_rate, 3),
        "total_distraction_s":    round(metrics.total_distraction_duration, 2),
        "attention_percentage":   round(metrics.attention_percentage, 1),
        "score_mean":             round(float(np.mean(scores)), 1),
        "score_std":              round(float(np.std(scores)),  1),
        "score_min":              round(float(np.min(scores)),  1),
        "score_max":              round(float(np.max(scores)),  1),
        "final_attention_score":  round(session_data[-1].attention_score, 1),
        "gaze_distribution":      metrics.direction_counts,
        "note": "This system does NOT diagnose ADHD or any other condition. "
                "Values are behavioural indicators only.",
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Analytics] JSON report saved → {output_path}")
    return output_path


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple centred moving-average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def self_threshold(v):
    """Passthrough – used so threshold constants are readable in plots."""
    return v
