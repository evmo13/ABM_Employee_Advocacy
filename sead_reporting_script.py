
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.environ.get("SEAD_BASE_DIR", "D:\Desktop\logs")
PATTERN  = os.path.join(BASE_DIR, "sead_timeseries_*.csv")
OUT_DIR  = os.path.join(BASE_DIR, "colored_plots")
os.makedirs(OUT_DIR, exist_ok=True)

def parse_sid(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"sead_timeseries_s(\d+)\.csv$", base, flags=re.IGNORECASE)
    return f"S{m.group(1)}" if m else base

def sid_num(sid: str) -> int:
    m = re.search(r"S(\d+)", sid, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 9999

def fmt_num(v) -> str:
    try:
        return f"{int(round(float(v))):,}".replace(",", " ")
    except Exception:
        return str(v)

def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("t").reset_index(drop=True)

files = sorted(glob.glob(PATTERN))
if not files:
    raise SystemExit(f"No files matched: {PATTERN}")

scenarios = sorted([parse_sid(f) for f in files], key=sid_num)
data = {}
for f in files:
    sid = parse_sid(f)
    df = pd.read_csv(f)
    if "t" not in df.columns:
        raise ValueError(f"Missing 't' column in: {f}")
    data[sid] = ensure_sorted(df)

# Unique, consistent color per scenario across all plots
cmap = plt.get_cmap("tab20", len(scenarios))
scenario_colors = {sid: cmap(i) for i, sid in enumerate(scenarios)}

def plot_lines(metric_key: str, title: str, ylabel: str, filename: str, cumulative: bool=False):
    plt.figure(figsize=(12, 6.5))
    plotted = 0
    for sid in scenarios:
        df = data[sid]
        if metric_key not in df.columns:
            continue
        x = df["t"].values
        y = df[metric_key].astype(float).values
        if cumulative:
            y = np.cumsum(y)
        if np.all(np.isnan(y)):
            continue
        plt.plot(x, y, label=sid, color=scenario_colors[sid], linewidth=2.0, alpha=0.9)
        plotted += 1
    plt.title(title, fontsize=18)
    plt.xlabel("Βήμα (t)", fontsize=12); plt.ylabel(ylabel, fontsize=12)
    if plotted > 0:
        plt.legend(loc="best", ncol=2, fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def plot_sead(state_col: str, label: str, filename: str):
    plt.figure(figsize=(12, 6.5))
    plotted = 0
    for sid in scenarios:
        df = data[sid]
        if state_col not in df.columns:
            continue
        plt.plot(df["t"].values, df[state_col].astype(float).values, label=sid, color=scenario_colors[sid], linewidth=2.0, alpha=0.9)
        plotted += 1
    plt.title(f"SEAD: {label} ανά βήμα (όλα τα σενάρια)", fontsize=18)
    plt.xlabel("Βήμα (t)", fontsize=12); plt.ylabel(label, fontsize=12)
    if plotted > 0:
        plt.legend(loc="best", ncol=2, fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def plot_bars(col: str, title: str, ylabel: str, filename: str):
    # Table of finals
    rows = []
    for sid in scenarios:
        df = data[sid]
        val = float(df[col].iloc[-1]) if col in df.columns else np.nan
        rows.append({"scenario": sid, col: val})
    tab = pd.DataFrame(rows)

    plt.figure(figsize=(12, 6.5))
    colors = [scenario_colors[sid] for sid in tab["scenario"]]
    bars = plt.bar(tab["scenario"], tab[col], color=colors, edgecolor="black", linewidth=0.5)
    plt.title(title, fontsize=18)
    plt.xlabel("Σενάριο", fontsize=12); plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    if not np.isnan(tab[col]).all():
        ymax = np.nanmax(tab[col]) * 1.10
        plt.ylim(top=ymax)
    # Labels on bars
    for rect, val in zip(bars, tab[col].values):
        if np.isnan(val): 
            continue
        x = rect.get_x() + rect.get_width()/2
        y = rect.get_height()
        plt.annotate(fmt_num(val), (x, y), textcoords="offset points", xytext=(0, 4),
                     ha="center", va="bottom", fontsize=11)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path

# ---- Generate all plots ----
# Shares
plot_lines("shares", "Διάδοση: Shares ανά βήμα (όλα τα σενάρια)", "Shares/βήμα",
           "shares_per_step_all_colored.png", cumulative=False)
plot_lines("shares", "Σωρευτική διάδοση: Shares (όλα τα σενάρια)", "Cumulative Shares",
           "shares_cumulative_all_colored.png", cumulative=True)

# Exposures
plot_lines("exposures", "Exposures ανά βήμα (όλα τα σενάρια)", "Exposures/βήμα",
           "exposures_per_step_all_colored.png", cumulative=False)
plot_lines("exposures", "Σωρευτικά Exposures (όλα τα σενάρια)", "Cumulative Exposures",
           "exposures_cumulative_all_colored.png", cumulative=True)

# SEAD
plot_sead("S", "S (Susceptible)", "sead_S_all_colored.png")
plot_sead("E", "E (Exposed)",    "sead_E_all_colored.png")
plot_sead("A", "A (Active)",     "sead_A_all_colored.png")
plot_sead("D", "D (Dormant)",    "sead_D_all_colored.png")

# Unique reach & Brand strength
plot_lines("unique_reach", "Unique reach ανά βήμα (όλα τα σενάρια)", "Unique reach",
           "unique_reach_all_colored.png", cumulative=False)
plot_lines("brand_strength", "Brand strength ανά βήμα (όλα τα σενάρια)", "Brand strength",
           "brand_strength_all_colored.png", cumulative=False)

# Final bars with labels
plot_bars("unique_reach", "Τελικό Unique reach στο βήμα 50 (όλα τα σενάρια)",
          "Unique reach (τελικό)", "unique_reach_final_bar_colored.png")
plot_bars("brand_strength", "Τελικό Brand strength στο βήμα 50 (όλα τα σενάρια)",
          "Brand strength (τελικό)", "brand_strength_final_bar_colored.png")

print("OK - Plots saved to:", OUT_DIR)
