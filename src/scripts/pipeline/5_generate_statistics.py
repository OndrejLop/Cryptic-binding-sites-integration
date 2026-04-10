"""
Generate statistics and plots from pipeline results.

This script analyzes outputs from previous pipeline steps:
1. Pipeline funnel: data attrition across steps (requires manual input)
2. Per-method pocket statistics: pocket count, size, and score distributions
4. Novel pocket counts per method
6. Summary table

Input:
  - P2Rank predictions:    data/input/P2Rank/{pdb_id}_predictions.csv
  - CryptoSite predictions: data/output/CS_predictions/{pdb_id}_predictions.csv
  - Comparison results:    data/output/results/novel_cs_pockets.csv
                           data/output/results/p2r_unique_pockets.csv
  - Clustering skip log:   data/output/CS_predictions/skipped_clustering.txt (optional)

Output:
  - data/output/analysis/summary.txt
  - data/output/analysis/plots/*.png
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent.parent

P2RANK_DIR     = ROOT / 'data' / 'input' / 'P2Rank'
CS_DIR         = ROOT / 'data' / 'output' / 'CS_predictions'
RESULTS_DIR    = ROOT / 'data' / 'output' / 'results'
PDB_DIR        = ROOT / 'data' / 'input' / 'pdb'
FASTA_DIR      = ROOT / 'data' / 'intermediate' / 'fastas'
PRED_DIR       = ROOT / 'data' / 'intermediate' / 'predictions'
STATS_DIR      = ROOT / 'data' / 'output' / 'analysis'
PLOTS_DIR      = STATS_DIR / 'plots'

STATS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================

def count_files(directory, pattern):
    """Count files matching a glob pattern in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))

def load_pocket_csv(csv_path):
    """Load a pocket predictions CSV and parse residue lists."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["residue_list"] = df["residue_ids"].apply(
        lambda x: x.strip().split() if pd.notna(x) else []
    )
    df["pocket_size"] = df["residue_list"].apply(len)
    return df

def collect_pocket_stats(predictions_dir, file_pattern):
    """
    Collect pocket statistics from all prediction CSV files in a directory.

    Returns:
        pockets_per_protein: list of pocket counts per protein
        pocket_sizes: list of all pocket sizes (residue count)
        pocket_scores: list of all pocket scores
    """
    pockets_per_protein = []
    pocket_sizes = []
    pocket_scores = []

    for csv_path in sorted(predictions_dir.glob(file_pattern)):
        try:
            df = load_pocket_csv(csv_path)
        except Exception:
            continue
        pockets_per_protein.append(len(df))
        pocket_sizes.extend(df["pocket_size"].tolist())
        if "score" in df.columns:
            pocket_scores.extend(df["score"].astype(float).tolist())

    return pockets_per_protein, pocket_sizes, pocket_scores

# ============================================================
# 1. Pipeline funnel
# ============================================================

def pipeline_funnel(out):
    out.write("=" * 60 + "\n")
    out.write("1. PIPELINE FUNNEL\n")
    out.write("=" * 60 + "\n\n")

    n_pdb = count_files(PDB_DIR, "*.pdb")
    n_fasta = count_files(FASTA_DIR, "*.fasta")
    n_predictions = count_files(PRED_DIR, "*_predictions.csv")
    n_cs_pockets = count_files(CS_DIR, "*_predictions.csv")
    n_p2r_pockets = count_files(P2RANK_DIR, "*_predictions.csv")

    out.write(f"  PDB files:                     {n_pdb}\n")
    out.write(f"  Extracted FASTA sequences:     {n_fasta}\n")
    out.write(f"  ESM2 predictions (chains):     {n_predictions}\n")
    out.write(f"  CryptoSite clustered (PDBs):   {n_cs_pockets}\n")
    out.write(f"  P2Rank predictions (PDBs):     {n_p2r_pockets}\n")

    # Try to read skipped_clustering.txt for attrition details
    skip_file = CS_DIR / "skipped_clustering.txt"
    if skip_file.exists():
        out.write(f"\n  Clustering skip breakdown ({skip_file.name}):\n")
        with open(skip_file) as f:
            for line in f:
                out.write(f"    {line.rstrip()}\n")

    out.write("\n")

    return {
        "n_pdb": n_pdb,
        "n_fasta": n_fasta,
        "n_predictions": n_predictions,
        "n_cs_pockets": n_cs_pockets,
        "n_p2r_pockets": n_p2r_pockets,
    }

def plot_funnel(funnel, out_path):
    stages = ["PDB files", "FASTA seqs", "ESM2 preds\n(chains)", "CS pockets\n(PDBs)", "P2R pockets\n(PDBs)"]
    counts = [funnel["n_pdb"], funnel["n_fasta"], funnel["n_predictions"],
              funnel["n_cs_pockets"], funnel["n_p2r_pockets"]]

    # Only plot stages that have data
    valid = [(s, c) for s, c in zip(stages, counts) if c > 0]
    if not valid:
        return
    stages, counts = zip(*valid)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(stages)), counts, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Pipeline Funnel: Data Attrition Across Steps", fontsize=14)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ============================================================
# 2. Per-method pocket statistics
# ============================================================

def per_method_stats(out):
    out.write("=" * 60 + "\n")
    out.write("2. PER-METHOD POCKET STATISTICS\n")
    out.write("=" * 60 + "\n\n")

    results = {}
    for method, directory, pattern in [
        ("CryptoSite", CS_DIR, "*_predictions.csv"),
        ("P2Rank", P2RANK_DIR, "*_predictions.csv"),
    ]:
        pockets_per_protein, pocket_sizes, pocket_scores = collect_pocket_stats(directory, pattern)

        if not pockets_per_protein:
            out.write(f"  {method}: no data found\n\n")
            results[method] = None
            continue

        out.write(f"  --- {method} ---\n")
        out.write(f"  Proteins analyzed:       {len(pockets_per_protein)}\n")
        out.write(f"  Total pockets:           {sum(pockets_per_protein)}\n")
        out.write(f"  Pockets per protein:\n")
        out.write(f"    Mean:   {np.mean(pockets_per_protein):.2f}\n")
        out.write(f"    Median: {np.median(pockets_per_protein):.0f}\n")
        out.write(f"    Min:    {np.min(pockets_per_protein)}\n")
        out.write(f"    Max:    {np.max(pockets_per_protein)}\n")
        out.write(f"  Pocket size (residues):\n")
        out.write(f"    Mean:   {np.mean(pocket_sizes):.2f}\n")
        out.write(f"    Median: {np.median(pocket_sizes):.0f}\n")
        out.write(f"    Min:    {np.min(pocket_sizes)}\n")
        out.write(f"    Max:    {np.max(pocket_sizes)}\n")
        if pocket_scores:
            out.write(f"  Pocket score:\n")
            out.write(f"    Mean:   {np.mean(pocket_scores):.4f}\n")
            out.write(f"    Median: {np.median(pocket_scores):.4f}\n")
            out.write(f"    Min:    {np.min(pocket_scores):.4f}\n")
            out.write(f"    Max:    {np.max(pocket_scores):.4f}\n")
        out.write("\n")

        results[method] = {
            "pockets_per_protein": pockets_per_protein,
            "pocket_sizes": pocket_sizes,
            "pocket_scores": pocket_scores,
        }

    return results

def plot_pocket_distributions(results, out_dir):
    for method, data in results.items():
        if data is None:
            continue

        label = method.lower().replace(" ", "_")

        # Pockets per protein histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        max_val = max(data["pockets_per_protein"])
        bins = range(0, max_val + 2)
        ax.hist(data["pockets_per_protein"], bins=bins, color='steelblue',
                edgecolor='black', alpha=0.8, align='left')
        ax.set_xlabel("Number of Pockets", fontsize=12)
        ax.set_ylabel("Number of Proteins", fontsize=12)
        ax.set_title(f"{method}: Pockets per Protein", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / f"{label}_pockets_per_protein.png", dpi=150)
        plt.close(fig)

        # Pocket size histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data["pocket_sizes"], bins=50, color='coral',
                edgecolor='black', alpha=0.8)
        ax.set_xlabel("Pocket Size (residues)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{method}: Pocket Size Distribution", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / f"{label}_pocket_sizes.png", dpi=150)
        plt.close(fig)

        # Pocket score histogram
        if data["pocket_scores"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data["pocket_scores"], bins=50, color='mediumpurple',
                    edgecolor='black', alpha=0.8)
            ax.set_xlabel("Pocket Score", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"{method}: Pocket Score Distribution", fontsize=14)
            plt.tight_layout()
            fig.savefig(out_dir / f"{label}_pocket_scores.png", dpi=150)
            plt.close(fig)

    # Combined comparison: pockets per protein
    valid = {m: d for m, d in results.items() if d is not None}
    if len(valid) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, (method, data) in zip(axes, valid.items()):
            max_val = max(data["pockets_per_protein"])
            bins = range(0, max_val + 2)
            ax.hist(data["pockets_per_protein"], bins=bins, color='steelblue',
                    edgecolor='black', alpha=0.8, align='left')
            ax.set_xlabel("Number of Pockets", fontsize=12)
            ax.set_title(method, fontsize=13)
        axes[0].set_ylabel("Number of Proteins", fontsize=12)
        fig.suptitle("Pockets per Protein: Method Comparison", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(out_dir / "comparison_pockets_per_protein.png", dpi=150)
        plt.close(fig)

        # Combined: pocket size boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        box_data = [d["pocket_sizes"] for d in valid.values()]
        bp = ax.boxplot(box_data, labels=list(valid.keys()), patch_artist=True)
        colors = ['steelblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Pocket Size (residues)", fontsize=12)
        ax.set_title("Pocket Size Comparison", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / "comparison_pocket_sizes_boxplot.png", dpi=150)
        plt.close(fig)

# ============================================================
# 4. Novel pocket counts
# ============================================================

def novel_pocket_stats(out):
    out.write("=" * 60 + "\n")
    out.write("4. NOVEL POCKET COUNTS\n")
    out.write("=" * 60 + "\n\n")

    results = {}
    for label, filename in [
        ("CryptoSite-unique", "novel_cs_pockets.csv"),
        ("P2Rank-unique", "novel_p2r_pockets.csv"),
    ]:
        csv_path = RESULTS_DIR / filename
        if not csv_path.exists():
            # Try alternate naming
            alt = RESULTS_DIR / filename.replace("novel_", "").replace("pockets", "unique_pockets")
            if alt.exists():
                csv_path = alt
            else:
                out.write(f"  {label}: file not found ({filename})\n")
                results[label] = None
                continue

        df = pd.read_csv(csv_path)
        n_proteins = len(df)
        # Each row has space-separated pocket numbers in 'pockets' column
        total_pockets = sum(len(str(row).split()) for row in df["pockets"])
        sizes_flat = []
        for row in df["sizes"]:
            sizes_flat.extend([int(x) for x in str(row).split()])

        out.write(f"  --- {label} ---\n")
        out.write(f"  Proteins with novel pockets: {n_proteins}\n")
        out.write(f"  Total novel pockets:         {total_pockets}\n")
        if sizes_flat:
            out.write(f"  Novel pocket size (residues):\n")
            out.write(f"    Mean:   {np.mean(sizes_flat):.2f}\n")
            out.write(f"    Median: {np.median(sizes_flat):.0f}\n")
            out.write(f"    Min:    {np.min(sizes_flat)}\n")
            out.write(f"    Max:    {np.max(sizes_flat)}\n")
        out.write("\n")

        results[label] = {
            "n_proteins": n_proteins,
            "total_pockets": total_pockets,
            "sizes": sizes_flat,
        }

    return results

def plot_novel_pockets(results, out_dir):
    # Bar chart: total novel pockets per method
    valid = {m: d for m, d in results.items() if d is not None}
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(valid.keys())
    counts = [d["total_pockets"] for d in valid.values()]
    colors = ['steelblue', 'coral'][:len(methods)]
    bars = ax.bar(methods, counts, color=colors, edgecolor='black')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Novel Pockets", fontsize=12)
    ax.set_title("Novel Pockets by Method", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "novel_pockets_count.png", dpi=150)
    plt.close(fig)

    # Novel pocket size distributions
    has_sizes = {m: d for m, d in valid.items() if d["sizes"]}
    if has_sizes:
        fig, ax = plt.subplots(figsize=(8, 5))
        box_data = [d["sizes"] for d in has_sizes.values()]
        bp = ax.boxplot(box_data, labels=list(has_sizes.keys()), patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(has_sizes)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Pocket Size (residues)", fontsize=12)
        ax.set_title("Novel Pocket Size Distribution", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / "novel_pocket_sizes_boxplot.png", dpi=150)
        plt.close(fig)

# ============================================================
# 6. Summary table
# ============================================================

def summary_table(funnel, method_stats, novel_stats, out):
    out.write("=" * 60 + "\n")
    out.write("6. SUMMARY TABLE\n")
    out.write("=" * 60 + "\n\n")

    header = f"{'Metric':<40} {'CryptoSite':>12} {'P2Rank':>12}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    def row(label, cs_val, p2r_val):
        cs_str = str(cs_val) if cs_val is not None else "N/A"
        p2r_str = str(p2r_val) if p2r_val is not None else "N/A"
        out.write(f"{label:<40} {cs_str:>12} {p2r_str:>12}\n")

    cs = method_stats.get("CryptoSite")
    p2r = method_stats.get("P2Rank")

    row("Proteins analyzed",
        len(cs["pockets_per_protein"]) if cs else None,
        len(p2r["pockets_per_protein"]) if p2r else None)
    row("Total pockets",
        sum(cs["pockets_per_protein"]) if cs else None,
        sum(p2r["pockets_per_protein"]) if p2r else None)
    row("Mean pockets/protein",
        f"{np.mean(cs['pockets_per_protein']):.2f}" if cs else None,
        f"{np.mean(p2r['pockets_per_protein']):.2f}" if p2r else None)
    row("Median pocket size (residues)",
        f"{np.median(cs['pocket_sizes']):.0f}" if cs else None,
        f"{np.median(p2r['pocket_sizes']):.0f}" if p2r else None)
    row("Mean pocket size (residues)",
        f"{np.mean(cs['pocket_sizes']):.2f}" if cs else None,
        f"{np.mean(p2r['pocket_sizes']):.2f}" if p2r else None)

    cs_novel = novel_stats.get("CryptoSite-unique")
    p2r_novel = novel_stats.get("P2Rank-unique")
    row("Novel pockets (total)",
        cs_novel["total_pockets"] if cs_novel else None,
        p2r_novel["total_pockets"] if p2r_novel else None)
    row("Proteins with novel pockets",
        cs_novel["n_proteins"] if cs_novel else None,
        p2r_novel["n_proteins"] if p2r_novel else None)

    out.write("\n")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys

    summary_path = STATS_DIR / "summary.txt"
    with open(summary_path, 'w') as out:
        out.write("Pipeline Statistics Report\n")
        out.write(f"Generated from: {ROOT}\n\n")

        # 1. Funnel
        funnel = pipeline_funnel(out)
        plot_funnel(funnel, PLOTS_DIR / "pipeline_funnel.png")

        # 2. Per-method stats
        method_stats = per_method_stats(out)
        plot_pocket_distributions(method_stats, PLOTS_DIR)

        # 4. Novel pockets
        novel_stats = novel_pocket_stats(out)
        plot_novel_pockets(novel_stats, PLOTS_DIR)

        # 6. Summary table
        summary_table(funnel, method_stats, novel_stats, out)

    print(f"Statistics written to: {summary_path}")
    print(f"Plots saved to:       {PLOTS_DIR}")

    # Print summary to stdout as well
    with open(summary_path) as f:
        print(f.read())

# ============================================================
# SUGGESTIONS TODO
# ============================================================
#
# 3. METHOD AGREEMENT (requires matched PDB IDs between methods)
#    - Per protein: count overlapping vs unique pockets
#    - Jaccard similarity of residue sets for matched pockets
#    - Fraction of proteins where both methods agree on >= 1 pocket
#    - Heatmap of agreement scores across proteins
#
# 5. RESIDUE-LEVEL ANALYSIS (requires _residues.csv from step 3)
#    - Distribution of binding probabilities across all residues
#    - Fraction of residues predicted as binding per protein
#    - Correlation between binding fraction and protein length
#    - ROC / PR curves if ground truth labels available
#
# Additional ideas:
#    - Pocket score vs pocket size scatter plot
#    - Per-chain analysis (multi-chain proteins)
#    - Protein length vs number of pockets correlation
#    - Export statistics as LaTeX tables for thesis
