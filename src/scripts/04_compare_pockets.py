import pandas as pd
from pathlib import Path

def load_pockets(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["residue_set"] = df["residue_ids"].apply(
        lambda x: frozenset(x.strip().split()) if pd.notna(x) else frozenset()
    )
    return df

def find_unmatched(source_df, target_df):
    """Returns rows from source_df that have no overlap with any pocket in target_df."""
    unmatched = []
    for _, row in source_df.iterrows():
        has_match = any(row["residue_set"] & t["residue_set"] for _, t in target_df.iterrows())
        if not has_match:
            unmatched.append(row)
    return pd.DataFrame(unmatched) if unmatched else pd.DataFrame()

def save_unmatched(df, out_path):
    if df.empty:
        return
    out_path.mkdir(parents=True, exist_ok=True)
    out = df[["name", "residue_ids"]].copy()
    out.columns = ["pocket", "residue_ids"]
    out.to_csv(out_path / "unmatched_pockets.csv", index=False)

PYMOL_COLORS = ["red", "blue", "green", "yellow", "magenta", "cyan", "orange", "violet", "salmon", "limon"]

def residue_ids_to_selection(residue_ids_str):
    """Convert 'A_101 B_45' to PyMOL selection string."""
    parts = []
    for res in residue_ids_str.strip().split():
        chain, resi = res.split("_")
        parts.append(f"(chain {chain} and resi {resi})")
    return " or ".join(parts)

def write_pymol_script(pdb_id, unmatched_df, pdb_dir, out_path, source_label):
    pdb_files = list(pdb_dir.glob(f"{pdb_id}*"))
    pdb_file  = pdb_files[0] if pdb_files else pdb_dir / f"{pdb_id}.pdb"
    lines = [
        f"load {pdb_file}, {pdb_id}",
        "hide everything",
        "show cartoon, all",
        "color grey80, all",
        "",
    ]
    sel_names = []
    for i, (_, row) in enumerate(unmatched_df.iterrows()):
        num   = ''.join(filter(str.isdigit, str(row["name"])))
        sname = f"{source_label}_pocket{num}"
        color = PYMOL_COLORS[i % len(PYMOL_COLORS)]
        sel   = residue_ids_to_selection(str(row["residue_ids"]))
        lines.append(f"select {sname}, {pdb_id} and ({sel})")
        lines.append(f"color {color}, {sname}")
        lines.append(f"show sticks, {sname}")
        lines.append("")
        sel_names.append(sname)
    if sel_names:
        all_sel = " or ".join(sel_names)
        lines.append(f"zoom {all_sel}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

# --- Paths ---
ROOT         = Path(__file__).parent.parent.parent
p2rank_dir   = ROOT / 'data' / 'input' / 'P2Rank'
cs_dir       = ROOT / 'data' / 'output' / 'CS_predictions'
pdb_dir      = ROOT / 'data' / 'input' / 'pdb'
out_base_dir = ROOT / 'data' / 'output' / 'results'


cs_log_rows  = []
p2r_log_rows = []

for p2r_csv in p2rank_dir.glob("*_predictions.csv"):
    pdb_id = p2r_csv.stem.replace('_predictions', '')
    cs_csv = cs_dir / f'{pdb_id}_predictions.csv'

    if not cs_csv.exists():
        print(f"Missing CryptoSite file for {pdb_id}, skipping.")
        continue

    p2r_df = load_pockets(p2r_csv)
    cs_df  = load_pockets(cs_csv)

    # Unmatched pockets
    p2r_unmatched = find_unmatched(p2r_df, cs_df)
    cs_unmatched  = find_unmatched(cs_df,  p2r_df)

    if p2r_unmatched.empty and cs_unmatched.empty:
        print(f"{pdb_id}: all pockets matched, no output.")
        continue

    save_unmatched(p2r_unmatched, out_base_dir / pdb_id / "p2r")
    save_unmatched(cs_unmatched,  out_base_dir / pdb_id / "cs")
    print(f"{pdb_id}: saved {len(p2r_unmatched)} unmatched P2R and {len(cs_unmatched)} unmatched CS pockets.")

    def pocket_number(name):
        return ''.join(filter(str.isdigit, str(name)))

    if not cs_unmatched.empty:
        cs_log_rows.append({
            "pdb_id":  pdb_id,
            "pockets": " ".join(pocket_number(r["name"]) for _, r in cs_unmatched.iterrows()),
            "sizes":   " ".join(str(len(r["residue_set"])) for _, r in cs_unmatched.iterrows()),
        })
        write_pymol_script(pdb_id, cs_unmatched, pdb_dir,
                           out_base_dir / pdb_id / "cs_novel.pml", "cs")
    if not p2r_unmatched.empty:
        p2r_log_rows.append({
            "pdb_id":  pdb_id,
            "pockets": " ".join(pocket_number(r["name"]) for _, r in p2r_unmatched.iterrows()),
            "sizes":   " ".join(str(len(r["residue_set"])) for _, r in p2r_unmatched.iterrows()),
        })
        write_pymol_script(pdb_id, p2r_unmatched, pdb_dir,
                           out_base_dir / pdb_id / "p2r_novel.pml", "p2r")

out_base_dir.mkdir(parents=True, exist_ok=True)
if cs_log_rows:
    pd.DataFrame(cs_log_rows).to_csv(out_base_dir / "novel_cs_pockets.csv", index=False)
    print(f"\nNovel CS pockets saved -> {out_base_dir / 'novel_cs_pockets.csv'}")
if p2r_log_rows:
    pd.DataFrame(p2r_log_rows).to_csv(out_base_dir / "p2r_unique_pockets.csv", index=False)
    print(f"Novel P2R pockets saved -> {out_base_dir / 'p2r_unique_pockets.csv'}")