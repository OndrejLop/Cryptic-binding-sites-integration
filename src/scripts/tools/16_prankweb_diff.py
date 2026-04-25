#!/usr/bin/env python3
"""
Two modes for working with the extracted prankweb archive.

--diff (default):
    Scan the tree for PDB IDs with predictions and compare against
    data/intermediate/pipeline_membership.csv. Writes three lists:
      prankweb_available.txt  all PDBs with a predictions CSV
      prankweb_new.txt        PDBs not yet in the membership CSV
      prankweb_overlap.txt    PDBs already in the membership CSV

--extract --input-list FILE:
    For every PDB ID in FILE (one per line), open
      {root}/{hash}/{PDB}/public/prankweb.zip
    read structure.cif_predictions.csv and structure.cif_residues.csv
    straight out of the zip, and write them into --dest (default
    data/input/P2Rank/) as
      pdb{id}_predictions.csv
      pdb{id}_residues.csv

Expected prankweb layout:
  {PRANKWEB_ROOT}/{hash}/{PDB_ID}/public/prankweb.zip
    (zip contains structure.cif_predictions.csv, structure.cif_residues.csv)
"""
import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MEMBERSHIP = ROOT / 'data' / 'intermediate' / 'pipeline_membership.csv'
DEFAULT_OUT = ROOT / 'data' / 'intermediate'
DEFAULT_P2R_DEST = ROOT / 'data' / 'input' / 'P2Rank'
DEFAULT_PRANKWEB = Path('/scratch/tmp/lopatkao/bachelor/prankweb/v4-conservation-hmm')


def load_membership_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        raise SystemExit(f"Membership CSV not found: {csv_path}  (run tool 15 first)")
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return {row["pdb_id"].strip().lower() for row in reader if row.get("pdb_id")}


def collect_prankweb_pdbs(root: Path) -> set[str]:
    if not root.exists():
        raise SystemExit(f"Prankweb root does not exist: {root}")
    ids = set()
    scanned = 0
    # {hash}/{PDB_ID}/public/prankweb.zip
    for z in root.glob("*/*/public/prankweb.zip"):
        ids.add(z.parent.parent.name.lower())
        scanned += 1
        if scanned % 10000 == 0:
            print(f"  scanned {scanned} prankweb.zip files...")
    return ids


def prankweb_zip_for(root: Path, pdb_id: str) -> Path | None:
    """Locate {root}/{hash}/{PDB}/public/prankweb.zip for a PDB ID.
    Tries the conventional (middle-2-chars) hash first, falls back to glob."""
    pdb_upper = pdb_id.upper()
    # Conventional prankweb hash: middle two chars, usually lowercase
    for hash_dir in (pdb_upper[1:3].lower(), pdb_upper[1:3]):
        z = root / hash_dir / pdb_upper / "public" / "prankweb.zip"
        if z.is_file():
            return z
    # Fallback: slow glob
    matches = list(root.glob(f"*/{pdb_upper}/public/prankweb.zip"))
    return matches[0] if matches else None


def extract_csv_from_zip(zip_path: Path, target_basename: str, dest: Path) -> bool:
    """Read an entry whose basename matches target_basename out of zip_path
    and write it to dest. Returns True on success."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for entry in zf.namelist():
                if Path(entry).name == target_basename:
                    with zf.open(entry) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    return True
    except (zipfile.BadZipFile, OSError) as e:
        print(f"  [WARN] {zip_path}: {e}")
    return False


def run_diff(args):
    print(f"Scanning prankweb tree: {args.prankweb_root}")
    prankweb_ids = collect_prankweb_pdbs(args.prankweb_root)
    print(f"Prankweb PDB IDs with predictions: {len(prankweb_ids)}")

    existing_ids = load_membership_ids(args.membership)
    print(f"Membership CSV rows: {len(existing_ids)}")

    new_ids = prankweb_ids - existing_ids
    overlap_ids = prankweb_ids & existing_ids

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("prankweb_available", prankweb_ids),
                      ("prankweb_new", new_ids),
                      ("prankweb_overlap", overlap_ids)]:
        path = args.output_dir / f"{name}.txt"
        with open(path, "w") as f:
            f.write("\n".join(sorted(ids)) + ("\n" if ids else ""))
        print(f"  {name:22s} {len(ids):>8d}  →  {path}")


def run_extract(args):
    if not args.input_list or not args.input_list.exists():
        sys.exit("--extract requires --input-list pointing to an existing file")

    with open(args.input_list) as f:
        pdb_ids = [line.strip().lower() for line in f
                   if line.strip() and not line.startswith("#")]
    print(f"Loaded {len(pdb_ids)} PDB IDs from {args.input_list}")
    print(f"Copying to: {args.dest}")
    args.dest.mkdir(parents=True, exist_ok=True)

    copied_both = 0
    copied_pred_only = 0
    missing_zip = 0
    missing_pred = 0

    for i, pid in enumerate(pdb_ids, 1):
        z = prankweb_zip_for(args.prankweb_root, pid)
        if z is None:
            missing_zip += 1
            continue
        got_pred = extract_csv_from_zip(
            z, "structure.cif_predictions.csv",
            args.dest / f"pdb{pid}_predictions.csv")
        if not got_pred:
            missing_pred += 1
            continue
        got_res = extract_csv_from_zip(
            z, "structure.cif_residues.csv",
            args.dest / f"pdb{pid}_residues.csv")
        if got_res:
            copied_both += 1
        else:
            copied_pred_only += 1

        if i % 1000 == 0:
            print(f"  processed {i}/{len(pdb_ids)}...")

    print(f"\nCopied {copied_both + copied_pred_only}/{len(pdb_ids)}")
    print(f"  both CSVs:         {copied_both}")
    print(f"  predictions only:  {copied_pred_only}")
    print(f"  no prankweb.zip:   {missing_zip}")
    print(f"  no predictions:    {missing_pred}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--diff", action="store_true",
                      help="Diff prankweb vs pipeline_membership.csv (default)")
    mode.add_argument("--extract", action="store_true",
                      help="Copy prediction CSVs for PDBs in --input-list into --dest")

    ap.add_argument("--prankweb-root", type=Path, default=DEFAULT_PRANKWEB,
                    help=f"Root of extracted prankweb tree (default: {DEFAULT_PRANKWEB})")
    ap.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP,
                    help=f"Pipeline membership CSV (default: {DEFAULT_MEMBERSHIP})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                    help="--diff: where to write the three lists")
    ap.add_argument("--input-list", type=Path, default=None,
                    help="--extract: file of PDB IDs (one per line)")
    ap.add_argument("--dest", type=Path, default=DEFAULT_P2R_DEST,
                    help=f"--extract: destination dir (default: {DEFAULT_P2R_DEST})")
    args = ap.parse_args()

    if args.extract:
        run_extract(args)
    else:
        run_diff(args)


if __name__ == "__main__":
    main()
