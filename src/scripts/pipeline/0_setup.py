#!/usr/bin/env python3
"""
Step 0 — set up the working environment from a fresh git clone.

Performs these stages, each idempotent (skips if target already present):
  1. Download trained models -> data/models/
  2. Download prankweb tarball + extract -> {prankweb-extract dir}
  3. Determine PDB scope (CLI; defaults to all PDBs available in prankweb)
  4. Download PDB structures from RCSB -> data/input/pdb/
  5. Extract per-PDB P2Rank CSVs from prankweb -> data/input/P2Rank/

CLI:
  --pdb-list PATH       Use only PDB IDs listed in PATH (one per line)
  --pdb-list none       Skip PDB download AND P2Rank extraction
  (no flag)             Use all PDBs that have a prankweb prediction

  --models-url URL      Override default models download URL
  --prankweb-url URL    Override default prankweb tarball URL
  --prankweb-tarball P  Where to keep the downloaded tarball
  --prankweb-extract P  Where to extract the tarball
  --pdb-threads N       Parallel PDB downloads (default 8)
  --skip-models         Skip stage 1
  --skip-prankweb       Skip stages 2 (download + untar)
  --skip-pdbs           Skip stage 4
  --skip-p2r-extract    Skip stage 5

Designed to run as a single long-running sbatch job. Many hours total
on first run (200+ GB tarball download/extract + 230k PDB downloads).
"""
import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / 'data'
MODELS_DIR = DATA / 'models'
PDB_DIR = DATA / 'input' / 'pdb'
P2R_DIR = DATA / 'input' / 'P2Rank'

MODEL_FILES = ["3B-model.pt", "smoother.pt"]
MODELS_URL_DEFAULT = "https://owncloud.cesnet.cz/index.php/f/676508323"
PRANKWEB_URL_DEFAULT = "https://prankweb.cz/www/prankweb/PDBe-p2rank-2.4-conservation-hmm.tar.gz"
PRANKWEB_LOCAL_DEFAULT = Path("/scratch/tmp/lopatkao/bachelor/prankweb.tar.gz")
PRANKWEB_EXTRACT_DEFAULT = Path("/scratch/tmp/lopatkao/bachelor/prankweb")
RCSB_BASE = "https://files.rcsb.org/download"


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True)


# ---------- 1. Models ----------

def stage_models(url, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    if all((dest_dir / f).exists() for f in MODEL_FILES):
        print(f"[1/5 models] already present in {dest_dir}, skipping")
        return
    print(f"[1/5 models] target dir: {dest_dir}")
    print(f"[1/5 models] source: {url}")
    print("[1/5 models] NOTE: ownCloud /f/ URLs require auth and cannot be downloaded")
    print("            directly. Either provide a public-share URL (--models-url with")
    print("            an /index.php/s/<token>/download path) or place the model files")
    print(f"            ({', '.join(MODEL_FILES)}) into {dest_dir} manually.")
    # Try the URL anyway — works if user supplied a public share endpoint
    try:
        target = dest_dir / "models_bundle"
        with urllib.request.urlopen(url, timeout=60) as r, open(target, "wb") as out:
            shutil.copyfileobj(r, out)
        # Heuristic: if it's a zip, unpack
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as zf:
                zf.extractall(dest_dir)
            target.unlink()
            print(f"[1/5 models] downloaded and extracted into {dest_dir}")
        else:
            print(f"[1/5 models] downloaded {target} ({target.stat().st_size} bytes) — "
                  "manual extraction may be required.")
    except Exception as e:
        print(f"[1/5 models] auto-download failed ({e}); please place models manually.")


# ---------- 2. Prankweb tarball ----------

def stage_prankweb(url, tarball, extract_dir):
    if (extract_dir).exists() and any(extract_dir.iterdir()):
        print(f"[2/5 prankweb] extracted tree already present at {extract_dir}, skipping")
        return
    if not tarball.exists():
        tarball.parent.mkdir(parents=True, exist_ok=True)
        print(f"[2/5 prankweb] downloading {url}")
        run(["wget", "-c", "--tries=20", "--timeout=60", "--waitretry=10",
             "--progress=dot:giga", "-O", str(tarball), url])
    else:
        print(f"[2/5 prankweb] tarball already present at {tarball}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    pigz = shutil.which("pigz") is not None
    if pigz:
        print("[2/5 prankweb] extracting with pigz (parallel)")
        run(["tar", "--use-compress-program=pigz -dc", "-xf", str(tarball),
             "-C", str(extract_dir)])
    else:
        print("[2/5 prankweb] pigz not available, single-threaded gzip")
        run(["tar", "-xzf", str(tarball), "-C", str(extract_dir)])


def list_prankweb_pdbs(extract_dir):
    """Lowercase 4-char PDB IDs that have a prankweb.zip in the tree."""
    ids = set()
    for z in extract_dir.rglob("public/prankweb.zip"):
        ids.add(z.parent.parent.name.lower())
    return sorted(ids)


# ---------- 3. PDB downloads ----------

def stage_pdbs(pdb_ids, dest_dir, n_threads):
    if not pdb_ids:
        print("[3/5 pdb] empty list, nothing to download")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3/5 pdb] downloading {len(pdb_ids):,} PDBs with {n_threads} threads -> {dest_dir}")

    def fetch(pid):
        target = dest_dir / f"pdb{pid.lower()}.pdb"
        if target.exists() and target.stat().st_size > 0:
            return "skip"
        url = f"{RCSB_BASE}/{pid.upper()}.pdb"
        try:
            with urllib.request.urlopen(url, timeout=30) as r, open(target, "wb") as f:
                shutil.copyfileobj(r, f)
            return "ok"
        except Exception:
            target.unlink(missing_ok=True)
            return "err"

    counts = {"ok": 0, "skip": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(fetch, pid) for pid in pdb_ids]
        for i, f in enumerate(as_completed(futures), 1):
            counts[f.result()] += 1
            if i % 1000 == 0:
                print(f"  progress: {i:,}/{len(pdb_ids):,} "
                      f"(ok={counts['ok']:,} skip={counts['skip']:,} err={counts['err']:,})")
    print(f"[3/5 pdb] done: ok={counts['ok']:,} skip={counts['skip']:,} err={counts['err']:,}")


# ---------- 4. P2Rank CSV extraction ----------

def find_prankweb_zip(root, pdb_id):
    pid_upper = pdb_id.upper()
    for hash_dir in (pid_upper[1:3].lower(), pid_upper[1:3]):
        z = root / hash_dir / pid_upper / "public" / "prankweb.zip"
        if z.is_file():
            return z
    matches = list(root.rglob(f"{pid_upper}/public/prankweb.zip"))
    return matches[0] if matches else None


def extract_csv(zip_path, basename, dest):
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for entry in zf.namelist():
                if Path(entry).name == basename:
                    with zf.open(entry) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    return True
    except (zipfile.BadZipFile, OSError) as e:
        print(f"  [WARN] {zip_path}: {e}")
    return False


def stage_extract_p2r(extract_dir, pdb_ids, dest_dir):
    if not pdb_ids:
        print("[4/5 p2r] empty list, nothing to extract")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[4/5 p2r] extracting CSVs for {len(pdb_ids):,} PDBs -> {dest_dir}")
    counts = {"both": 0, "pred_only": 0, "no_zip": 0, "no_pred": 0}
    for i, pid in enumerate(pdb_ids, 1):
        z = find_prankweb_zip(extract_dir, pid)
        if z is None:
            counts["no_zip"] += 1
            continue
        if not extract_csv(z, "structure.cif_predictions.csv",
                           dest_dir / f"pdb{pid.lower()}_predictions.csv"):
            counts["no_pred"] += 1
            continue
        if extract_csv(z, "structure.cif_residues.csv",
                       dest_dir / f"pdb{pid.lower()}_residues.csv"):
            counts["both"] += 1
        else:
            counts["pred_only"] += 1
        if i % 1000 == 0:
            print(f"  progress: {i:,}/{len(pdb_ids):,}")
    print(f"[4/5 p2r] done: both={counts['both']:,} pred_only={counts['pred_only']:,} "
          f"no_zip={counts['no_zip']:,} no_pred={counts['no_pred']:,}")


# ---------- main ----------

def load_pdb_list(path):
    with open(path) as f:
        return sorted({line.strip().lower() for line in f
                       if line.strip() and not line.startswith("#")})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-list", default=None,
                    help="File with PDB IDs to download. Use 'none' to skip PDB download. "
                         "Default: all PDBs available in prankweb.")
    ap.add_argument("--models-url", default=MODELS_URL_DEFAULT)
    ap.add_argument("--prankweb-url", default=PRANKWEB_URL_DEFAULT)
    ap.add_argument("--prankweb-tarball", type=Path, default=PRANKWEB_LOCAL_DEFAULT)
    ap.add_argument("--prankweb-extract", type=Path, default=PRANKWEB_EXTRACT_DEFAULT)
    ap.add_argument("--pdb-threads", type=int, default=8)
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-prankweb", action="store_true")
    ap.add_argument("--skip-pdbs", action="store_true")
    ap.add_argument("--skip-p2r-extract", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("Pipeline setup (step 0)")
    print(f"Project root: {ROOT}")
    print("=" * 60)

    if not args.skip_models:
        stage_models(args.models_url, MODELS_DIR)

    if not args.skip_prankweb:
        stage_prankweb(args.prankweb_url, args.prankweb_tarball, args.prankweb_extract)

    if args.pdb_list == "none":
        print("[scope] --pdb-list none: skipping PDB download and P2Rank extraction")
        sys.exit(0)
    elif args.pdb_list:
        pdb_ids = load_pdb_list(Path(args.pdb_list))
        print(f"[scope] {len(pdb_ids):,} PDB IDs from {args.pdb_list}")
    else:
        if not args.prankweb_extract.exists():
            sys.exit("Cannot determine 'all available' PDBs — prankweb tree missing. "
                     "Re-run without --skip-prankweb, or pass --pdb-list explicitly.")
        pdb_ids = list_prankweb_pdbs(args.prankweb_extract)
        print(f"[scope] using all {len(pdb_ids):,} PDBs available in prankweb")

    if not args.skip_pdbs:
        stage_pdbs(pdb_ids, PDB_DIR, args.pdb_threads)
    if not args.skip_p2r_extract:
        stage_extract_p2r(args.prankweb_extract, pdb_ids, P2R_DIR)

    print("\nSetup complete.")


if __name__ == "__main__":
    main()
