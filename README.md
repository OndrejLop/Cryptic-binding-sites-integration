# Protein-Ligand Binding Site Prediction Pipeline

**Integration of sequence and structure-based approaches for protein-ligand binding site prediction**

This repository contains a complete bioinformatics pipeline for predicting protein-ligand binding sites using fine-tuned cryptic Seq2Pocket ESM2 model and P2Rank and comparison of the predictions from both methods. 

## Features

✅ **Structure-based prediction of binding sites** using P2Rank framework
✅ **Sequence-based prediction of cryptic binding sites** using cryptic Seq2Pocket
✅ **Comparative analysis** of predictions from both methods
✅ **PyMOL visualization scripts and statistics** summary statistics as well as individual results

## Quick Start

### Requirements

- Python 3.12+
- Java 17-24
- PyTorch with CUDA support (GPU recommended for predictions)
- 8 GB of free storage + additional storage for your files

### Installation and setup

```bash
# Clone repository
git clone https://github.com/OndrejLop/Cryptic-binding-sites-integration
cd Cryptic-binding-sites-integration

# Install dependencies
pip install -r src/utilities/requirements.txt

# Run initial setup
python src/scripts/pipeline/0_setup.py
```

Place your proteins of interest in PDB format into path ```data/input/pdb/```

### Usage

To run the predictions, comparison and to generate statistics, you need to run the pipeline in consecutive steps 1-5 located in ```src/scripts/pipeline```

**Individual steps:**
```bash
# Filter sequences by length (optional)
python scripts/00_filter_fastas.py

# Clustering with custom parameters, timestamped output
python src/scripts/03_cluster_pockets.py \
  --decision-threshold 0.65 \
  --distance-threshold 12 \
  --timestamp

# Compare with flexible overlap threshold
python src/scripts/04_compare_pockets.py \
  --max-overlap-residues 3 \
  --max-overlap-percent 25 \
  --timestamp
```

## Data Structure

```
data/                        # Directory for all processed data
├── input/                       # Inputs for processing
│   ├── pdb/                       # Input PDB protein structures
│   └── P2Rank/                    # P2Rank predictions for comparison
├── intermediate/                # Files that are needed along the way (do not touch)
│   ├── fastas/                    # FASTA sequences
│   ├── predictions/               # ESM2 binding probabilities (per-residue)
│   ├── embeddings/                # ESM2 token embeddings (1024-dim)
│   ├── p2rank_dataset.ds          # List of files to be predicted
│   └── pipeline_membership.csv    # Matrix of processed files in each step
├── models/                      
│   ├── 3B-model.pt                # Fine-tuned ESM2 3B model
│   └── smoother.pt                # Smoothing refinement model
└── output/                      # Resulting data (you want)
   ├── Seq2Pockets/                # Final pocket predictions
   ├── analysis/                   # Statistics of the compared results
   └── results/                    # Comparison results and pymol for visualization
```

## Pipeline Description

### Step 1: Extract Sequences
Extracts FASTA sequences for Seq2Pocket prediction and list of files for P2Rank prediction

### Step 2: Run Predictions
Runs predictions for both methods

### Step 3: Cluster Pockets
Clusters the Seq2Pocket predictions into pockets

**Algorithm:**
- Smoothing: refine predictions using local context
- Clustering: MeanShift on 3D surface points (Fibonacci lattice)
- Voting: propagate cluster labels from points → atoms → residues

**Configurable parameters:**
- `--decision-threshold`: Binding probability cutoff (default: 0.7)
- `--distance-threshold`: Neighbor search distance in Å (default: 10)
- `--timestamp`: Prevent output overwrites (optional)

### Step 4: Compare Pockets
Compares predictions from both methods to obtain uniquely predicted pockets

**Configurable parameters:**
- `--max-overlap-residues`: Max shared residues for "unique" pocket (default: 0)
- `--max-overlap-percent`: Max overlap % relative to smaller pocket (default: 0)
- `--timestamp`: Prevent output overwrites (optional)

### Step 5: Generate statistics
Generates summary statistics for the results

**Configurable parameters:**
- `--results-dir`: Specific comparison-results subdirectory under data/output/results/ if you run with multiple settings
- `--exclude-file`: List of files to exclude from the analysis
- `--timestamp`: Prevent output overwrites (optional)
