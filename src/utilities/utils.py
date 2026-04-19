"""
Shared utilities for the Seq2Pocket pipeline.

Contains:
- SASA surface-point clustering (MeanShift / BayesianGMM) used by step 3
- CryptoBenchClassifier: smoothing model architecture

## COPYRIGHT NOTICE
Parts of this module were inspired by https://github.com/luk27official/cryptoshow-benchmark/
See LICENSE for details.
"""
import numpy as np
from collections import Counter
from sklearn.cluster import MeanShift
from sklearn.mixture import BayesianGaussianMixture
import torch.nn as nn

# --- SASA / clustering constants ---
POINTS_DENSITY_PER_ATOM = 50
PROBE_RADIUS = 1.6

aal_prot = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "HIE", "HID", "HIP", "LYN", "CYX", "CYM", "TYM"
}


def cluster_atoms_by_surface(all_points, point_to_atom_map, eps=1.5, gmm=False):
    """
    Cluster surface points using MeanShift or Gaussian Mixture Model.

    Process:
    1. Apply clustering algorithm to 3D surface points
    2. Use majority voting to assign each atom a cluster label
    3. Return atom ID -> cluster label mapping

    Args:
        all_points (np.array): (N, 3) array of 3D surface point coordinates
        point_to_atom_map (np.array): (N,) array mapping each point to atom serial number
        eps (float): MeanShift bandwidth or DBSCAN epsilon in Angstroms (default: 1.5)
        gmm (bool): Use Gaussian Mixture Model instead of MeanShift

    Returns:
        atom_labels (dict): {atom_id: cluster_label} mapping each atom to cluster
    """
    if not gmm:
        clustering = MeanShift(bandwidth=eps, bin_seeding=True, n_jobs=-1)
        clustering.fit(all_points)
        point_labels = clustering.labels_
    else:
        bgmm = BayesianGaussianMixture(
            n_components=max(len(all_points), 1) - 1,
            random_state=42,
            covariance_type='spherical',
        )
        bgmm.fit(all_points)

        active_clusters = sum(bgmm.weights_ > 0.1)
        clustering = BayesianGaussianMixture(
            n_components=max(active_clusters, 1),
            random_state=42,
            covariance_type='spherical',
        )
        point_labels = clustering.fit_predict(all_points)

    atom_labels = {}
    unique_atoms = np.unique(point_to_atom_map)
    for atom_id in unique_atoms:
        indices = np.where(point_to_atom_map == atom_id)[0]
        current_labels = point_labels[indices]
        counts = Counter(current_labels)
        majority_label = counts.most_common(1)[0][0]
        atom_labels[atom_id] = majority_label

    return atom_labels


# --- Smoothing model ---
SMOOTHING_DECISION_THRESHOLD = 0.4  # see src/C-optimize-smoother/classifier-for-cryptoshow.ipynb (https://github.com/skrhakv/cryptic-finetuning)

DROPOUT = 0.5
LAYER_WIDTH = 2048
ESM2_DIM = 2560
INPUT_DIM = ESM2_DIM * 2


class CryptoBenchClassifier(nn.Module):
    def __init__(self, dim=LAYER_WIDTH, dropout=DROPOUT):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=INPUT_DIM, out_features=dim)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_2 = nn.Linear(in_features=dim, out_features=dim)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_3 = nn.Linear(in_features=dim, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.dropout2(self.relu(self.layer_2(self.dropout1(self.relu(self.layer_1(x)))))))
