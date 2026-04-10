"""DREAM4 data loader for the GRN World Model.

Parses all files for a given network (Size10 or Size100) into structured
numpy arrays ready for ODE fitting and graph inference.

File formats:
  *wildtype.tsv          — 1 row × N_genes, tab-separated, quoted header
  *knockouts.tsv         — N_genes rows × N_genes cols (row i = gene i KO)
  *knockdowns.tsv        — N_genes rows × N_genes cols (row i = gene i KD)
  *timeseries.tsv        — 5 (Size10) or 10 (Size100) blocks of 21 timepoints,
                           blank-line separated, single header row at top
  DREAM4_GoldStandard_InSilico_Size{N}_{id}.tsv
                         — edge list: (source, target, 0/1), no header,
                           located in the Size{N} parent directory
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TimeSeries:
    """One perturbation time series.

    DREAM4 timeseries are multi-gene perturbations: all genes receive a random
    basal activation shift simultaneously (not a single-gene knockdown).
    The exact perturbation vector is stored in timeseries_perturbations.tsv
    (separate download). We approximate it as x(t=0) - wildtype, which is the
    basal shift that the ODE model needs as its 'perturbation' input.
    """
    timepoints: np.ndarray   # (T,)   e.g. [0, 50, ..., 1000]
    expression: np.ndarray   # (T, N) expression matrix
    perturbation: np.ndarray # (N,)   approx basal shift = x(t=0) - wildtype


@dataclass
class DREAM4Network:
    """All data for one DREAM4 in-silico network."""
    network_id: int              # 1–5
    size: int                    # 10 or 100
    gene_names: list[str]        # ['G1', ..., 'GN']

    wildtype: np.ndarray         # (N,)
    knockouts: np.ndarray        # (N, N)  row i = steady-state when gene i KO'd
    knockdowns: np.ndarray       # (N, N)  row i = steady-state when gene i KD'd
    timeseries: list[TimeSeries] # 5 (Size10) or 10 (Size100) series

    gold_standard: Optional[np.ndarray] = None  # (N, N) binary adjacency, None if missing

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def knockout_pairs(self) -> list[tuple[int, np.ndarray]]:
        """List of (gene_idx, steady_state_expression) for each knockout."""
        return [(i, self.knockouts[i]) for i in range(self.n_genes)]


def _parse_matrix(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    """Parse a TSV with a quoted header row. Returns (gene_names, array)."""
    df = pd.read_csv(path, sep="\t")
    gene_names = list(df.columns)
    return gene_names, df.values.astype(np.float64)


def _parse_timeseries(
    path: pathlib.Path,
    wildtype: np.ndarray,
) -> list[TimeSeries]:
    """Parse blank-line-separated timeseries blocks.

    Each block has 21 rows (t=0,50,...,1000). The Time column is the first
    column. The perturbed gene is identified as argmax |x(t=0) - wildtype|.
    """
    series_list: list[TimeSeries] = []

    with open(path) as f:
        lines = f.readlines()

    # First non-empty line is the header
    header = None
    current_block: list[list[float]] = []

    def _flush(block: list[list[float]]) -> None:
        if not block:
            return
        arr = np.array(block, dtype=np.float64)   # (T, 1+N)
        timepoints = arr[:, 0]
        expression = arr[:, 1:]                    # (T, N)
        perturbation = expression[0] - wildtype    # (N,) approx basal shift
        series_list.append(TimeSeries(
            timepoints=timepoints,
            expression=expression,
            perturbation=perturbation,
        ))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line = end of block
            _flush(current_block)
            current_block = []
            continue
        if header is None:
            header = stripped  # skip header row, don't add to block
            continue
        vals = [float(v) for v in stripped.split("\t")]
        current_block.append(vals)

    _flush(current_block)  # last block (no trailing blank line)
    return series_list


def _parse_gold_standard(path: pathlib.Path, n_genes: int) -> np.ndarray:
    """Parse DREAM4 gold standard → binary (N, N) adjacency matrix.

    Format: edge list with no header, three tab-separated columns:
        source_gene  target_gene  label(0/1)
    e.g. "G1  G2  1"

    All N×N directed pairs are listed (including zeros).
    """
    adj = np.zeros((n_genes, n_genes), dtype=np.float64)

    def _idx(name: str) -> int:
        return int(name.strip().lstrip("G")) - 1  # 'G3' → 2

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            src, tgt, label = parts
            i, j = _idx(src), _idx(tgt)
            if int(label):
                adj[i, j] = 1.0

    return adj


def load_dream4_network(
    data_dir: pathlib.Path,
    size: int,
    network_id: int,
) -> DREAM4Network:
    """Load all data for one DREAM4 network.

    Parameters
    ----------
    data_dir    : path to data/ directory (contains DREAM4_InSilico_Size10/ etc.)
    size        : 10 or 100
    network_id  : 1–5

    Returns
    -------
    DREAM4Network with all parsed arrays.
    """
    assert size in (10, 100), f"size must be 10 or 100, got {size}"
    assert 1 <= network_id <= 5, f"network_id must be 1–5, got {network_id}"

    prefix = f"insilico_size{size}_{network_id}"
    net_dir = data_dir / f"DREAM4_InSilico_Size{size}" / prefix

    if not net_dir.exists():
        raise FileNotFoundError(f"Network directory not found: {net_dir}")

    # Wildtype
    wt_path = net_dir / f"{prefix}_wildtype.tsv"
    gene_names, wt_arr = _parse_matrix(wt_path)
    wildtype = wt_arr[0]  # (N,)
    n_genes = len(gene_names)

    # Knockouts: (N_genes, N_genes)
    ko_path = net_dir / f"{prefix}_knockouts.tsv"
    _, knockouts = _parse_matrix(ko_path)

    # Knockdowns: (N_genes, N_genes)
    kd_path = net_dir / f"{prefix}_knockdowns.tsv"
    _, knockdowns = _parse_matrix(kd_path)

    # Timeseries
    ts_path = net_dir / f"{prefix}_timeseries.tsv"
    timeseries = _parse_timeseries(ts_path, wildtype)

    # Gold standard — lives in the Size{N} parent directory
    size_dir = data_dir / f"DREAM4_InSilico_Size{size}"
    gs_path = size_dir / f"DREAM4_GoldStandard_InSilico_Size{size}_{network_id}.tsv"
    gold_standard = None
    if gs_path.exists():
        try:
            gold_standard = _parse_gold_standard(gs_path, n_genes)
        except Exception as e:
            print(f"Warning: could not parse gold standard {gs_path}: {e}")

    return DREAM4Network(
        network_id=network_id,
        size=size,
        gene_names=gene_names,
        wildtype=wildtype,
        knockouts=knockouts,
        knockdowns=knockdowns,
        timeseries=timeseries,
        gold_standard=gold_standard,
    )


def load_all_networks(
    data_dir: pathlib.Path,
    size: int = 10,
) -> list[DREAM4Network]:
    """Load all 5 networks for a given size."""
    return [load_dream4_network(data_dir, size, nid) for nid in range(1, 6)]
