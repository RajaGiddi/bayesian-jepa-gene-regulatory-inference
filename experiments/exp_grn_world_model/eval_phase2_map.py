"""Phase 2a evaluation: MAP parameter inference on all 5 Size10 networks.

For each network (fixed gold-standard graph):
  - Run MAP (timeseries + knockouts)
  - Compute trajectory MSE vs observed timeseries
  - Compute knockout steady-state MSE
  - Save results to results/grn_world_model/phase2_fixed_graph/

Run from repo root:
    python experiments/exp_grn_world_model/eval_phase2_map.py
"""
import pathlib, sys, json, warnings
warnings.filterwarnings('ignore')

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

from grn_world_model.data_loader import load_dream4_network
from grn_world_model.jax_model import jax_map
from grn_world_model.ode_model import ode_solve, ode_steady_state, apply_knockout

DATA_DIR    = ROOT / 'data'
RESULTS_DIR = ROOT / 'results' / 'grn_world_model' / 'phase2_fixed_graph'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 10
records = []

for net_id in range(1, 6):
    net = load_dream4_network(DATA_DIR, size=SIZE, network_id=net_id)
    N = net.n_genes
    active_edges = np.argwhere(net.gold_standard)

    print(f'\n=== net{net_id} | {N} genes | {int(net.gold_standard.sum())} edges ===')

    # ------------------------------------------------------------------
    # JAX gradient-based MAP (Adam, 2000 steps) — never gets stuck
    # ------------------------------------------------------------------
    import time as _time
    t_map = _time.time()
    map_vals = jax_map(net.gold_standard, net,
                       use_knockouts=True, use_timeseries=True,
                       num_steps=1000, lr=0.02)
    print(f'  MAP done in {_time.time()-t_map:.1f}s')

    w_flat = map_vals['w_flat']
    gamma  = map_vals['gamma']
    sigma  = map_vals['sigma_obs']
    W = np.zeros((N, N))
    W[active_edges[:, 0], active_edges[:, 1]] = w_flat
    G = net.gold_standard

    print(f'  gamma: min={gamma.min():.3f}  max={gamma.max():.3f}  mean={gamma.mean():.3f}')
    print(f'  |w|:   min={np.abs(w_flat).min():.3f}  max={np.abs(w_flat).max():.3f}')

    # ------------------------------------------------------------------
    # Timeseries trajectory MSE
    # ------------------------------------------------------------------
    ts_mse_list = []
    for ts in net.timeseries:
        try:
            x_pred = ode_solve(W, G, gamma, net.wildtype,
                               ts.timepoints, ts.expression[0], ts.perturbation)
            mse = float(np.mean((x_pred - ts.expression) ** 2))
        except Exception:
            mse = float('nan')
        ts_mse_list.append(mse)

    ts_mse_mean = float(np.nanmean(ts_mse_list))
    print(f'  Timeseries MSE: {ts_mse_mean:.5f}  (per series: {[f"{v:.4f}" for v in ts_mse_list]})')

    # ------------------------------------------------------------------
    # Knockout steady-state MSE
    # ------------------------------------------------------------------
    ko_mse_list = []
    for gene_idx, ko_obs in net.knockout_pairs:
        G_ko = apply_knockout(G, gene_idx)
        try:
            x_ss = ode_steady_state(W, G_ko, gamma, net.wildtype,
                                    x0=net.wildtype.copy(),
                                    knocked_out_gene=gene_idx)
            mse  = float(np.mean((x_ss - ko_obs) ** 2))
        except Exception:
            mse = float('nan')
        ko_mse_list.append(mse)

    ko_mse_mean = float(np.nanmean(ko_mse_list))
    print(f'  Knockout MSE:   {ko_mse_mean:.5f}')

    # ------------------------------------------------------------------
    # Save per-network MAP parameters
    # ------------------------------------------------------------------
    np.savez(
        str(RESULTS_DIR / f'net{net_id}_map_params.npz'),
        w_flat=w_flat, gamma=gamma, sigma_obs=sigma,
        active_edges=active_edges,
    )

    record = {
        'network':       f'net{net_id}',
        'n_genes':       N,
        'n_edges':       int(net.gold_standard.sum()),
        'ts_mse_mean':   round(ts_mse_mean,  6),
        'ts_mse_series': [round(v, 6) for v in ts_mse_list],
        'ko_mse_mean':   round(ko_mse_mean,  6),
    }
    records.append(record)

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
print('\n\n=== Phase 2a Summary (MAP, fixed gold-standard graph) ===')
print(f'{"Network":<10} {"Edges":>6} {"TS MSE":>10} {"KO MSE":>10}')
print('-' * 42)
for r in records:
    print(f'{r["network"]:<10} {r["n_edges"]:>6} {r["ts_mse_mean"]:>10.5f} {r["ko_mse_mean"]:>10.5f}')

ts_mse_all = np.mean([r['ts_mse_mean'] for r in records])
ko_mse_all = np.mean([r['ko_mse_mean'] for r in records])
print('-' * 42)
print(f'{"Mean":<10} {"":>6} {ts_mse_all:>10.5f} {ko_mse_all:>10.5f}')

with open(RESULTS_DIR / 'phase2a_summary.json', 'w') as f:
    json.dump({'networks': records,
               'ts_mse_mean': round(float(ts_mse_all), 6),
               'ko_mse_mean': round(float(ko_mse_all), 6)}, f, indent=2)
print(f'\nSaved to {RESULTS_DIR / "phase2a_summary.json"}')
