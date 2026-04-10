"""Phase 3 evaluation: graph inference on all 5 Size10 networks.

Methods compared
----------------
3D  knockout_response   : |X_ko_i[j] - X_wt[j]| — pure interventional signal
3C  gradient_matching   : horseshoe on linearised ODE (finite-difference dx/dt)
Random                  : edge density prior (AUPR = n_pos / n_possible)
GENIE3 reference        : ~0.35–0.45 (timeseries RF, no knockout data)

Run from repo root:
    python experiments/exp_grn_world_model/eval_phase3_joint.py
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import pathlib, sys, json, time, warnings
warnings.filterwarnings('ignore')

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from grn_world_model.data_loader import load_dream4_network
from grn_world_model.phase3_joint import (
    run_phase3_knockout_response,
    run_phase3_gradient_matching,
    run_phase3_horseshoe,
    compute_aupr,
    compute_auroc,
)

DATA_DIR    = ROOT / 'data'
RESULTS_DIR = ROOT / 'results' / 'grn_world_model' / 'phase3_joint'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GENIE3_AUPR_REF = 0.40   # approximate DREAM4 Size10 mean

SIZE      = 10
GM_STEPS  = 3000
GM_LR     = 0.01
GM_TAU0   = 0.1

records = []

for net_id in range(1, 6):
    net = load_dream4_network(DATA_DIR, size=SIZE, network_id=net_id)
    N   = net.n_genes
    n_pos      = int(net.gold_standard.sum())
    n_possible = N * (N - 1)
    random_aupr = n_pos / n_possible

    print(f'\n=== net{net_id} | {N} genes | {n_pos} true edges ===')

    # ------------------------------------------------------------------
    # 3D: knockout perturbation response (no model, instant)
    # ------------------------------------------------------------------
    res_ko = run_phase3_knockout_response(net)
    aupr_ko  = compute_aupr(res_ko['edge_scores'],  net.gold_standard)
    auroc_ko = compute_auroc(res_ko['edge_scores'], net.gold_standard)
    print(f'  [3D] Knockout response   AUPR={aupr_ko:.3f}  AUROC={auroc_ko:.3f}')

    # ------------------------------------------------------------------
    # 3D+ND: knockout response + network deconvolution
    # ------------------------------------------------------------------
    res_nd = run_phase3_knockout_response(net, deconvolve=True, alpha=0.9)
    aupr_nd  = compute_aupr(res_nd['edge_scores'],  net.gold_standard)
    auroc_nd = compute_auroc(res_nd['edge_scores'], net.gold_standard)
    print(f'  [3D+ND] Deconvolved KO   AUPR={aupr_nd:.3f}  AUROC={auroc_nd:.3f}')

    # ------------------------------------------------------------------
    # 3C: gradient matching horseshoe MAP
    # ------------------------------------------------------------------
    t0 = time.time()
    res_gm = run_phase3_gradient_matching(
        net, num_steps=GM_STEPS, lr=GM_LR, tau0=GM_TAU0,
        seed=net_id, verbose=False,
    )
    elapsed = time.time() - t0
    aupr_gm  = compute_aupr(res_gm['edge_scores'],  net.gold_standard)
    auroc_gm = compute_auroc(res_gm['edge_scores'], net.gold_standard)
    print(f'  [3C] Gradient matching   AUPR={aupr_gm:.3f}  AUROC={auroc_gm:.3f}  ({elapsed:.0f}s)')

    # ------------------------------------------------------------------
    # Hybrid: knockout response + gradient matching
    # ------------------------------------------------------------------
    es_ko = res_ko['edge_scores']
    es_gm = res_gm['edge_scores']
    es_ko_n = es_ko / (es_ko.max() + 1e-9)
    es_gm_n = es_gm / (es_gm.max() + 1e-9)
    hybrid = 0.5 * es_ko_n + 0.5 * es_gm_n
    aupr_hyb  = compute_aupr(hybrid,  net.gold_standard)
    auroc_hyb = compute_auroc(hybrid, net.gold_standard)
    print(f'  [Hybrid]                 AUPR={aupr_hyb:.3f}  AUROC={auroc_hyb:.3f}')
    print(f'  Random baseline          AUPR={random_aupr:.3f}  AUROC=0.500')

    # ------------------------------------------------------------------
    # 3B: horseshoe MAP, knockout-only (no timeseries noise)
    # ------------------------------------------------------------------
    t0 = time.time()
    res_hs = run_phase3_horseshoe(
        net, use_timeseries=False, use_knockouts=True,
        num_steps=GM_STEPS, lr=GM_LR, tau0=0.1,
        seed=net_id, verbose=False,
    )
    elapsed_hs = time.time() - t0
    aupr_hs  = compute_aupr(res_hs['edge_scores'],  net.gold_standard)
    auroc_hs = compute_auroc(res_hs['edge_scores'], net.gold_standard)
    print(f'  [3B] Horseshoe KO-only   AUPR={aupr_hs:.3f}  AUROC={auroc_hs:.3f}  ({elapsed_hs:.0f}s)')

    # Save knockout response scores
    np.save(str(RESULTS_DIR / f'net{net_id}_ko_response.npy'), es_ko)

    records.append({
        'network':       f'net{net_id}',
        'n_true_edges':  n_pos,
        'random_aupr':   round(random_aupr, 4),
        'aupr_ko':       round(aupr_ko,  4),
        'auroc_ko':      round(auroc_ko, 4),
        'aupr_nd':       round(aupr_nd,  4),
        'auroc_nd':      round(auroc_nd, 4),
        'aupr_hs':       round(aupr_hs,  4),
        'auroc_hs':      round(auroc_hs, 4),
        'aupr_gm':       round(aupr_gm,  4),
        'auroc_gm':      round(auroc_gm, 4),
        'aupr_hybrid':   round(aupr_hyb, 4),
        'auroc_hybrid':  round(auroc_hyb, 4),
    })

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
print('\n\n=== Phase 3 Summary ===')
print(f'{"Network":<10} {"Edges":>6} {"Random":>8} '
      f'{"KO-raw":>8} {"KO+ND":>7} {"Horseshoe":>10} {"GradMatch":>10}')
print('-' * 68)
for r in records:
    print(f'{r["network"]:<10} {r["n_true_edges"]:>6} {r["random_aupr"]:>8.3f} '
          f'{r["aupr_ko"]:>8.3f} {r["aupr_nd"]:>7.3f} '
          f'{r["aupr_hs"]:>10.3f} {r["aupr_gm"]:>10.3f}')

mean_ko  = np.mean([r['aupr_ko']  for r in records])
mean_nd  = np.mean([r['aupr_nd']  for r in records])
mean_hs  = np.mean([r['aupr_hs']  for r in records])
mean_gm  = np.mean([r['aupr_gm']  for r in records])
print('-' * 68)
print(f'{"Mean":<10} {"":>6} {"":>8} '
      f'{mean_ko:>8.3f} {mean_nd:>7.3f} {mean_hs:>10.3f} {mean_gm:>10.3f}')
print(f'\nGENIE3 reference AUPR ≈ {GENIE3_AUPR_REF:.3f}  (timeseries RF, no knockout data)')

summary = {
    'networks':        records,
    'mean_aupr_ko':    round(float(mean_ko),  4),
    'mean_aupr_nd':    round(float(mean_nd),  4),
    'mean_aupr_hs':    round(float(mean_hs),  4),
    'mean_aupr_gm':    round(float(mean_gm),  4),
    'genie3_ref':      GENIE3_AUPR_REF,
}
with open(RESULTS_DIR / 'phase3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nSaved to {RESULTS_DIR / "phase3_summary.json"}')
