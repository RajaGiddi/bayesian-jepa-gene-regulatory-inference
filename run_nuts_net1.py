"""Run Phase 2b NUTS for net1 (Size10) using JAX/diffrax/numpyro."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import pathlib, sys, json, time
sys.path.insert(0, 'src')

import numpy as np
from grn_world_model.data_loader import load_dream4_network
from grn_world_model.jax_model import run_nuts, mcmc_to_arviz

DATA_DIR    = pathlib.Path('data')
RESULTS_DIR = pathlib.Path('results/grn_world_model/phase2_fixed_graph')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

net = load_dream4_network(DATA_DIR, size=10, network_id=1)
print(f'net1: {net.n_genes} genes, {int(net.gold_standard.sum())} edges', flush=True)

t0 = time.time()
mcmc = run_nuts(
    net.gold_standard, net,
    num_samples=500, num_warmup=500, num_chains=2, seed=42,
    use_knockouts=True, use_timeseries=False,   # knockout-only NUTS
)
elapsed = time.time() - t0
print(f'Done in {elapsed/60:.1f} min', flush=True)

mcmc.print_summary()

idata = mcmc_to_arviz(mcmc)
idata.to_netcdf(str(RESULTS_DIR / 'net1_nuts_trace.nc'))
print('Trace saved.', flush=True)
