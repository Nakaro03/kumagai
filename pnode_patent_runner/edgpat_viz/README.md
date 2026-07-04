# EDGPAT latent map — "rising technologies" visualization

Visualises **which technologies are heating up** on top of EDGPAT's learned
CPC (`field_5`) memory. We do *not* train a new encoder — we read EDGPAT's own
time-evolving code embeddings and render momentum on a 2D projection.

## Pipeline

1. **`build_section_subset.py`** — EDGPAT only ships the ALL dataset; this
   reconstructs a single IPC-section subset (`G`=Physics, `F`=Mechanism) from
   `data-time-fin.csv` by filtering codes + re-indexing + rebuilding the
   hierarchy reflect files and truth json (with the year-walk termination
   invariant). Run inside the EDGPAT repo: `python build_section_subset.py G G`.
2. Train EDGPAT on the subset (see EDGPAT `main.py`, args printed by the builder).
3. **`extract_cpc_trajectory.py`** — replays all events in time order through the
   trained model, snapshotting `field_5` memory at each calendar year →
   `cpc_traj_G.npz`  shape `(years=9, n_cpc=6707, dim=128)`.
4. **Visualisations** (PCA 2D layout + per-year momentum heat + Moran's I):
   - `viz_rising_tech.py`     raw layout, momentum = memory velocity / filing growth
   - `viz_rising_tech_norm.py` L2-normalised layout (direction), directional momentum
   - `viz_rising_hybrid.py`   **recommended** — normalised layout + raw-velocity heat

## Key findings (1-epoch model, reconstructed Physics/G subset)

- **Legibility gate PASSES.** "Rising" is spatially clustered, not random:
  Moran's I of memory-velocity ≈ **+0.12 … +0.23** across years (raw filing
  growth is weaker/noisier, +0.02 … +0.17).
- **The rising signal is MAGNITUDE, not direction.** Direction-normalising the
  vectors gives a richer 2D map (PCA var 0.98/0.02 → 0.54/0.40) but the momentum
  signal collapses (Moran's I → ~0.04). Hence the **hybrid**: normalised layout
  for a readable map, raw velocity for the strong heat (Moran's I back to ~0.13–0.23).
- **Coherent emergence story** (top rising by memory velocity):
  optics/sensors (2011–14) → computing/software (2015–16) →
  **G16H healthcare informatics (2017)** → **G06F16 big-data/IR (2018)**.
  Real, recognizable technology trends are surfaced as the hottest.

## Caveats

- 1-epoch undertrained EDGPAT; reconstructed section subset (not the authors'
  exact PHYSICS split); PCA 2D is the standard but lossy projection.
- Descriptive only — this **observes** momentum, it does not predict it.

## Next

- (B) continuous-time: swap EDGPAT's `memory_updater` GRU for a GRU-ODE so the
  memory evolves between events → smooth trajectories + (bonus) gap-regime
  prediction. The piecewise-constant memory is EDGPAT's structural limitation.
