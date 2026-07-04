# DRIFT — Depth-Recurrent Interpretable Forecasting of Topics

**Proposed model using OpenMythos (Recurrent-Depth Transformer) for emerging-topic forecasting on bipartite scholarly graphs**

**Target venue**: KDD / WWW / WSDM (applied data mining track)
**Date**: 2026-05-20
**Lineage**: Successor to X5 (PI-SDE) — carries the empirically-validated Φ-anchor mechanism into a recurrent-depth transformer.

---

## 0. One-paragraph pitch

Temporal graph neural networks are now known to *memorize* recurring edges and collapse on genuinely unseen ones (EdgeBank, NeurIPS 2022; TGB-Seq, ICLR 2025) — exactly the failure our own X5/X3-clean leave-one-out experiment exhibited (Spearman ρ = −0.35 on held-out timepoints). We propose **DRIFT**, a depth-recurrent transformer that reframes emerging-topic forecasting as *adaptive-depth latent reasoning* over a bipartite inventor–topic graph: the number of loop iterations a topic receives is a learned measure of forecast difficulty (PonderNet-style halting), and each iteration emits an interpretable intermediate potential landscape Φ. This both (a) generalizes to unseen/emerging topics better than memorization-prone temporal GNNs, and (b) resolves the central open weakness of recurrent-depth transformers — that their latent iterations are uninterpretable (Lu et al., COLM 2025) — by anchoring each iteration to an observable growth signal.

---

## 1. Motivation chain (each link grounded in real literature)

1. **Temporal GNNs memorize, not generalize.**
   - *EdgeBank* (Poursafaei, Huang, Pelrine, Rabbany — NeurIPS 2022 D&B): a pure-memorization baseline rivals deep temporal models under easy negatives.
   - *TGB-Seq* (ICLR 2025, arXiv 2502.02975): on datasets with minimal repeated edges, GraphMixer/DyGFormer largely memorize and degrade on unseen edges.
   - **Our connection**: X3-clean's leave-one-out collapse (ρ = −0.35) is the same phenomenon. We have first-hand evidence of the open problem.

2. **Recurrent-depth transformers scale test-time reasoning** but are opaque.
   - *Universal Transformers* (Dehghani et al., ICLR 2019): weight-tied depth recurrence + ACT halting.
   - *Huginn / Recurrent Depth* (Geiping, McLeish et al., NeurIPS 2025 spotlight, arXiv 2502.05171): iterate a recurrent block to arbitrary test-time depth for latent reasoning. **= OpenMythos's architecture.**
   - *Lu et al.* (COLM 2025 workshop, arXiv 2507.02199): probing Huginn shows **most latent iterations are NOT interpretable** — the clearest open weakness.

3. **Adaptive halting is mature enough to use.**
   - *ACT* (Graves, 2016); *PonderNet* (Banino, Balaguer, Blundell — ICML 2021 workshop): probabilistic halting with unbiased gradients, improves extrapolation.

4. **We hold the missing piece: interpretability.**
   - Our X5 ablation proved the Φ-anchor (`Φ(centroid_k, t) ≈ −growth_k`) is the *only* mechanism that drives predictive ranking (A2 ablation: Hits@10 0.46 → 0.16 when removed; equals PRESCIENT). We repurpose it as the per-iteration interpretability hook for DRIFT.

---

## 2. The model

### 2.1 Inputs (reuse existing data pipeline)
- Bipartite graph `G_t = (Inventors ∪ Topics, edges)` per timepoint, from `data/{DOMAIN}/alltime/fate_train.pt`.
- Topic centroids `c_k(t) ∈ R^d`, growth signal `g_k(t)` (already extracted).
- Node features = VGAE bipartite embedding (already trained).

### 2.2 Architecture (maps directly onto OpenMythos `MythosConfig`)

```
                    ┌─────────────────────────────────────────┐
 graph state h^(0)  │   PRELUDE  (prelude_layers)              │  encode G_{1..T_obs}
 ─────────────────► │   ────────────────────────────────────  │
                    │   RECURRENT CORE  (looped block)         │  ← OpenMythos max_loop_iters
                    │     for n = 1 .. N (adaptive):           │
                    │       h^(n) = Block(h^(n-1), t)          │     MLA/GQA attention
                    │       Φ^(n) = PotentialHead(h^(n))       │     sparse MoE (regime experts)
                    │       p_halt^(n) = HaltHead(h^(n))       │     PonderNet halting
                    │   ────────────────────────────────────  │
                    │   CODA  (coda_layers)                    │  decode topic-growth ranking
                    └─────────────────────────────────────────┘
                                       │
                            forecast g_k(T_obs + Δ)
```

| OpenMythos config field | DRIFT role |
|---|---|
| `max_loop_iters` | maximum forecast-reasoning depth |
| `prelude_layers` / `coda_layers` | graph encode / growth decode |
| `attn_type=mla`, `kv_lora_rank` | efficient attention over many inventor nodes |
| `n_experts`, `n_experts_per_tok` | **temporal-regime experts** (stable / emerging / declining / bursting) |
| (new) `PotentialHead` | per-iteration Φ landscape (interpretability) |
| (new) `HaltHead` | PonderNet halting → "forecast difficulty" |

### 2.3 Losses

```
L = L_forecast            # ranking/regression of held-out future growth (NDCG/listwise)
  + λ_anchor · L_anchor   # Σ_n Σ_k (Φ^(n)(c_k,t) + g_k)²   ← carried from X5, applied per-iteration
  + λ_ponder · L_ponder   # PonderNet halting KL prior (Banino 2021)
  + λ_lb     · L_balance  # MoE load-balancing (Switch Transformer / DeepSeekMoE)
```

Note: `L_anchor` is applied at **every** loop iteration `n`, so the model learns a *trajectory* of refining Φ landscapes — this is what makes iterations interpretable.

---

## 3. Contributions (and which gap each fills)

| # | Contribution | Gap it fills | Risk |
|---|---|---|---|
| **C1** | Reframe emerging-topic forecasting as *adaptive-depth latent reasoning* (loop count = forecast difficulty) | "new-cluster emergence forecasting is thin" (TGB-Seq) | low |
| **C2** | Show recurrent-depth generalizes to **unseen/emerging** topics where temporal GNNs memorize | EdgeBank/TGB-Seq central critique | medium — must beat DyGFormer/GraphMixer on held-out emerging edges |
| **C3** | **Interpretable halting**: each iteration emits an anchored Φ landscape; halting depth = forecast confidence | Lu et al. (COLM 2025): RDT iterations are uninterpretable | **high — this is the make-or-break claim** |
| **C4** | MoE experts specialize by temporal regime; routing is an interpretable dynamics label | (novel) | medium |

**C3 is the intellectual core.** It converts X5's "thin" Φ-anchor into the solution for RDT's biggest documented weakness. That re-framing is what elevates this from engineering to a contribution.

---

## 4. Evaluation plan

### 4.1 Datasets
- Our 3 domains: patent_energy (Y02 CPC), arxiv_construction, jp_construction.
- **Add a public benchmark for credibility**: TGB-Seq citation-network domain, and/or Impact4Cast (Gu & Krenn, *Mach. Learn. Sci. Technol.* 2025) for high-impact link forecasting.

### 4.2 Protocol (the honest one — true held-out)
- Train on years ≤ Y; forecast held-out future window (Y+1 … Y+Δ).
- Report on **emerging** topics specifically (topics whose growth rank changes), not just stable ones — this is where memorization baselines fail.

### 4.3 Metrics (reuse x5/eval.py)
- Ranking: NDCG@10, Hits@10, MRR, AP.
- Distribution (if SDE-style rollout retained): W1, MMD.
- **Interpretability metrics** (for C3): correlation between halting depth and forecast error; whether per-iteration Φ converges monotonically; expert-routing purity by regime.

### 4.4 Baselines (already have most)
- Memorization: EdgeBank, Persistence.
- Temporal GNN: TGN, DyGFormer, GraphMixer (TGB-Seq leaders).
- Trajectory: PRESCIENT, MIOFlow (already reimplemented), X5.
- TSFM: Chronos, Moirai (already run).
- Ablations: no-recurrence (1 loop), no-anchor, no-halting (fixed depth), no-MoE.

---

## 5. Why KDD/WWW/WSDM (not NeurIPS)

- The method = a *coherent recombination* (RDT + PonderNet halting + MoE + Φ-anchor) rather than a new theorem — KDD/WWW reward this when paired with a real application and strong empirics.
- The application (patent/science trend forecasting) has clear industrial value.
- C3 (interpretable latent reasoning) is novel enough to be the headline, but is framed as "making a powerful architecture usable/trustworthy for analysts" — an applied framing, not a theory claim.

---

## 6. Honest risk register (lessons from X5)

| Risk | Mitigation |
|---|---|
| **C3 fails** — anchored iterations still uninterpretable (as Lu et al. found for vanilla Huginn) | Even a *negative* interpretability result is publishable if rigorous; pre-register the interpretability metrics. The anchor *forces* structure that vanilla Huginn lacks, so we have a mechanistic reason to expect success. |
| DRIFT does not beat DyGFormer on TGB-Seq | Position around *emerging* edges specifically; report where each method wins. |
| Recurrent-depth training instability (DEQ/Huginn known issue) | Use truncated/randomized unroll depth; OpenMythos already implements the looped block. |
| "Just X5 + a transformer" critique | Ablation must show recurrence + halting add value *beyond* the anchor alone (the thing X5 lacked). If they don't, fall back to applied/benchmark paper. |
| arxiv_construction too small for transformers | Treat as a stress-test domain; lead with patent_energy + a large public benchmark. |

---

## 7. Minimal first experiment (1 week, decides go/no-go)

1. Wrap existing bipartite data as a sequence for OpenMythos (`.mythos_venv`).
2. Implement `PotentialHead` + per-iteration `L_anchor` + PonderNet `HaltHead`.
3. Train DRIFT (small config) on patent_energy with **true held-out** future window.
4. Check the two make-or-break signals:
   - Does held-out NDCG@10 beat X5 and EdgeBank? (C2)
   - Does halting depth correlate with forecast difficulty, and do per-iteration Φ refine monotonically? (C3)
5. If both fire → full study. If only C2 → applied paper. If neither → revert to X5/X3-clean descriptive (VIS).

---

## Appendix: key citations (verify author strings before final submission)

- EdgeBank — Poursafaei et al., NeurIPS 2022 D&B (arXiv 2207.10128)
- TGB-Seq — ICLR 2025 (arXiv 2502.02975) *[verify authors]*
- TGB — Huang et al., NeurIPS 2023 D&B (arXiv 2307.01026)
- DyGFormer/DyGLib — Yu et al., NeurIPS 2023
- GraphMixer — Cong et al., ICLR 2023
- Universal Transformers — Dehghani et al., ICLR 2019 (arXiv 1807.03819)
- Huginn / Recurrent Depth — Geiping, McLeish et al., NeurIPS 2025 (arXiv 2502.05171)
- Latent CoT probing — Lu et al., COLM 2025 workshop (arXiv 2507.02199)
- PonderNet — Banino, Balaguer, Blundell, ICML 2021 workshop (arXiv 2107.05407)
- ACT — Graves, 2016 (arXiv 1603.08983)
- MLA — DeepSeek-V2, 2024 (arXiv 2405.04434)
- GQA — Ainslie et al., EMNLP 2023 (arXiv 2305.13245)
- DeepSeekMoE — Dai et al., 2024 (arXiv 2401.06066)
- CG-ODE — Huang, Sun, Wang, KDD 2021
- Impact4Cast — Gu & Krenn, Mach. Learn. Sci. Technol. 2025 (arXiv 2402.08640)
- Science of science — Fortunato et al., Science 2018
- Disruption index — Wu, Wang, Evans, Nature 2019; Park, Leahey, Funk, Nature 2023
