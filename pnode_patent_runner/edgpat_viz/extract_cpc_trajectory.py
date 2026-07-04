"""
(A) Extract per-year CPC (field_5) memory snapshots from a trained EDGPAT model.
Replays all events in chronological order, snapshotting field_5 memory at each
year boundary -> a trajectory v_j^t for every CPC code. Saved for 2D-projection
visualisation of "which technologies are heating up".
"""
import warnings; warnings.filterwarnings("ignore")
import math, json, numpy as np, pandas as pd, torch

from model.EDGPAT import EDGPAT
from utils.data_processing import get_data

D = "/tmp/EDGPAT/data"
TAG = "G"
PREFIX = "proj-field128-classall-hierarchy_sum_time"
MODEL = f"./saved_models/{PREFIX}/patent_{TAG}-2.pth"
MEM_DIM = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- counts (match the trained run) ----------------------------------------
NUM = dict(company=8529, field_5=6707, field_4=976, field_3=78, field_2=14, field_1=1)
TYPES = ["company", "field_5", "field_4", "field_3", "field_2", "field_1"]
n_nodes = {t: NUM[t] for t in TYPES}
num_hir = {t: NUM[t] for t in TYPES[2:]}

# --- data ------------------------------------------------------------------
reflect, full, tr, val, te = get_data(
    f"patent_{TAG}", [f"5-4-level_{TAG}", f"4-3-level_{TAG}", f"3-2-level_{TAG}", f"2-1-level_{TAG}"],
    use_validation=True)

# --- model -----------------------------------------------------------------
tgn = EDGPAT(device=device, n_layers=1, n_nodes=n_nodes, time_dim=64, use_time=True,
             time_enc_type="sin", dropout=0, type=TYPES, message_dimension=128,
             memory_dimension=MEM_DIM, embedding_module_type="identity",
             message_function="mlp", memory_updater_type="gru", reflect=reflect,
             loss_alpha=1, use_history=True, num_hier=num_hir).to(device)
state = torch.load(MODEL, map_location=device)
tgn.load_state_dict(state, strict=False)
tgn.eval()
print("model loaded from", MODEL)

# --- replay events in natural (time-sorted) order, snapshot per calendar year
# `label` is a prediction-anchor flag (0 = memory-building only), NOT the year.
# Use the calendar `year` column; process rows in file order (timestamps are
# globally monotonic, satisfying the memory-updater's assertion).
import ast
ev = pd.read_csv(f"{D}/patent_{TAG}.csv")
src = ev["company"].astype(int).to_numpy()
dst = [ast.literal_eval(s) for s in ev["fields"]]
ts = ev["timestamp2"].astype(int).to_numpy()          # same time field as training
cyr = ev["year"].astype(int).to_numpy()               # calendar year 2010..2018
years = sorted(set(cyr.tolist()))
print("calendar years:", years)

snapshots = {}                                         # calendar year -> (n_field_5, dim)
BS = 256
tgn.init_memory()
with torch.no_grad():
    for y in years:
        idx = np.where(cyr == y)[0]                    # already contiguous & ordered
        for s in range(0, len(idx), BS):
            b = idx[s:s + BS]
            upd = tgn.compute_update_memory(src[b], [dst[k] for k in b], ts[b])
            tgn.update_self_memory(upd)
            tgn.detach_memory()
        snapshots[y] = tgn.memory.memory["field_5"].data.cpu().numpy().copy()
        nz = int((np.linalg.norm(snapshots[y], axis=1) > 1e-6).sum())
        print(f"  year {y}: events={len(idx):6d}  active CPCs (nonzero mem)={nz}")

# --- CPC metadata for colouring -------------------------------------------
l5 = pd.read_csv(f"{D}/5-4-level_{TAG}.csv")
cpc_str = l5["node"].astype(str).tolist()          # new L5 id -> code string
cpc_cls = [s[:3] for s in cpc_str]                 # e.g. G01, G06, G11

arr = np.stack([snapshots[y] for y in years], axis=0)   # (T, n_field_5, dim)
np.savez_compressed(f"{D}/cpc_traj_{TAG}.npz",
                    traj=arr, years=np.array(years),
                    cpc_str=np.array(cpc_str), cpc_cls=np.array(cpc_cls))
print(f"\nsaved {D}/cpc_traj_{TAG}.npz  shape={arr.shape}  (T, n_cpc, dim)")
