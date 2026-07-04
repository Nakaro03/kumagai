"""
Build a single-IPC-section subset of the EDGPAT ALL dataset (data-time-fin.csv).
Sections: G=Physics, F=Mechanism. The ALL dataset is the only one the authors
shared; PHYSICS/MECHANISM are reconstructed here by filtering to one section.

Hierarchy is derived purely from the L5 code string:
  L5 "G01R35/04" -> L4 "G01R35" (before '/') -> L3 "G01R" (4 chars)
  -> L2 "G01" (3 chars) -> L1 "G" (section).
Reflect files: model reads the LAST id column as the coarser-level parent id.
"""
import sys, ast, json
import pandas as pd

SEC = sys.argv[1] if len(sys.argv) > 1 else "G"     # G=Physics, F=Mechanism
TAG = sys.argv[2] if len(sys.argv) > 2 else "G"
D = "/tmp/EDGPAT/data"

L4 = lambda s: s.split("/")[0]
L3 = lambda s: s[:4]
L2 = lambda s: s[:3]
L1 = lambda s: s[:1]

# --- L5 id -> code string -------------------------------------------------
l5 = pd.read_csv(f"{D}/5-4-level.csv")
l5str = dict(zip(l5["Unnamed: 0"].astype(int), l5["node"].astype(str)))
target = {i: s for i, s in l5str.items() if s[:1] == SEC}      # old L5 id -> str
old_ids = set(target)
print(f"[{SEC}] L5 codes in section: {len(old_ids)}")

# --- per-company section truth (padded years 0..YMAX) ----------------------
# get_truth_patents walks years upward from `label` and needs a non-empty
# future year; otherwise it runs off the key range -> KeyError. Mirror the
# original invariant: keep only events whose company has a non-empty section
# truth at some year >= label.
truth_all = json.load(open(f"{D}/real-data.json"))
YMAX = max(int(y) for c in truth_all for y in truth_all[c])    # = 8 (2018)
g_truth_old = {}            # old company id -> {year(0..YMAX): [G L5 ids]}
max_nonempty = {}           # old company id -> max year with non-empty G truth
for c, rec in truth_all.items():
    yrs = {str(y): [x for x in rec.get(str(y), []) if x in old_ids]
           for y in range(YMAX + 1)}
    g_truth_old[int(c)] = yrs
    ne = [int(y) for y in yrs if yrs[y]]
    max_nonempty[int(c)] = max(ne) if ne else -1

# --- filter events: section codes present AND company has reachable truth ---
df = pd.read_csv(f"{D}/data-time-fin.csv")
def keep(fs):
    return [x for x in ast.literal_eval(fs) if x in old_ids]
df["fk"] = df["fields"].map(keep)
df = df[df["fk"].map(len) > 0].copy()
df = df[df.apply(lambda r: max_nonempty.get(int(r["company"]), -1) >= int(r["label"]), axis=1)].copy()
print(f"[{SEC}] events kept: {len(df)} / 373880")

# --- re-index L5 (stable by old id) and derive coarser vocabularies --------
new_l5 = sorted(old_ids)
l5_o2n = {o: n for n, o in enumerate(new_l5)}
K5 = len(new_l5)
strs = [target[o] for o in new_l5]
l4v = sorted({L4(s) for s in strs}); l4_id = {s: i for i, s in enumerate(l4v)}; K4 = len(l4v)
l3v = sorted({L3(s) for s in strs}); l3_id = {s: i for i, s in enumerate(l3v)}; K3 = len(l3v)
l2v = sorted({L2(s) for s in strs}); l2_id = {s: i for i, s in enumerate(l2v)}; K2 = len(l2v)
l1v = sorted({L1(s) for s in strs}); l1_id = {s: i for i, s in enumerate(l1v)}; K1 = len(l1v)
print(f"[{SEC}] vocab  L5={K5} L4={K4} L3={K3} L2={K2} L1={K1}")

# --- re-index companies ----------------------------------------------------
comps = sorted(df["company"].astype(int).unique())
c_o2n = {o: n for n, o in enumerate(comps)}
C = len(comps)
print(f"[{SEC}] companies: {C}")

# --- rewrite events --------------------------------------------------------
df["company"] = df["company"].astype(int).map(c_o2n)
df["fields"] = df["fk"].map(lambda l: "[" + ", ".join(str(l5_o2n[x]) for x in l) + "]")
df = df.drop(columns=["fk"])
df.to_csv(f"{D}/patent_{TAG}.csv", index=False)

# --- reflect files: columns [idx, node, self_id, parent_id]; parent = last --
def write_reflect(name, rows):
    pd.DataFrame(rows, columns=["node", "self", "parent"]).to_csv(f"{D}/{name}_{TAG}.csv")

write_reflect("5-4-level", [(strs[n], n, l4_id[L4(strs[n])]) for n in range(K5)])
write_reflect("4-3-level", [(s, l4_id[s], l3_id[L3(s)]) for s in l4v])
write_reflect("3-2-level", [(s, l3_id[s], l2_id[L2(s)]) for s in l3v])
write_reflect("2-1-level", [(s, l2_id[s], l1_id[L1(s)]) for s in l2v])

# --- truth json: company -> year(0..YMAX) -> [re-indexed L5 ids] ------------
out = {}
for old_c in comps:
    yrs = g_truth_old[old_c]                      # already section-filtered, padded
    out[str(c_o2n[old_c])] = {y: [l5_o2n[x] for x in yrs[y]] for y in yrs}
json.dump(out, open(f"{D}/real-data_{TAG}.json", "w"))
print(f"[{SEC}] truth companies written: {len(out)}")

print(f"\n=== ARGS for main.py ===")
print(f"-d patent_{TAG} --truth real-data_{TAG} "
      f"--reflect5_4 5-4-level_{TAG} --reflect4_3 4-3-level_{TAG} "
      f"--reflect3_2 3-2-level_{TAG} --reflect2_1 2-1-level_{TAG} "
      f"--n_company {C} --n_field_5 {K5} --n_field_4 {K4} "
      f"--n_field_3 {K3} --n_field_2 {K2} --n_field_1 {K1}")
