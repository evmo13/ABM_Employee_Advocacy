
# build_networks_all_in_one.py
# Creates multilayer networks & saves CSVs to a specified folder (Windows path supported).
# ASCII-only to avoid encoding issues on Windows.

import os, json, time
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict

# ===================== USER SETTINGS =====================
INPUT_CSV = r"D:\downloads\df_88225.csv"   # path to your dataset (same folder as this script or absolute)
OUT_DIR   = r"D:\\Desktop\\population\\3"       # target output folder (will be created)
SEED      = 42

# Choose one company for the internal network:
SELECTED_WORK_ID = None   # set to an integer Work_ID to force a company, or keep None to auto-pick largest

# Isolates pruning (global active set)
PRUNE_ISOLATES = True
# ===================== END SETTINGS ======================


# ---------- utilities ----------
rng = np.random.default_rng(SEED)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def layer_stats(name, G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    try:
        iso = sum(1 for _ in nx.isolates(G))
    except Exception:
        iso = None
    msg = f"[{name}] nodes={n}, edges={m}" + (f", isolates={iso}" if iso is not None else "")
    print(msg)
    return {"nodes": n, "edges": m, "isolates": iso}

def ws_edges_for_group(ids, k, p, seed=None):
    """Watts-Strogatz edges inside a set of ids; handles tiny groups gracefully."""
    ids = list(ids)
    n = len(ids)
    if n <= 1:
        return []
    if n == 2:
        return [(ids[0], ids[1])]
    # k must be < n and even-ish; make it workable per group size
    kk = min(k, n - 1 - ((n - 1) % 2))
    if kk < 2:
        kk = 2 if n >= 3 else 1
    G = nx.watts_strogatz_graph(n=n, k=kk, p=p, seed=seed)
    mapping = {i: ids[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping, copy=True)
    return list(G.edges())

def save_edges(df_edges, path):
    if df_edges is not None and not df_edges.empty:
        df_edges.to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["src","dst","layer"]).to_csv(path, index=False)


# ---------- load data ----------
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)

# Ensure Agent_ID exists
if "Agent_ID" not in df.columns:
    df = df.reset_index().rename(columns={"index": "Agent_ID"})

# Cast common fields
for col in ["Agent_ID", "Family_ID", "Work_ID", "School_ID", "Work_Status", "School_Status", "Age"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

ensure_dir(OUT_DIR)

# ---------- select company for internal layer ----------
workers = df[df["Work_Status"] == 1].copy()
if SELECTED_WORK_ID is None:
    if not workers.empty:
        SELECTED_WORK_ID = workers["Work_ID"].value_counts().idxmax()
    else:
        SELECTED_WORK_ID = df["Work_ID"].value_counts().idxmax()
SELECTED_WORK_ID = int(SELECTED_WORK_ID)
print(f"[pick-company] Work_ID={SELECTED_WORK_ID}")

emp_mask = (df["Work_ID"] == SELECTED_WORK_ID) & (df["Work_Status"] == 1)
employee_ids = sorted(df.loc[emp_mask, "Agent_ID"].dropna().astype(int).unique().tolist())
print(f"[employees in selected company] {len(employee_ids)}")

# ---------- nodes table ----------
nodes_cols = ["Agent_ID","Gender","Age","Family_ID","Work_ID","School_ID","Work_Status","School_Status"]
present_cols = [c for c in nodes_cols if c in df.columns]
nodes = df[present_cols].copy()
nodes["is_employee_selected_company"] = nodes["Agent_ID"].isin(employee_ids).astype(int)

# Fill Age NA for later use
if "Age" in nodes.columns:
    if nodes["Age"].isna().all():
        nodes["Age"] = 35  # fallback constant
    else:
        nodes["Age"] = nodes["Age"].fillna(nodes["Age"].median())

# ---------- INTERNAL (selected company; undirected) ----------
internal_k = 4      # even, < n
internal_p = 0.10

edges_internal = ws_edges_for_group(employee_ids, internal_k, internal_p, seed=SEED)
df_internal = pd.DataFrame(edges_internal, columns=["src","dst"])
if not df_internal.empty:
    df_internal["layer"] = "internal"
    df_internal["work_id"] = SELECTED_WORK_ID
G_internal = nx.from_pandas_edgelist(df_internal, "src", "dst") if not df_internal.empty else nx.Graph()
G_internal.add_nodes_from(employee_ids)  # ensure isolates exist
stats_internal = layer_stats("internal", G_internal)

# ---------- FAMILY (clique within family; undirected) ----------
edges_family = []
if "Family_ID" in df.columns:
    for fam_id, grp in df.groupby("Family_ID"):
        members = grp["Agent_ID"].dropna().astype(int).unique().tolist()
        if len(members) >= 2:
            edges_family.extend(list(combinations(members, 2)))
df_family = pd.DataFrame(edges_family, columns=["src","dst"])
if not df_family.empty:
    df_family["layer"] = "family"
G_family = nx.from_pandas_edgelist(df_family, "src", "dst") if not df_family.empty else nx.Graph()
G_family.add_nodes_from(nodes["Agent_ID"].dropna().astype(int).tolist())  # include isolates
stats_family = layer_stats("family", G_family)

# ---------- SCHOOL (per school WS; undirected) ----------
school_k = 4
school_p = 0.05
edges_school = []
if "School_Status" in df.columns and "School_ID" in df.columns:
    students = df[df["School_Status"] == 1]
    for sid, grp in students.groupby("School_ID"):
        ids = sorted(grp["Agent_ID"].dropna().astype(int).unique().tolist())
        if len(ids) == 0:
            continue
        if len(ids) == 1:
            continue
        if len(ids) == 2:
            edges_school.append((ids[0], ids[1]))
            continue
        seed_local = SEED + (int(sid) if pd.notna(sid) else 0)
        edges_school.extend(ws_edges_for_group(ids, school_k, school_p, seed=seed_local))

df_school = pd.DataFrame(edges_school, columns=["src","dst"])
if not df_school.empty:
    df_school["layer"] = "school"
G_school = nx.from_pandas_edgelist(df_school, "src", "dst") if not df_school.empty else nx.Graph()
G_school.add_nodes_from(nodes["Agent_ID"].dropna().astype(int).tolist())
stats_school = layer_stats("school", G_school)

# ---------- PERSONAL (Facebook-ish) ----------
rng = np.random.default_rng(SEED)

G_fb = nx.Graph()
ids_all = nodes["Agent_ID"].dropna().astype(int).to_numpy()
G_fb.add_nodes_from(ids_all)

age_map = dict(zip(nodes["Agent_ID"], nodes["Age"]))
school_map = dict(zip(nodes["Agent_ID"], nodes["School_ID"]))
family_map = dict(zip(nodes["Agent_ID"], nodes["Family_ID"]))

# 1) Household cliques
by_family = defaultdict(list)
for aid, fid in family_map.items():
    if pd.notna(aid) and pd.notna(fid):
        by_family[int(fid)].append(int(aid))

for fid, members in by_family.items():
    m = len(members)
    if m >= 2:
        for i in range(m):
            for j in range(i+1, m):
                G_fb.add_edge(members[i], members[j])

# 2) Same-school / alumni ties (few per person, near-age)
by_school = defaultdict(list)
for aid, sid in school_map.items():
    if pd.notna(aid) and pd.notna(sid):
        by_school[int(sid)].append(int(aid))

def pick_extra(mu=6):
    # median around 5-6
    return int(np.clip(rng.lognormal(mean=np.log(mu), sigma=0.6), 0, 50))

LAMBDA_AGE_FB = 0.08
EXTRA_PER_PERSON_CAP = 10

for u in ids_all:
    sid = school_map.get(u, None)
    pool = set(by_school.get(int(sid), [])) if pd.notna(sid) else set()
    pool.discard(int(u))
    if not pool:
        continue
    cands = list(pool)
    a_u = age_map.get(u, 35)
    w = np.array([np.exp(-LAMBDA_AGE_FB * abs((a_u if pd.notna(a_u) else 35) - (age_map.get(v, 35) if pd.notna(age_map.get(v, 35)) else 35))) for v in cands], float)
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        continue
    w /= w_sum
    k = min(pick_extra(), EXTRA_PER_PERSON_CAP, len(cands))
    chosen = rng.choice(cands, size=k, replace=False, p=w)
    for v in chosen:
        if not G_fb.has_edge(int(u), int(v)):
            G_fb.add_edge(int(u), int(v))

# 3) Triadic closure (light)
P_TC = 0.15
SAMPLES_PER_NODE = 20
for u in G_fb.nodes():
    nbrs = list(G_fb.neighbors(u))
    if len(nbrs) < 2:
        continue
    for _ in range(SAMPLES_PER_NODE):
        if len(nbrs) < 2:
            break
        a, b = rng.choice(nbrs, size=2, replace=False)
        if a != b and not G_fb.has_edge(a, b) and rng.random() < P_TC:
            G_fb.add_edge(a, b)

# 4) Weak ties (very sparse global bridges)
P_WEAK = 2e-6  # adjust for reach
M_weak = int(P_WEAK * len(ids_all) * (len(ids_all)-1) / 2)
for _ in range(M_weak):
    u, v = rng.choice(ids_all, size=2, replace=False)
    if not G_fb.has_edge(int(u), int(v)):
        G_fb.add_edge(int(u), int(v))

df_fb = pd.DataFrame([(u, v) for u, v in G_fb.edges()], columns=["src","dst"]).assign(layer="personal_facebook")
stats_fb = layer_stats("personal_facebook", G_fb)

# ---------- PROFESSIONAL (LinkedIn-like connections; optimized) ----------
t0 = time.time()

# Tunables for speed vs realism
P_MAX = 20000            # cap number of workers used to build the layer (None = all)
D_MAX = 600              # target degree clip
MU, SIG = 2.0, 0.9       # lognormal (slightly lighter tail)
ALPHA, BETA = 1.0, 0.75  # PA strength
LAMBDA_AGE = 0.03
H_SCHOOL = 0.8
H_CLUSTER = 0.4

CAND_LIMIT = 200         # max candidates per user before sampling
SPILLOVER_LIMIT = 200    # random spillover candidates
LOG_EVERY = 5000         # progress interval

# Triadic closure (thin)
P_TRIADIC = 0.10
TRIADIC_SAMPLE_PER_NODE = 30

workers_prof = nodes[nodes["Work_Status"] == 1].copy()
if P_MAX is not None and len(workers_prof) > P_MAX:
    workers_prof = workers_prof.sample(P_MAX, random_state=SEED)

# Cluster by company size decile (industry proxy)
company_sizes = workers_prof.groupby("Work_ID")["Agent_ID"].count().rename("size").sort_values(ascending=False)
qs = np.quantile(company_sizes, np.linspace(0,1,11)) if not company_sizes.empty else np.arange(11)
def size_decile(x):
    if pd.isna(x) or x not in company_sizes.index:
        return -1
    val = company_sizes.loc[x]
    return int(np.searchsorted(qs, val, side="right")-1)

workers_prof["cluster"] = workers_prof["Work_ID"].apply(size_decile)

# Ensure age numeric
workers_prof["Age"] = pd.to_numeric(workers_prof["Age"], errors="coerce")
if workers_prof["Age"].isna().any():
    workers_prof["Age"] = workers_prof["Age"].fillna(workers_prof["Age"].median())

ids_prof = workers_prof["Agent_ID"].to_numpy(dtype=int)
age = dict(zip(workers_prof["Agent_ID"], workers_prof["Age"]))
school = dict(zip(workers_prof["Agent_ID"], workers_prof["School_ID"]))
cluster = dict(zip(workers_prof["Agent_ID"], workers_prof["cluster"]))

deg_target = np.minimum((rng.lognormal(MU, SIG, size=len(workers_prof))).astype(int), D_MAX)
workers_prof = workers_prof.assign(D_target=deg_target)

by_school_prof = defaultdict(list)
for aid, sid in zip(workers_prof["Agent_ID"], workers_prof["School_ID"]):
    by_school_prof[sid].append(int(aid))
by_cluster_prof = defaultdict(list)
for aid, cid in zip(workers_prof["Agent_ID"], workers_prof["cluster"]):
    by_cluster_prof[cid].append(int(aid))

G_prof = nx.Graph()
G_prof.add_nodes_from(ids_prof)
deg = defaultdict(int)
all_arr = ids_prof  # ndarray for fast sampling

def score(u, v):
    s = (deg[v] + ALPHA)**BETA
    if school.get(u) == school.get(v) and pd.notna(school.get(u)):
        s *= (1.0 + H_SCHOOL)
    if cluster.get(u) == cluster.get(v) and cluster.get(u) != -1:
        s *= (1.0 + H_CLUSTER)
    au, av = age.get(u), age.get(v)
    if au is not None and av is not None:
        s *= np.exp(-LAMBDA_AGE * abs(au - av))
    return s

N = len(ids_prof)
for idx, (u, Du) in enumerate(zip(ids_prof, workers_prof["D_target"].to_numpy())):
    if Du <= 0:
        if (idx % LOG_EVERY) == 0:
            print(f"[professional] progress {idx}/{N} (edges={G_prof.number_of_edges()})")
        continue

    # candidate pool: same school + same cluster + spillover
    cands = set()
    sid = school.get(u, None)
    cid = cluster.get(u, None)
    if sid in by_school_prof: cands.update(by_school_prof[sid])
    if cid in by_cluster_prof: cands.update(by_cluster_prof[cid])

    # random spillover
    if len(cands) < SPILLOVER_LIMIT and N > 1:
        needed = SPILLOVER_LIMIT - len(cands)
        spill = rng.choice(all_arr, size=min(needed, N-1), replace=False)
        cands.update(int(x) for x in spill)

    cands.discard(u)
    if not cands:
        if (idx % LOG_EVERY) == 0:
            print(f"[professional] progress {idx}/{N} (edges={G_prof.number_of_edges()})")
        continue

    # cap candidate set
    if len(cands) > CAND_LIMIT:
        cands = set(rng.choice(list(cands), size=CAND_LIMIT, replace=False))

    cands = list(cands)
    w = np.array([score(u, v) for v in cands], dtype=float)
    ssum = w.sum()
    if not np.isfinite(ssum) or ssum <= 0:
        if (idx % LOG_EVERY) == 0:
            print(f"[professional] progress {idx}/{N} (edges={G_prof.number_of_edges()})")
        continue
    w /= ssum

    k = int(min(Du, len(cands)))
    chosen = rng.choice(cands, size=k, replace=False, p=w)
    for v in chosen:
        if u != v and not G_prof.has_edge(u, v):
            G_prof.add_edge(u, v)
            deg[u] += 1; deg[v] += 1

    if (idx % LOG_EVERY) == 0:
        print(f"[professional] progress {idx}/{N} (edges={G_prof.number_of_edges()})")

# Triadic closure (thin)
if P_TRIADIC > 0 and TRIADIC_SAMPLE_PER_NODE > 0:
    print("[professional] triadic closure...")
    for idx, u in enumerate(G_prof.nodes()):
        nbrs = list(G_prof.neighbors(u))
        d = len(nbrs)
        if d >= 2:
            # sample a few random neighbor pairs per node
            seen_pairs = set()
            for _ in range(TRIADIC_SAMPLE_PER_NODE):
                if d < 2:
                    break
                a, b = rng.choice(nbrs, size=2, replace=False)
                if a > b:
                    a, b = b, a
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                if not G_prof.has_edge(a, b) and rng.random() < P_TRIADIC:
                    G_prof.add_edge(a, b)
                    deg[a] += 1; deg[b] += 1
        if (idx % (LOG_EVERY // 2 or 1)) == 0:
            print(f"[professional] triadic progress {idx}/{G_prof.number_of_nodes()}")

df_prof = pd.DataFrame([(u, v) for u, v in G_prof.edges()], columns=["src","dst"])
if not df_prof.empty:
    df_prof["layer"] = "professional"
stats_prof = layer_stats("professional", G_prof)
print(f"[professional] built in {time.time()-t0:.1f}s")

# ---------- FOLLOWS (directed) ----------
t1 = time.time()
FOLLOW_EMP_COMPANY = 0.95
FOLLOW_EXT_COMPANY = 0.10
K_INFLUENCERS = 10
FOLLOW_TO_INFLUENCER_BASE = 0.005
FOLLOW_PA_BETA = 0.9

F = nx.DiGraph()
page_node = f"PAGE_{SELECTED_WORK_ID}"
F.add_node(page_node)

emp_ids_sel = [int(x) for x in employee_ids if x in set(ids_prof)]
ext_ids_sel = [int(u) for u in ids_prof if u not in set(emp_ids_sel)]

# Company page follows
if emp_ids_sel:
    mask = rng.random(len(emp_ids_sel)) < FOLLOW_EMP_COMPANY
    for u in np.array(emp_ids_sel)[mask]:
        F.add_edge(int(u), page_node)
if ext_ids_sel:
    mask = rng.random(len(ext_ids_sel)) < FOLLOW_EXT_COMPANY
    for u in np.array(ext_ids_sel)[mask]:
        F.add_edge(int(u), page_node)

# Influencers with simple PA-follow
influencers = [f"INFL_{i}" for i in range(K_INFLUENCERS)]
F.add_nodes_from(influencers)
follower_count = defaultdict(int)
for u in ids_prof:
    # consider a small subset of influencers
    k_consider = min(3, K_INFLUENCERS)
    infl_subset = rng.choice(influencers, size=k_consider, replace=False)
    for infl in infl_subset:
        base = FOLLOW_TO_INFLUENCER_BASE
        pa_boost = (follower_count[infl] + 1)**FOLLOW_PA_BETA
        p_follow = 1.0 - np.exp(-base * pa_boost)
        if rng.random() < p_follow:
            F.add_edge(int(u), infl)
            follower_count[infl] += 1

df_follow = pd.DataFrame([(u, v) for u, v in F.edges()], columns=["src","dst"])
if not df_follow.empty:
    df_follow["layer"] = "follow"
print(f"[follow] nodes={F.number_of_nodes()}, edges={F.number_of_edges()} (in {time.time()-t1:.1f}s)")

# ---------- save all (FULL) ----------
print("Saving CSVs (full)...")
nodes_path          = os.path.join(OUT_DIR, "nodes.csv")
edges_internal_path = os.path.join(OUT_DIR, "edges_internal.csv")
edges_family_path   = os.path.join(OUT_DIR, "edges_family.csv")
edges_school_path   = os.path.join(OUT_DIR, "edges_school.csv")
edges_fb_path       = os.path.join(OUT_DIR, "edges_personal_facebook.csv")
edges_prof_path     = os.path.join(OUT_DIR, "edges_professional.csv")
edges_follow_path   = os.path.join(OUT_DIR, "edges_follow.csv")
edges_all_path      = os.path.join(OUT_DIR, "edges_all.csv")
summary_path        = os.path.join(OUT_DIR, "summary.json")

nodes.to_csv(nodes_path, index=False)
save_edges(df_internal, edges_internal_path)
save_edges(df_family, edges_family_path)
save_edges(df_school, edges_school_path)
save_edges(df_fb, edges_fb_path)
save_edges(df_prof, edges_prof_path)
save_edges(df_follow, edges_follow_path)

# Combine all edges
dfs_all = []
for p in [edges_internal_path, edges_family_path, edges_school_path, edges_fb_path, edges_prof_path, edges_follow_path]:
    if os.path.exists(p):
        part = pd.read_csv(p)
        if not part.empty:
            dfs_all.append(part)
edges_all_df = pd.concat(dfs_all, ignore_index=True) if dfs_all else pd.DataFrame(columns=["src","dst","layer"])
edges_all_df.to_csv(edges_all_path, index=False)

# ---------- summary (FULL) ----------
summary = {
    "selected_work_id": SELECTED_WORK_ID,
    "counts": {
        "nodes_total": int(nodes.shape[0]),
        "internal": stats_internal,
        "family": stats_family,
        "school": stats_school,
        "personal_facebook": layer_stats("personal_facebook(recap)", G_fb),
        "professional": layer_stats("professional(recap)", G_prof),
        "follow": {"nodes": F.number_of_nodes(), "edges": F.number_of_edges()},
    },
    "paths": {
        "nodes": nodes_path,
        "edges_internal": edges_internal_path,
        "edges_family": edges_family_path,
        "edges_school": edges_school_path,
        "edges_personal_facebook": edges_fb_path,
        "edges_professional": edges_prof_path,
        "edges_follow": edges_follow_path,
        "edges_all": edges_all_path,
    },
    "parameters": {
        "seed": SEED,
        "internal_k": internal_k,
        "internal_p": internal_p,
        "school_k": school_k,
        "school_p": school_p,
        "personal_facebook": {
            "lambda_age": float(0.08),
            "extra_per_person_cap": EXTRA_PER_PERSON_CAP,
            "p_tc": P_TC,
            "samples_per_node": SAMPLES_PER_NODE,
            "p_weak": P_WEAK
        },
        "professional": {
            "P_MAX": P_MAX, "MU": MU, "SIG": SIG, "D_MAX": D_MAX,
            "ALPHA": ALPHA, "BETA": BETA, "LAMBDA_AGE": LAMBDA_AGE,
            "H_SCHOOL": H_SCHOOL, "H_CLUSTER": H_CLUSTER,
            "P_TRIADIC": P_TRIADIC, "TRIADIC_SAMPLE_PER_NODE": TRIADIC_SAMPLE_PER_NODE,
            "CAND_LIMIT": CAND_LIMIT, "SPILLOVER_LIMIT": SPILLOVER_LIMIT
        },
        "follows": {
            "FOLLOW_EMP_COMPANY": FOLLOW_EMP_COMPANY,
            "FOLLOW_EXT_COMPANY": FOLLOW_EXT_COMPANY,
            "K_INFLUENCERS": K_INFLUENCERS,
            "FOLLOW_TO_INFLUENCER_BASE": FOLLOW_TO_INFLUENCER_BASE,
            "FOLLOW_PA_BETA": FOLLOW_PA_BETA
        }
    }
}

# ---------- PRUNE ISOLATES (GLOBAL ACTIVE SET) ----------
if PRUNE_ISOLATES:
    print("Pruning isolates (global active set) and saving *_active CSVs...")
    # 1) Active set = κόμβοι που συμμετέχουν σε ΚΑΠΟΙΑ ακμή (σε οποιοδήποτε layer)
    active = set()

    # Undirected layers
    for G in [G_internal, G_family, G_school, G_fb, G_prof]:
        active.update([n for n, d in G.degree() if d > 0])

    # Directed follows: in_degree + out_degree > 0
    active.update([n for n in F.nodes() if (F.in_degree(n) + F.out_degree(n)) > 0])

    # Μόνο άτομα στο nodes_active (όχι PAGE_*, INFL_*)
    person_ids = set(nodes["Agent_ID"].dropna().astype(int).tolist())
    active_persons = [aid for aid in active if isinstance(aid, (int, np.integer)) and aid in person_ids]

    # 2) Φιλτράρισμα nodes & edges
    nodes_active = nodes[nodes["Agent_ID"].isin(active_persons)].copy()

    def prune_edges_df(df_edges):
        if df_edges is None or df_edges.empty:
            return df_edges
        keep = df_edges["src"].isin(active) & df_edges["dst"].isin(active)
        return df_edges[keep].copy()

    df_internal_active = prune_edges_df(df_internal)
    df_family_active   = prune_edges_df(df_family)
    df_school_active   = prune_edges_df(df_school)
    df_fb_active       = prune_edges_df(df_fb)
    df_prof_active     = prune_edges_df(df_prof)
    df_follow_active   = prune_edges_df(df_follow)  # κρατά και PAGE_*, INFL_*

    # 3) Γράψιμο αρχείων _active
    nodes_active_path          = os.path.join(OUT_DIR, "nodes_active.csv")
    edges_internal_active_path = os.path.join(OUT_DIR, "edges_internal_active.csv")
    edges_family_active_path   = os.path.join(OUT_DIR, "edges_family_active.csv")
    edges_school_active_path   = os.path.join(OUT_DIR, "edges_school_active.csv")
    edges_fb_active_path       = os.path.join(OUT_DIR, "edges_personal_facebook_active.csv")
    edges_prof_active_path     = os.path.join(OUT_DIR, "edges_professional_active.csv")
    edges_follow_active_path   = os.path.join(OUT_DIR, "edges_follow_active.csv")
    edges_all_active_path      = os.path.join(OUT_DIR, "edges_all_active.csv")

    nodes_active.to_csv(nodes_active_path, index=False)
    for df_, path_ in [
        (df_internal_active, edges_internal_active_path),
        (df_family_active,   edges_family_active_path),
        (df_school_active,   edges_school_active_path),
        (df_fb_active,       edges_fb_active_path),
        (df_prof_active,     edges_prof_active_path),
        (df_follow_active,   edges_follow_active_path),
    ]:
        save_edges(df_, path_)

    # Ενοποιημένο ενεργό edge list
    dfs_active = [x for x in [
        df_internal_active, df_family_active, df_school_active,
        df_fb_active, df_prof_active, df_follow_active
    ] if x is not None and not x.empty]
    edges_all_active_df = pd.concat(dfs_active, ignore_index=True) if dfs_active else pd.DataFrame(columns=["src","dst","layer"])
    edges_all_active_df.to_csv(edges_all_active_path, index=False)

    # 4) Active layer stats για σύνοψη
    G_int_a  = nx.from_pandas_edgelist(df_internal_active, "src", "dst") if df_internal_active is not None and not df_internal_active.empty else nx.Graph()
    G_fam_a  = nx.from_pandas_edgelist(df_family_active,   "src", "dst") if df_family_active   is not None and not df_family_active.empty   else nx.Graph()
    G_sch_a  = nx.from_pandas_edgelist(df_school_active,   "src", "dst") if df_school_active   is not None and not df_school_active.empty   else nx.Graph()
    G_fb_a   = nx.from_pandas_edgelist(df_fb_active,       "src", "dst") if df_fb_active       is not None and not df_fb_active.empty       else nx.Graph()
    G_prof_a = nx.from_pandas_edgelist(df_prof_active,     "src", "dst") if df_prof_active     is not None and not df_prof_active.empty     else nx.Graph()
    F_a      = nx.from_pandas_edgelist(df_follow_active, "src", "dst", create_using=nx.DiGraph) if df_follow_active is not None and not df_follow_active.empty else nx.DiGraph()

    summary["counts_active"] = {
        "nodes_total_active": int(nodes_active.shape[0]),
        "internal": layer_stats("internal_active", G_int_a),
        "family": layer_stats("family_active", G_fam_a),
        "school": layer_stats("school_active", G_sch_a),
        "personal_facebook": layer_stats("personal_facebook_active", G_fb_a),
        "professional": layer_stats("professional_active", G_prof_a),
        "follow": {"nodes": F_a.number_of_nodes(), "edges": F_a.number_of_edges()}
    }
    summary["paths_active"] = {
        "nodes": nodes_active_path,
        "edges_internal": edges_internal_active_path,
        "edges_family": edges_family_active_path,
        "edges_school": edges_school_active_path,
        "edges_personal_facebook": edges_fb_active_path,
        "edges_professional": edges_prof_active_path,
        "edges_follow": edges_follow_active_path,
        "edges_all": edges_all_active_path,
    }

# ---------- write summary ----------
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nAll done.")
print("Saved to:", OUT_DIR)
print("Quick counts:", json.dumps(summary.get("counts", {}), indent=2))
if PRUNE_ISOLATES:
    print("Quick counts (active):", json.dumps(summary.get("counts_active", {}), indent=2))
