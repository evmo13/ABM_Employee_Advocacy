from __future__ import annotations
import math, random, time, os, csv, argparse, datetime, logging, json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd
import networkx as nx
from mpi4py import MPI

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="SEAD (edge-list mode only) with per-layer logging")

    # Required / optional inputs
    p.add_argument("--nodes", required=True, help="Path to nodes.csv (must include Age; optional Work_Status or explicit role cols)")
    p.add_argument("--edges-internal", default=None, help="Path to edges_internal.csv")
    p.add_argument("--edges-professional", default=None, help="Path to edges_professional.csv")
    p.add_argument("--edges-school", default=None, help="Path to edges_school.csv (merged into professional)")
    p.add_argument("--edges-personal", default=None, help="Path to edges_personal.csv")
    p.add_argument("--edges-family", default=None, help="Path to edges_family.csv (merged into personal)")

    # General
    p.add_argument("--out", default=None, help="Output CSV for per-layer time series (default: logs/sead_timeseries_<ts>.csv)")
    p.add_argument("--steps", type=int, default=30, help="Simulation steps")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--aggressive", action="store_true", help="Use aggressive diffusion preset")

    # New: JSON config overrides
    p.add_argument("--config", default=None, help="Path to JSON file with overrides (applied after --aggressive if present)")

    return p.parse_args()

# ----------------------- CONFIG -----------------------
# Layer coupling (P(post on TO | seen on FROM))
DEFAULT_KAPPA = {
    "internal":     {"internal": 0.9, "professional": 0.7, "personal": 0.6},
    "professional": {"internal": 0.3, "professional": 0.9, "personal": 0.5},
    "personal":     {"internal": 0.2, "professional": 0.4, "personal": 0.9},
}

# Visibility / attention
LAYER_VIS = {"internal": 0.60, "professional": 0.35, "personal": 0.45}
LAYER_ATTN_K = {"internal": 10, "professional": 8, "personal": 8}
VISIBILITY_SLOPE = 0.10

# Age homophily
AGE_GAP = 5
P_AGE_IN = 1.0
P_AGE_OUT = 0.6
A7_AGE_SIM = 0.25

# Dynamics
INCENTIVE_LEVEL = 0.3
CSMA_LEVEL = 0.4
PEER_PRESSURE_ALPHA = 0.5

# Aggressive preset (overrides)
AGGR_PRESET = dict(
    DEFAULT_KAPPA={
        "internal":     {"internal": 0.95, "professional": 0.90, "personal": 0.80},
        "professional": {"internal": 0.40, "professional": 0.95, "personal": 0.75},
        "personal":     {"internal": 0.30, "professional": 0.70, "personal": 0.95},
    },
    LAYER_VIS={"internal": 0.80, "professional": 0.55, "personal": 0.60},
    LAYER_ATTN_K={"internal": 20, "professional": 16, "personal": 16},
    VISIBILITY_SLOPE=0.20,
    AGE_GAP=10, P_AGE_OUT=0.9, A7_AGE_SIM=0.15,
    INCENTIVE_LEVEL=0.6, CSMA_LEVEL=0.7, PEER_PRESSURE_ALPHA=0.7,
    EMPLOYEE_COOLDOWN=0, COMPLIANCE_THRESHOLD=0.9, MEAN_CONTENTS_PER_TICK=2.0,
)

# ----------------------- Utils -----------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

LAYER_NAMES = ("internal", "professional", "personal")

class SEA(Enum):
    S = auto(); E = auto(); A = auto(); D = auto()

@dataclass
class Content:
    cid: int
    topic_vec: np.ndarray
    quality: float
    sensitivity: float
    geo_relevance: float
    timestamp: int
    social_proof: float = 0.0
    shares: int = 0
    engagements: int = 0
    def decay(self, rho: float = 0.9) -> None:
        self.social_proof *= rho

# ---------------- Helpers for robust CSV I/O ----------------
def _read_csv_robust(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1", **kw)

def _detect_edge_cols(df_edges: pd.DataFrame) -> Tuple[str, str]:
    cols = [c.lower() for c in df_edges.columns]
    pairs = [
        ("src", "dst"), ("source", "target"), ("u", "v"),
        ("from", "to"), ("i", "j"), ("a", "b"),
        ("node1", "node2"), ("id1", "id2")
    ]
    for a, b in pairs:
        if a in cols and b in cols:
            A = df_edges.columns[cols.index(a)]
            B = df_edges.columns[cols.index(b)]
            return A, B
    return df_edges.columns[0], df_edges.columns[1]

def _edge_list_df_to_tuples(df_edges: pd.DataFrame, id_map: Dict[object,int]) -> List[Tuple[int,int]]:
    if df_edges is None or df_edges.empty:
        return []
    ucol, vcol = _detect_edge_cols(df_edges)
    edges = []
    for _, row in df_edges.iterrows():
        u0 = row[ucol]; v0 = row[vcol]
        if pd.isna(u0) or pd.isna(v0): 
            continue
        if u0 not in id_map or v0 not in id_map: 
            continue
        u = int(id_map[u0]); v = int(id_map[v0])
        if u != v:
            edges.append((min(u, v), max(u, v)))
    return list(set(edges))

def _load_undirected_graph(n: int, edges: List[Tuple[int,int]]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G

# ---------------- Data adapter: EDGE LISTS ONLY ----------------
def build_graphs_from_edge_lists(
    nodes_path: str,
    edges_internal_path: Optional[str],
    edges_prof_path: Optional[str],
    edges_school_path: Optional[str],
    edges_personal_path: Optional[str],
    edges_family_path: Optional[str],
    seed: int = 123,
) -> Tuple[nx.Graph, nx.Graph, nx.Graph, Dict[str, np.ndarray], np.ndarray]:
    """
    Returns (G_internal, G_professional, G_personal, roles, ages)
    - IDs are remapped to 0..N-1 using nodes.csv
    - Professional = union(professional, school)
    - Personal = union(personal, family)
    - Roles:
        1) If nodes.csv has explicit Employee/Peer/Customer/Public -> use them.
        2) Else, if internal layer has edges -> employees = nodes with deg>0 in internal.
           Then (optional) peers from Work_Status==1 excluding employees (60% of leftovers).
           Remaining split to customers/public 50/50.
        3) Else, attempt inference from Work_Status like (2) but without internal constraint.
    """
    rng = np.random.default_rng(seed)

    # ---- nodes
    df_nodes = _read_csv_robust(nodes_path)

    # Detect id column
    id_col = None
    for cand in ["agent_id", "id", "node_id", "person_id", "uid", "Id", "ID"]:
        if cand in df_nodes.columns:
            id_col = cand; break
    if id_col is None:
        id_col = df_nodes.columns[0]
    raw_ids = df_nodes[id_col].tolist()
    id_map = {raw_id: i for i, raw_id in enumerate(raw_ids)}
    N = len(raw_ids)

    # Ages
    age_col = None
    for cand in ["Age", "age", "AGE"]:
        if cand in df_nodes.columns:
            age_col = cand; break
    if age_col is None:
        raise ValueError("nodes.csv must include an Age column.")
    ages = df_nodes[age_col].astype(int).to_numpy()

    def _has(col): return col in df_nodes.columns

    work_status = df_nodes["Work_Status"].fillna(0).astype(int).to_numpy() if _has("Work_Status") else np.zeros(N, dtype=int)

    # ---- edges
    def _load_edges(path):
        if path is None: return []
        df_e = _read_csv_robust(path)
        return _edge_list_df_to_tuples(df_e, id_map)

    E_int = _load_edges(edges_internal_path)
    E_prof = _load_edges(edges_prof_path)
    E_school = _load_edges(edges_school_path)
    E_pers = _load_edges(edges_personal_path)
    E_fam = _load_edges(edges_family_path)

    # Unions
    E_prof_all = list(set(E_prof + E_school))
    E_personal_all = list(set(E_pers + E_fam))

    # Graphs
    G_internal = _load_undirected_graph(N, E_int)
    G_prof = _load_undirected_graph(N, E_prof_all)
    G_personal = _load_undirected_graph(N, E_personal_all)

    # Roles
    has_explicit_roles = all(_has(c) for c in ["Employee","Peer","Customer","Public"])
    if has_explicit_roles:
        employee_ids = np.where(df_nodes["Employee"].astype(int).to_numpy() == 1)[0]
        peer_ids     = np.where(df_nodes["Peer"].astype(int).to_numpy() == 1)[0]
        customer_ids = np.where(df_nodes["Customer"].astype(int).to_numpy() == 1)[0]
        public_ids   = np.where(df_nodes["Public"].astype(int).to_numpy() == 1)[0]
    else:
        # Prefer internal membership as employee set if internal has edges
        if G_internal.number_of_edges() > 0:
            employee_ids = np.array([n for n, d in G_internal.degree() if d > 0], dtype=int)
        else:
            # fallback: small random slice of workers as employees (2%)
            workers = np.where(work_status == 1)[0]
            n_emp = max(1, int(0.02 * len(workers))) if len(workers)>0 else 0
            employee_ids = np.sort(rng.choice(workers, size=n_emp, replace=False)) if n_emp>0 else np.array([], dtype=int)

        # Peers: from remaining workers (60%), if available
        workers_left = np.array([i for i in np.where(work_status==1)[0] if i not in set(employee_ids)], dtype=int)
        n_peers = int(0.60 * len(workers_left)) if len(workers_left)>0 else 0
        peer_ids = np.sort(rng.choice(workers_left, size=n_peers, replace=False)) if n_peers>0 else np.array([], dtype=int)

        everyone = np.arange(N, dtype=int)
        leftover = np.setdiff1d(everyone, np.union1d(employee_ids, peer_ids), assume_unique=True)
        half = len(leftover) // 2
        customer_ids = leftover[:half]
        public_ids = leftover[half:]

    roles = {
        "employee_ids": np.array(sorted(employee_ids), dtype=int),
        "peer_ids": np.array(sorted(peer_ids), dtype=int),
        "customer_ids": np.array(sorted(customer_ids), dtype=int),
        "public_ids": np.array(sorted(public_ids), dtype=int),
    }
    return G_internal, G_prof, G_personal, roles, ages

# ---------------- Logger ----------------
class RunLogger:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self._writer = None
        self._fh = None

    def open(self, fieldnames: List[str]):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self._fh = open(self.out_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        self._writer.writeheader()

    def write(self, row: Dict[str, object]):
        self._writer.writerow(row)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

# ---------------- ABM core ----------------
def union_neighbors(nid: int, G_internal: nx.Graph, G_prof: nx.Graph, G_personal: nx.Graph) -> List[int]:
    neigh: Set[int] = set()
    if nid in G_internal:  neigh.update(G_internal.neighbors(nid))
    if nid in G_prof:      neigh.update(G_prof.neighbors(nid))
    if nid in G_personal:  neigh.update(G_personal.neighbors(nid))
    neigh.discard(nid)
    return list(neigh)

class BaseNode:
    def __init__(self, unique_id: int, model: "DistributedModel"):
        self.unique_id = unique_id
        self.model = model
        self.inbox_by_layer: Dict[str, List[int]] = {ln: [] for ln in LAYER_NAMES}
        self.location: Tuple[float, float] = (random.random(), random.random())
        self.age: Optional[int] = None
        self.age_group: Optional[str] = None

class EmployeeAgent(BaseNode):
    def __init__(self, unique_id: int, model: "DistributedModel"):
        super().__init__(unique_id, model)
        self.org_id = random.betavariate(2, 2)
        self.risk_tol = random.betavariate(2, 5)
        self.compliance_awareness = random.betavariate(3, 2)
        self.audience_size = int(np.random.lognormal(mean=2.0, sigma=1.0))
        self.topic_vec = np.random.normal(0, 1, size=model.topic_dim)
        self.past_reward = 0.0
        self.cooldown = 0

    def p_share(self, c: Content, peer_press: float, csma_t: float) -> float:
        fit = cosine_similarity(self.topic_vec, c.topic_vec)
        risk = c.sensitivity * (1.0 - self.compliance_awareness) * (1.0 - self.risk_tol)
        incentive = self.model.incentive_level
        geo_rel = c.geo_relevance
        z = (
            self.model.w1 * self.org_id
            + self.model.w2 * fit
            + self.model.w3 * incentive
            - self.model.w4 * risk
            + self.model.w5 * peer_press
            + self.model.w6 * self.past_reward
            + self.model.w7 * geo_rel
            + self.model.w8 * csma_t
        )
        base = sigmoid(z)
        compliant = (c.sensitivity <= self.model.compliance_threshold)
        if not compliant or self.cooldown > 0:
            return 0.0
        return base

    def step(self) -> None:
        csma_t = self.model.csma_level
        visible = self.model.visible_content_for(self)
        for (cid, seen_layer) in visible:
            c = self.model.contents[cid]
            peer_press = self.model.get_peer_pressure(self, layer=seen_layer)
            if random.random() < self.p_share(c, peer_press, csma_t):
                src_info = self.model.last_source_of(cid, self.unique_id)
                if src_info:
                    p_id, p_layer = src_info
                    self.model.parents.setdefault(cid, {})
                    self.model.parents[cid][self.unique_id] = (p_id, p_layer)
                    if p_layer != seen_layer:
                        self.model.interlayer[(p_layer, seen_layer)] += 1
                for to_layer in self.model.sample_share_layers(seen_layer):
                    if src_info:
                        p_id, p_layer = src_info
                        if p_layer != to_layer:
                            self.model.interlayer[(p_layer, to_layer)] += 1
                    self.model.share_content(self, c, layer=to_layer)
                self.model.employee_shared.add(self.unique_id)
                self.past_reward = 0.7 * self.past_reward + 0.3 * (0.5 + 0.5 * c.quality)
                self.cooldown = self.model.employee_cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
        for ln in LAYER_NAMES:
            self.inbox_by_layer[ln].clear()

class AudienceAgent(BaseNode):
    def __init__(self, unique_id: int, model: "DistributedModel", role: str):
        super().__init__(unique_id, model)
        self.role = role  # "customer" | "peer" | "public"
        self.state: SEA = SEA.S
        self.brand_affinity = random.betavariate(2, 2)
        self.skepticism = random.betavariate(2, 5)
        self.identity_fit = random.betavariate(2, 2)
        self.topic_vec = np.random.normal(0, 1, size=model.topic_dim)

    def p_engage(self, c: Content, src: BaseNode) -> float:
        tie = self.model.tie_strength(src, self)
        cred = self.model.credibility(src)
        topic_match = cosine_similarity(self.topic_vec, c.topic_vec)
        social_proof = c.social_proof
        geo_rel = c.geo_relevance
        age_sim = 0.0
        if hasattr(src, "age") and self.age is not None and getattr(src, "age", None) is not None:
            age_sim = self.model.age_similarity(src.age, self.age)
        z = (
            self.model.a1 * tie
            + self.model.a2 * cred
            + self.model.a3 * topic_match
            + self.model.a4 * social_proof
            - self.model.a5 * self.skepticism
            + self.model.a6 * geo_rel
            + self.model.a7_age_sim * age_sim
        )
        return sigmoid(z)

    def p_advocate(self, c: Content) -> float:
        z = (
            self.model.b1 * (0.5 + 0.5 * c.quality)
            + self.model.b2 * self.identity_fit
            - self.model.b3 * c.sensitivity
            + self.model.b4 * self.brand_affinity
        )
        return sigmoid(z)

    def step(self) -> None:
        if self.state == SEA.D and any(self.inbox_by_layer[ln] for ln in LAYER_NAMES):
            self.state = SEA.S
            self.model.tick["d_to_s"] += 1
            for ln in LAYER_NAMES:
                if self.inbox_by_layer[ln]:
                    self.model.tick[f"d_to_s_{ln}"] += 1
                    break

        visible = []
        for ln in LAYER_NAMES:
            for cid in set(self.inbox_by_layer[ln]):
                visible.append((cid, ln))

        engaged_this_tick = False
        for (cid, seen_layer) in visible:
            c = self.model.contents[cid]
            src_info = self.model.any_source_of_on(cid, self.unique_id)
            if src_info is None:
                continue
            src_id, _src_layer = src_info
            src = self.model.agents_by_id.get(src_id)
            if src is None:
                continue

            pe = self.p_engage(c, src)
            if random.random() < pe:
                if not engaged_this_tick and self.state == SEA.S:
                    self.model.tick["s_to_e"] += 1
                    self.model.tick[f"s_to_e_{seen_layer}"] += 1
                engaged_this_tick = True
                c.engagements += 1
                c.social_proof += 1.0
                self.model.tick["engagements"] += 1
                self.model.tick[f"engagements_{seen_layer}"] += 1
                self.state = SEA.E

                # audience re-share (only professional/personal)
                if random.random() < pe:
                    to_layers = [l for l in self.model.sample_share_layers(seen_layer)
                                 if l in ("professional", "personal")]
                    p = self.model.last_source_of(c.cid, self.unique_id)
                    for to_layer in to_layers:
                        if p:
                            p_id, p_layer = p
                            if p_layer != to_layer:
                                self.model.interlayer[(p_layer, to_layer)] += 1
                        self.model.reshare_content(self, c, layer=to_layer)

                if random.random() < self.p_advocate(c):
                    self.model.tick["e_to_a"] += 1
                    self.model.tick[f"e_to_a_{seen_layer}"] += 1
                    self.state = SEA.A

        prev = self.state
        if random.random() < self.model.dormancy_rate:
            self.state = SEA.D
            last_seen_layer = visible[-1][1] if visible else None
            if prev == SEA.E:
                self.model.tick["e_to_d"] += 1
                if last_seen_layer: self.model.tick[f"e_to_d_{last_seen_layer}"] += 1
            elif prev == SEA.A:
                self.model.tick["a_to_d"] += 1
                if last_seen_layer: self.model.tick[f"a_to_d_{last_seen_layer}"] += 1

        for ln in LAYER_NAMES:
            self.inbox_by_layer[ln].clear()

# ---------------- Model ----------------
class DistributedModel:
    def __init__(
        self,
        G_internal: nx.Graph, G_prof: nx.Graph, G_personal: nx.Graph,
        employee_ids: List[int], peer_ids: List[int], customer_ids: List[int], public_ids: List[int],
        ages: Optional[np.ndarray] = None,
        # dynamics
        topic_dim: int = 8, steps: int = 50,
        incentive_level: float = 0.2, csma_level: float = 0.3,
        compliance_threshold: float = 0.6, employee_cooldown: int = 1,
        # employee share weights
        w1: float = 1.0, w2: float = 1.2, w3: float = 0.6, w4: float = 1.0,
        w5: float = 0.8, w6: float = 0.5, w7: float = 0.3, w8: float = 0.7,
        # audience weights
        a1: float = 1.0, a2: float = 0.8, a3: float = 1.0, a4: float = 0.6,
        a5: float = 0.8, a6: float = 0.3, b1: float = 0.8, b2: float = 0.7, b3: float = 0.6, b4: float = 0.5,
        # content process
        mean_contents_per_tick: float = 1.0, social_proof_decay: float = 0.9, dormancy_rate: float = 0.02,
        # visibility
        base_visibility: float = 0.15, visibility_slope: float = VISIBILITY_SLOPE, geo_decay: float = 0.0,
        layer_visibility: Optional[Dict[str, float]] = None,
        layer_attn_k: Optional[Dict[str, int]] = None,
        # brand
        brand_lambda: float = 0.95, brand_eta_pos: float = 0.01, brand_eta_neg: float = 0.02,
        # coupling & norms
        layer_coupling: Optional[Dict[str, Dict[str, float]]] = None,
        peer_pressure_alpha: float = PEER_PRESSURE_ALPHA,
        # age homophily
        age_gap: int = AGE_GAP, p_age_in: float = P_AGE_IN, p_age_out: float = P_AGE_OUT, a7_age_sim: float = A7_AGE_SIM,
        # misc
        seed: Optional[int] = 123, comm: Optional[MPI.Comm] = None,
        print_all_ranks: bool = False, rank_tag: bool = True,
        timing: bool = True, timing_every: int = 1,
        # logging
        history_csv: Optional[str] = None,
    ):
        # MPI
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed + self.rank)

        # graphs & roles
        self.G_internal = G_internal
        self.G_prof = G_prof
        self.G_personal = G_personal
        self.employee_ids = list(employee_ids)
        self.peer_ids = list(peer_ids)
        self.customer_ids = list(customer_ids)
        self.public_ids = list(public_ids)

        # total agents must be 0..N-1
        all_ids = sorted(set(self.employee_ids) | set(self.peer_ids) | set(self.customer_ids) | set(self.public_ids))
        self.total_agents = len(all_ids)
        assert all_ids == list(range(self.total_agents)), "Agent ids must be 0..N-1; check input files or role derivation."

        # ownership partition
        per = self.total_agents // self.size; rem = self.total_agents % self.size
        self.owners = np.empty(self.total_agents, dtype=np.int32)
        start = 0
        for r in range(self.size):
            n = per + (1 if r < rem else 0)
            self.owners[start:start+n] = r
            if r == self.rank:
                self.local_start, self.local_end = start, start + n
            start += n
        self.local_agent_ids = list(range(self.local_start, self.local_end))

        # params
        self.topic_dim, self.steps = topic_dim, steps
        self.incentive_level, self.csma_level = incentive_level, csma_level
        self.compliance_threshold, self.employee_cooldown = compliance_threshold, employee_cooldown
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4
        self.w5, self.w6, self.w7, self.w8 = w5, w6, w7, w8
        self.a1, self.a2, self.a3, self.a4 = a1, a2, a3, a4
        self.a5, self.a6, self.b1, self.b2, self.b3, self.b4 = a5, a6, b1, b2, b3, b4
        self.mean_contents_per_tick, self.social_proof_decay, self.dormancy_rate = mean_contents_per_tick, social_proof_decay, dormancy_rate
        self.base_visibility, self.visibility_slope, self.geo_decay = base_visibility, visibility_slope, geo_decay
        self.brand_lambda, self.brand_eta_pos, self.brand_eta_neg = brand_lambda, brand_eta_pos, brand_eta_neg
        self.layer_visibility = layer_visibility or LAYER_VIS
        self.layer_attn_k = layer_attn_k or LAYER_ATTN_K
        self.layer_coupling = layer_coupling or DEFAULT_KAPPA
        self.peer_pressure_alpha = float(np.clip(peer_pressure_alpha, 0.0, 1.0))

        # age homophily
        self.age_gap = int(age_gap)
        self.p_age_in = float(np.clip(p_age_in, 0.0, 1.0))
        self.p_age_out = float(np.clip(p_age_out, 0.0, 1.0))
        self.a7_age_sim = float(a7_age_sim)

        # content store & metrics
        self.contents: Dict[int, Content] = {}
        self.next_cid = 0
        self.deliveries: Dict[int, Set[Tuple[int, int, str]]] = {}
        self.parents: Dict[int, Dict[int, Tuple[int, str]]] = {}
        self.interlayer: Dict[Tuple[str, str], int] = {(a,b):0 for a in LAYER_NAMES for b in LAYER_NAMES if a!=b}
        self.employee_shared: Set[int] = set()

        self.t = 0
        self.total_exposures = 0
        self.unique_reach: Set[int] = set()
        self.tick = self._blank_tick()
        self.brand_strength = 0.0
        self.participation = 0.0
        self.runtime_sec: Optional[float] = None
        self.sea_counts: Dict[str, int] = {"S":0,"E":0,"A":0,"D":0}
        self._outbox: Dict[int, List[Tuple[int,int,int,str]]] = {r: [] for r in range(self.size)}

        # printing & timing
        self.print_all_ranks = print_all_ranks
        self.rank_tag = rank_tag
        self.timing = timing
        self.timing_every = max(1, int(timing_every))
        self.last_step_time = 0.0
        self.avg_step_time = 0.0
        self._timing_sum = 0.0
        self._timing_count = 0

        # agents
        self.agents_by_id: Dict[int, BaseNode] = {}
        for i in self.local_agent_ids:
            if i in self.employee_ids: a = EmployeeAgent(i, self)
            elif i in self.customer_ids: a = AudienceAgent(i, self, role="customer")
            elif i in self.peer_ids: a = AudienceAgent(i, self, role="peer")
            else: a = AudienceAgent(i, self, role="public")
            self.agents_by_id[i] = a

        # inject ages
        self.ages = ages
        if self.ages is not None:
            for i in self.local_agent_ids:
                ag = self.agents_by_id[i]
                ag.age = int(self.ages[i])
                ag.age_group = self._age_group_of(ag.age)

        # logging
        if history_csv is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            history_csv = os.path.join("logs", f"sead_timeseries_{ts}.csv")
        self.history_csv = history_csv
        self._logger = RunLogger(self.history_csv)
        self._logger_opened = False

    # ---- helpers ----
    def _blank_tick(self) -> Dict[str, int]:
        d = {"content_created":0, "exposures":0, "unique_reach_added":0,
             "exp_employees":0, "exp_customers":0, "exp_peers":0, "exp_public":0,
             "engagements":0, "s_to_e":0, "e_to_a":0, "e_to_d":0, "a_to_d":0, "d_to_s":0,
             "shares":0}
        for ln in LAYER_NAMES:
            d[f"exp_{ln}"] = 0
            d[f"shares_{ln}"] = 0
            d[f"engagements_{ln}"] = 0
            d[f"s_to_e_{ln}"] = 0
            d[f"e_to_a_{ln}"] = 0
            d[f"e_to_d_{ln}"] = 0
            d[f"a_to_d_{ln}"] = 0
            d[f"d_to_s_{ln}"] = 0
        return d

    def owner_of(self, agent_id: int) -> int:
        return int(self.owners[agent_id])

    def neighbors_in(self, agent_id: int, layer: str) -> List[int]:
        if layer == "internal" and agent_id in self.G_internal:   return list(self.G_internal.neighbors(agent_id))
        if layer == "professional" and agent_id in self.G_prof:   return list(self.G_prof.neighbors(agent_id))
        if layer == "personal" and agent_id in self.G_personal:   return list(self.G_personal.neighbors(agent_id))
        return []

    def neighbors_union(self, agent_id: int) -> List[int]:
        return union_neighbors(agent_id, self.G_internal, self.G_prof, self.G_personal)

    def tie_strength(self, src: BaseNode, dst: BaseNode) -> float:
        try:
            d = nx.shortest_path_length(self.G_personal, src.unique_id, dst.unique_id)
            return 1.0 / (1.0 + d)
        except Exception:
            return 0.1

    def credibility(self, src: BaseNode) -> float:
        return 0.8 if isinstance(src, EmployeeAgent) else 0.5

    @property
    def virtual_company_id(self) -> int:
        return -1

    # age helpers
    def _age_group_of(self, age: int) -> str:
        if age <= 24: return "18-24"
        if age <= 34: return "25-34"
        if age <= 44: return "35-44"
        if age <= 54: return "45-54"
        if age <= 64: return "55-64"
        return "65+"

    def age_similarity(self, a: int, b: int) -> float:
        diff = abs(a - b)
        cap = 30.0
        return max(0.0, 1.0 - (diff / cap))

    def age_share_prob(self, src_age: int, dst_age: int) -> float:
        return self.p_age_in if abs(src_age - dst_age) <= self.age_gap else self.p_age_out

    # content & delivery
    def _local_deliver(self, cid: int, src_id: int, dst_id: int, layer: str) -> None:
        dst = self.agents_by_id.get(dst_id)
        if dst is None: return
        if cid not in self.deliveries: self.deliveries[cid] = set()
        key = (src_id, dst_id, layer)
        if key in self.deliveries[cid]: return
        dst.inbox_by_layer[layer].append(cid)
        self.deliveries[cid].add(key)
        self._count_exposure(dst_id)
        self.tick[f"exp_{layer}"] += 1

    def deliver(self, cid: int, src_id: int, dst_id: int, layer: str) -> None:
        dst_rank = self.owner_of(dst_id)
        if dst_rank == self.rank: self._local_deliver(cid, src_id, dst_id, layer)
        else: self._outbox[dst_rank].append((cid, src_id, dst_id, layer))

    def spawn_company_content_distributed(self) -> None:
        if self.rank == 0:
            lam = max(0.01, self.mean_contents_per_tick * (1.0 + self.csma_level))
            k = int(np.random.poisson(lam))
            self.tick["content_created"] += k
            new_contents = []; seed_msgs = []
            for _ in range(k):
                cid = self.next_cid
                c = Content(cid=cid,
                            topic_vec=np.random.normal(0, 1, size=self.topic_dim),
                            quality=float(random.random()),
                            sensitivity=float(random.random() * 0.8),
                            geo_relevance=float(random.random()),
                            timestamp=self.t)
                self.contents[cid] = c
                self.deliveries[cid] = set()
                new_contents.append((cid, c.topic_vec.tolist(), c.quality, c.sensitivity, c.geo_relevance, c.timestamp))
                self.next_cid += 1
                seeds = random.sample(self.employee_ids, k=min(5, len(self.employee_ids))) if self.employee_ids else []
                for sid in seeds:
                    seed_msgs.append((cid, self.virtual_company_id, sid, "internal"))
        else:
            new_contents = None; seed_msgs = None

        new_contents = self.comm.bcast(new_contents, root=0)
        seed_msgs = self.comm.bcast(seed_msgs, root=0)

        for (cid, topic_vec, q, s, g, ts) in new_contents:
            if cid not in self.contents:
                self.contents[cid] = Content(cid=cid, topic_vec=np.array(topic_vec), quality=q,
                                             sensitivity=s, geo_relevance=g, timestamp=ts)
                self.deliveries[cid] = set()
            self.next_cid = max(self.next_cid, cid + 1)
        for (cid, src, dst, layer) in seed_msgs:
            self.deliver(cid, src, dst, layer)

    def visible_content_for(self, agent: BaseNode) -> List[Tuple[int, str]]:
        selected: List[Tuple[int, str]] = []
        for layer in LAYER_NAMES:
            inbox = agent.inbox_by_layer[layer]
            if not inbox: continue
            vis = self.layer_visibility.get(layer, self.base_visibility)
            kcap = self.layer_attn_k.get(layer, 8)
            scored = []
            for cid in inbox:
                c = self.contents.get(cid)
                if not c: continue
                score = min(1.0, vis + self.visibility_slope * c.social_proof)
                if self.geo_decay > 0 and random.random() < self.geo_decay: continue
                if random.random() < score: scored.append((score, cid))
            scored.sort(reverse=True)
            for _, cid in scored[:kcap]:
                selected.append((cid, layer))
        return selected

    def sample_share_layers(self, seen_layer: str) -> List[str]:
        probs = self.layer_coupling.get(seen_layer, {})
        chosen: List[str] = []
        for to_layer, p in probs.items():
            p = max(0.0, min(1.0, p))
            if random.random() < p: chosen.append(to_layer)
        if seen_layer not in chosen: chosen.append(seen_layer)
        return [l for l in LAYER_NAMES if l in set(chosen)]

    def _recent_share_to(self, neighbor_id: int, employee_id: int, within: int = 3, layer: Optional[str] = None) -> bool:
        for cid, triples in self.deliveries.items():
            if layer is None:
                if any((src == neighbor_id and dst == employee_id) for (src, dst, _l) in triples):
                    if self.t - self.contents[cid].timestamp <= within: return True
            else:
                if any((src == neighbor_id and dst == employee_id and _l == layer) for (src, dst, _l) in triples):
                    if self.t - self.contents[cid].timestamp <= within: return True
        return False

    def get_peer_pressure(self, employee: BaseNode, layer: Optional[str] = None) -> float:
        alpha = self.peer_pressure_alpha

        neigh_union = self.neighbors_union(employee.unique_id)
        mult = 0.0
        if len(neigh_union) > 0:
            def _w(nid):
                ag_n = self.agents_by_id.get(nid)
                if ag_n is None or employee.age is None or ag_n.age is None: return 1.0
                return 0.5 + 0.5 * self.age_similarity(employee.age, ag_n.age)
            wsum = 0.0; recw = 0.0
            for n in neigh_union:
                w = _w(n); wsum += w
                if self._recent_share_to(n, employee.unique_id, within=3, layer=None): recw += w
            mult = recw / max(1e-9, wsum)

        if layer is None: return mult

        neigh_layer = self.neighbors_in(employee.unique_id, layer)
        lay = 0.0
        if len(neigh_layer) > 0:
            def _w(nid):
                ag_n = self.agents_by_id.get(nid)
                if ag_n is None or employee.age is None or ag_n.age is None: return 1.0
                return 0.5 + 0.5 * self.age_similarity(employee.age, ag_n.age)
            wsum = 0.0; recw = 0.0
            for n in neigh_layer:
                w = _w(n); wsum += w
                if self._recent_share_to(n, employee.unique_id, within=3, layer=layer): recw += w
            lay = recw / max(1e-9, wsum)

        return alpha * mult + (1.0 - alpha) * lay

    def share_content(self, src: BaseNode, c: Content, layer: str) -> None:
        c.shares += 1
        self.tick["shares"] += 1
        self.tick[f"shares_{layer}"] += 1
        for nid in self.neighbors_in(src.unique_id, layer):
            if nid == src.unique_id: continue
            dst_agent = self.agents_by_id.get(nid)
            if dst_agent is not None and src.age is not None and dst_agent.age is not None:
                if random.random() >= self.age_share_prob(src.age, dst_agent.age): continue
            self.deliver(c.cid, src.unique_id, nid, layer)

    def reshare_content(self, src: BaseNode, c: Content, layer: str) -> None:
        self.share_content(src, c, layer)

    def any_source_of_on(self, cid: int, dst_id: int) -> Optional[Tuple[int, str]]:
        cand = [(src, layer) for (src, dst, layer) in self.deliveries.get(cid, set()) if dst == dst_id]
        return cand[-1] if cand else None

    def last_source_of(self, cid: int, dst_id: int) -> Optional[Tuple[int, str]]:
        return self.any_source_of_on(cid, dst_id)

    def _count_exposure(self, dst_id: int) -> None:
        self.total_exposures += 1
        self.tick["exposures"] += 1
        if dst_id in self.employee_ids: self.tick["exp_employees"] += 1
        elif dst_id in self.customer_ids: self.tick["exp_customers"] += 1
        elif dst_id in self.peer_ids: self.tick["exp_peers"] += 1
        elif dst_id in self.public_ids: self.tick["exp_public"] += 1
        before = dst_id in self.unique_reach
        self.unique_reach.add(dst_id)
        if not before: self.tick["unique_reach_added"] += 1

    def _compute_sea_counts_local(self) -> None:
        s = e = a = d = 0
        for aid, ag in self.agents_by_id.items():
            if not hasattr(ag, "state"): continue
            if ag.state == SEA.S: s += 1
            elif ag.state == SEA.E: e += 1
            elif ag.state == SEA.A: a += 1
            elif ag.state == SEA.D: d += 1
        self.sea_counts = {"S": s, "E": e, "A": a, "D": d}

    # ---- logging ----
    def _log_tick_row(self):
        row = {
            "t": self.t - 1,
            "S": self.sea_counts["S"], "E": self.sea_counts["E"], "A": self.sea_counts["A"], "D": self.sea_counts["D"],
            "brand_strength": self.brand_strength,
            "participation": self.participation,
            "unique_reach": len(self.unique_reach),
            "exposures": self.tick["exposures"],
            "shares": self.tick["shares"],
            "content_created": self.tick["content_created"],
        }
        for ln in LAYER_NAMES:
            row[f"exp_{ln}"] = self.tick[f"exp_{ln}"]
            row[f"shares_{ln}"] = self.tick[f"shares_{ln}"]
            row[f"engagements_{ln}"] = self.tick[f"engagements_{ln}"]
            row[f"s_to_e_{ln}"] = self.tick[f"s_to_e_{ln}"]
            row[f"e_to_a_{ln}"] = self.tick[f"e_to_a_{ln}"]
            row[f"e_to_d_{ln}"] = self.tick[f"e_to_d_{ln}"]
            row[f"a_to_d_{ln}"] = self.tick[f"a_to_d_{ln}"]
            row[f"d_to_s_{ln}"] = self.tick[f"d_to_s_{ln}"]
        if not self._logger_opened:
            self._logger.open(fieldnames=list(row.keys()))
            self._logger_opened = True
        self._logger.write(row)

    # ---- step ----
    def step(self, progress: bool = False) -> None:
        t0 = time.perf_counter() if self.timing else None
        self.tick = self._blank_tick()

        self.spawn_company_content_distributed()
        for c in self.contents.values(): c.decay(self.social_proof_decay)
        for r in range(self.size): self._outbox[r].clear()

        order = self.local_agent_ids[:]; random.shuffle(order)
        for aid in order: self.agents_by_id[aid].step()

        send = [self._outbox[r] for r in range(self.size)]
        recv = self.comm.alltoall(send)
        for msgs in recv:
            for (cid, src, dst, layer) in msgs:
                if self.owner_of(dst) == self.rank: self._local_deliver(cid, src, dst, layer)

        # participation
        local_emp_ids = [e for e in self.local_agent_ids if e in self.employee_ids]
        local_part = len([e for e in local_emp_ids if e in self.employee_shared]) / max(1, len(local_emp_ids))
        part_arr = np.array([local_part], dtype=np.float64); part_out = np.zeros_like(part_arr)
        self.comm.Allreduce(part_arr, part_out, op=MPI.SUM); self.participation = float(part_out[0] / self.size)

        # SEA global
        self._compute_sea_counts_local()
        sea_loc = np.array([self.sea_counts["S"], self.sea_counts["E"], self.sea_counts["A"], self.sea_counts["D"]], dtype=np.int64)
        sea_glob = np.zeros_like(sea_loc); self.comm.Allreduce(sea_loc, sea_glob, op=MPI.SUM)
        self.sea_counts = {"S": int(sea_glob[0]), "E": int(sea_glob[1]), "A": int(sea_glob[2]), "D": int(sea_glob[3])}

        # exposures / shares
        kpi_loc = np.array([
            self.tick["exposures"], self.tick["shares"],
            self.tick["exp_internal"], self.tick["exp_professional"], self.tick["exp_personal"]
        ], dtype=np.int64)
        kpi_glob = np.zeros_like(kpi_loc); self.comm.Allreduce(kpi_loc, kpi_glob, op=MPI.SUM)
        self.tick["exposures"], self.tick["shares"], self.tick["exp_internal"], self.tick["exp_professional"], self.tick["exp_personal"] = map(int, kpi_glob)

        # per-layer engage/transition reductions
        def _pack_layers(key_base):
            return np.array([self.tick[f"{key_base}_internal"],
                             self.tick[f"{key_base}_professional"],
                             self.tick[f"{key_base}_personal"]], dtype=np.int64)
        for base in ["engagements", "s_to_e", "e_to_a", "e_to_d", "a_to_d", "d_to_s"]:
            loc = _pack_layers(base); glob = np.zeros_like(loc)
            self.comm.Allreduce(loc, glob, op=MPI.SUM)
            self.tick[f"{base}_internal"], self.tick[f"{base}_professional"], self.tick[f"{base}_personal"] = map(int, glob)

        # brand (global engagements)
        eng_loc = np.array([sum(c.engagements for c in self.contents.values())], dtype=np.float64)
        eng_glob = np.zeros_like(eng_loc); self.comm.Allreduce(eng_loc, eng_glob, op=MPI.SUM)
        pos = float(eng_glob[0]); neg = 0.0
        self.brand_strength = (self.brand_lambda * self.brand_strength + self.brand_eta_pos * pos - self.brand_eta_neg * neg)

        # timing
        if self.timing:
            dt_loc = time.perf_counter() - t0
            arr = np.array([dt_loc], dtype=np.float64); out = np.zeros_like(arr)
            self.comm.Allreduce(arr, out, op=MPI.MAX)
            self.last_step_time = float(out[0]); self._timing_sum += self.last_step_time
            self._timing_count += 1; self.avg_step_time = self._timing_sum / max(1, self._timing_count)

        self.t += 1

        # log row
        if self.rank == 0: self._log_tick_row()

        # print
        if progress and (self.print_all_ranks or self.rank == 0):
            if (not self.timing) or (self.t % self.timing_every == 0):
                s,e,a,d = self.sea_counts["S"], self.sea_counts["E"], self.sea_counts["A"], self.sea_counts["D"]
                prefix = f"[r{self.rank}] " if self.rank_tag else ""
                if self.timing:
                    print("{}[t={:02d}] S={} E={} A={} D={} | shares={} exp={} | step_time={:.4f}s avg_step={:.4f}s".format(
                        prefix, self.t - 1, s, e, a, d, self.tick["shares"], self.tick["exposures"],
                        self.last_step_time, self.avg_step_time
                    ))
                else:
                    print("{}[t={:02d}] S={} E={} A={} D={} | shares={} exp={}".format(
                        prefix, self.t - 1, s, e, a, d, self.tick["shares"], self.tick["exposures"]
                    ))

    def run(self, steps: Optional[int] = None, progress: bool = True) -> float:
        total_steps = steps if steps is not None else self.steps
        t0 = time.perf_counter()
        for _ in range(total_steps):
            self.step(progress=progress)
        self.runtime_sec = time.perf_counter() - t0
        if hasattr(self, "_logger"):
            try: self._logger.close()
            except Exception: pass
        return self.runtime_sec

# ---------------- CONFIG OVERRIDES HELPERS ----------------
def load_overrides(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config JSON root must be an object/dict.")
    return data

def _update_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _update_dict(out[k], v)
        else:
            out[k] = v
    return out

def apply_global_overrides(ov: Dict[str, Any]) -> Dict[str, Any]:
    """
    Εφαρμόζει overrides σε παγκόσμιες σταθερές (module-level).
    Επιστρέφει ένα λεξικό με τα overrides που εφαρμόστηκαν (για logging).
    """
    applied = {}

    global DEFAULT_KAPPA, LAYER_VIS, LAYER_ATTN_K, VISIBILITY_SLOPE
    global AGE_GAP, P_AGE_IN, P_AGE_OUT, A7_AGE_SIM
    global INCENTIVE_LEVEL, CSMA_LEVEL, PEER_PRESSURE_ALPHA

    if "DEFAULT_KAPPA" in ov:
        DEFAULT_KAPPA = _update_dict(DEFAULT_KAPPA, ov["DEFAULT_KAPPA"])
        applied["DEFAULT_KAPPA"] = DEFAULT_KAPPA
    if "LAYER_VIS" in ov:
        LAYER_VIS = _update_dict(LAYER_VIS, ov["LAYER_VIS"])
        applied["LAYER_VIS"] = LAYER_VIS
    if "LAYER_ATTN_K" in ov:
        LAYER_ATTN_K = _update_dict(LAYER_ATTN_K, ov["LAYER_ATTN_K"])
        applied["LAYER_ATTN_K"] = LAYER_ATTN_K
    if "VISIBILITY_SLOPE" in ov:
        VISIBILITY_SLOPE = float(ov["VISIBILITY_SLOPE"])
        applied["VISIBILITY_SLOPE"] = VISIBILITY_SLOPE

    if "AGE_GAP" in ov:
        AGE_GAP = int(ov["AGE_GAP"]); applied["AGE_GAP"] = AGE_GAP
    if "P_AGE_IN" in ov:
        P_AGE_IN = float(ov["P_AGE_IN"]); applied["P_AGE_IN"] = P_AGE_IN
    if "P_AGE_OUT" in ov:
        P_AGE_OUT = float(ov["P_AGE_OUT"]); applied["P_AGE_OUT"] = P_AGE_OUT
    if "A7_AGE_SIM" in ov:
        A7_AGE_SIM = float(ov["A7_AGE_SIM"]); applied["A7_AGE_SIM"] = A7_AGE_SIM

    if "INCENTIVE_LEVEL" in ov:
        INCENTIVE_LEVEL = float(ov["INCENTIVE_LEVEL"]); applied["INCENTIVE_LEVEL"] = INCENTIVE_LEVEL
    if "CSMA_LEVEL" in ov:
        CSMA_LEVEL = float(ov["CSMA_LEVEL"]); applied["CSMA_LEVEL"] = CSMA_LEVEL
    if "PEER_PRESSURE_ALPHA" in ov:
        PEER_PRESSURE_ALPHA = float(ov["PEER_PRESSURE_ALPHA"]); applied["PEER_PRESSURE_ALPHA"] = PEER_PRESSURE_ALPHA

    return applied

def apply_model_overrides(model: "DistributedModel", ov: Dict[str, Any]) -> Dict[str, Any]:
    """
    Εφαρμόζει overrides σε παραμέτρους του στιγμιοτύπου (attributes του model).
    Επιστρέφει τι εφαρμόστηκε.
    Επιτρέπει κλειδιά:
    - mean_contents_per_tick, social_proof_decay, dormancy_rate
    - compliance_threshold, employee_cooldown
    - visibility_slope, layer_visibility, layer_attn_k, geo_decay, base_visibility
    - layer_coupling
    - peer_pressure_alpha
    - βάρη w1..w8, a1..a6, b1..b4
    """
    applied = {}

    # Scalars
    scalar_keys = [
        "mean_contents_per_tick","social_proof_decay","dormancy_rate",
        "compliance_threshold","employee_cooldown",
        "visibility_slope","geo_decay","base_visibility",
        "peer_pressure_alpha",
        "brand_lambda","brand_eta_pos","brand_eta_neg"
    ]
    for k in scalar_keys:
        if k in ov:
            setattr(model, k, float(ov[k]) if k not in ("employee_cooldown",) else int(ov[k]))
            applied[k] = getattr(model, k)

    # Weight groups
    for k in ["w1","w2","w3","w4","w5","w6","w7","w8",
              "a1","a2","a3","a4","a5","a6",
              "b1","b2","b3","b4"]:
        if k in ov:
            setattr(model, k, float(ov[k])); applied[k] = getattr(model, k)

    # Dicts
    if "layer_visibility" in ov or "LAYER_VIS" in ov:
        val = ov.get("layer_visibility", ov.get("LAYER_VIS"))
        model.layer_visibility = _update_dict(model.layer_visibility, val)
        applied["layer_visibility"] = model.layer_visibility
    if "layer_attn_k" in ov or "LAYER_ATTN_K" in ov:
        val = ov.get("layer_attn_k", ov.get("LAYER_ATTN_K"))
        # ensure ints
        merged = _update_dict(model.layer_attn_k, val)
        model.layer_attn_k = {k:int(v) for k,v in merged.items()}
        applied["layer_attn_k"] = model.layer_attn_k
    if "layer_coupling" in ov or "DEFAULT_KAPPA" in ov:
        val = ov.get("layer_coupling", ov.get("DEFAULT_KAPPA"))
        model.layer_coupling = _update_dict(model.layer_coupling, val)
        applied["layer_coupling"] = model.layer_coupling

    # Age homophily at instance level (optional override)
    for k_model, k_global in [("age_gap","AGE_GAP"),("p_age_in","P_AGE_IN"),("p_age_out","P_AGE_OUT"),("a7_age_sim","A7_AGE_SIM")]:
        if k_model in ov:
            setattr(model, k_model, float(ov[k_model]) if k_model!="age_gap" else int(ov[k_model]))
            applied[k_model] = getattr(model, k_model)
        elif k_global in ov:
            # if only global key provided, also mirror to instance to be explicit
            val = ov[k_global]
            setattr(model, k_model, float(val) if k_model!="age_gap" else int(val))
            applied[k_model] = getattr(model, k_model)

    # Incentives & csma mirrors (if given by global keys)
    if "INCENTIVE_LEVEL" in ov:
        model.incentive_level = float(ov["INCENTIVE_LEVEL"]); applied["incentive_level"] = model.incentive_level
    if "CSMA_LEVEL" in ov:
        model.csma_level = float(ov["CSMA_LEVEL"]); applied["csma_level"] = model.csma_level

    return applied

# ---------------- MAIN ----------------
def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logging.basicConfig(level=logging.INFO if rank==0 else logging.WARNING)
    log = logging.getLogger("SEAD")

    # Aggressive overrides
    global DEFAULT_KAPPA, LAYER_VIS, LAYER_ATTN_K, VISIBILITY_SLOPE
    global AGE_GAP, P_AGE_OUT, A7_AGE_SIM, P_AGE_IN
    global INCENTIVE_LEVEL, CSMA_LEVEL, PEER_PRESSURE_ALPHA

    employee_cooldown_override = None
    compliance_threshold_override = None
    mean_contents_per_tick_override = None

    if args.aggressive:
        DEFAULT_KAPPA = AGGR_PRESET["DEFAULT_KAPPA"]
        LAYER_VIS = AGGR_PRESET["LAYER_VIS"]
        LAYER_ATTN_K = AGGR_PRESET["LAYER_ATTN_K"]
        VISIBILITY_SLOPE = AGGR_PRESET["VISIBILITY_SLOPE"]
        AGE_GAP = AGGR_PRESET["AGE_GAP"]
        P_AGE_OUT = AGGR_PRESET["P_AGE_OUT"]
        A7_AGE_SIM = AGGR_PRESET["A7_AGE_SIM"]
        # P_AGE_IN παραμένει 1.0 (baseline), εκτός αν δοθεί στο config.
        INCENTIVE_LEVEL = AGGR_PRESET["INCENTIVE_LEVEL"]
        CSMA_LEVEL = AGGR_PRESET["CSMA_LEVEL"]
        PEER_PRESSURE_ALPHA = AGGR_PRESET["PEER_PRESSURE_ALPHA"]
        employee_cooldown_override = AGGR_PRESET["EMPLOYEE_COOLDOWN"]
        compliance_threshold_override = AGGR_PRESET["COMPLIANCE_THRESHOLD"]
        mean_contents_per_tick_override = AGGR_PRESET["MEAN_CONTENTS_PER_TICK"]

    # ----- Load JSON overrides (applied after aggressive) -----
    json_overrides = {}
    try:
        json_overrides = load_overrides(args.config)
    except Exception as e:
        if rank == 0:
            log.error(f"Failed to load config overrides: {e}")
        # Proceed without JSON overrides

    # Apply global-level overrides first (affects constructor defaults)
    if json_overrides:
        applied = apply_global_overrides(json_overrides)
        if rank == 0 and applied:
            log.info("Applied GLOBAL overrides from JSON: %s", list(applied.keys()))

    if rank == 0:
        print("Edge-list mode: loading nodes & multilayer edges")

    G_internal, G_prof, G_personal, roles, ages = build_graphs_from_edge_lists(
        nodes_path=args.nodes,
        edges_internal_path=args.edges_internal,
        edges_prof_path=args.edges_professional,
        edges_school_path=args.edges_school,
        edges_personal_path=args.edges_personal,
        edges_family_path=args.edges_family,
        seed=args.seed
    )

    if rank == 0:
        print(f"[internal] nodes={G_internal.number_of_nodes()} edges={G_internal.number_of_edges()}")
        print(f"[professional] nodes={G_prof.number_of_nodes()} edges={G_prof.number_of_edges()}")
        print(f"[personal] nodes={G_personal.number_of_nodes()} edges={G_personal.number_of_edges()}")
        print(f"[roles] employees={len(roles['employee_ids'])} peers={len(roles['peer_ids'])} customers={len(roles['customer_ids'])} public={len(roles['public_ids'])}")

    # Output path
    out_csv = args.out
    if out_csv is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join("logs", f"sead_timeseries_{ts}.csv")
    if rank == 0:
        print("Per-layer CSV will be saved to:", out_csv)

    # Model
    model = DistributedModel(
        G_internal=G_internal, G_prof=G_prof, G_personal=G_personal,
        employee_ids=roles["employee_ids"].tolist(),
        peer_ids=roles["peer_ids"].tolist(),
        customer_ids=roles["customer_ids"].tolist(),
        public_ids=roles["public_ids"].tolist(),
        ages=ages,
        steps=args.steps, incentive_level=INCENTIVE_LEVEL, csma_level=CSMA_LEVEL,
        layer_visibility=LAYER_VIS, layer_attn_k=LAYER_ATTN_K,
        layer_coupling=DEFAULT_KAPPA,
        peer_pressure_alpha=PEER_PRESSURE_ALPHA,
        seed=args.seed, comm=MPI.COMM_WORLD,
        print_all_ranks=False, rank_tag=True,
        timing=True, timing_every=1,
        history_csv=out_csv,
    )

    # Instance-level overrides from aggressive preset
    if employee_cooldown_override is not None:
        model.employee_cooldown = employee_cooldown_override
    if compliance_threshold_override is not None:
        model.compliance_threshold = compliance_threshold_override
    if mean_contents_per_tick_override is not None:
        model.mean_contents_per_tick = mean_contents_per_tick_override
    model.visibility_slope = VISIBILITY_SLOPE

    # Instance-level overrides from JSON (highest priority)
    if json_overrides:
        applied_inst = apply_model_overrides(model, json_overrides)
        if rank == 0 and applied_inst:
            log.info("Applied MODEL overrides from JSON: %s", list(applied_inst.keys()))

    elapsed = model.run(progress=True)
    if rank == 0:
        print("\nExperiment runtime: {:.3f} sec ({:.2f} min) - avg_step={:.4f}s".format(elapsed, elapsed/60.0, model.avg_step_time))
        print("Saved per-layer time series to:", model.history_csv)

if __name__ == "__main__":
    main()
