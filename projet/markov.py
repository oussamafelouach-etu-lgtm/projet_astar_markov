

import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Constantes pour états spéciaux
# ---------------------------------------------------------------------------
GOAL_STATE = "GOAL"
FAIL_STATE  = "FAIL"


# ---------------------------------------------------------------------------
# Construction de la matrice de transition P
# ---------------------------------------------------------------------------

def build_transition_matrix(
    policy: Dict[Tuple[int, int], Tuple[int, int]],
    grid: List[List[int]],
    goal: Tuple[int, int],
    epsilon: float = 0.1,
    include_fail: bool = False,
    fail_cells: Optional[set] = None,
) -> Tuple[np.ndarray, List]:

    rows, cols = len(grid), len(grid[0])

    # Collecter tous les états non-GOAL de la politique
    states_cells = set(policy.keys())
    states_cells.add(goal)  # goal sera remplacé par GOAL
    if include_fail and fail_cells:
        states_cells |= set(fail_cells)

    # Construction de l'index
    cell_states = sorted([s for s in states_cells if s != goal], key=lambda x: (x[0], x[1]))
    idx = cell_states + [GOAL_STATE]
    if include_fail:
        idx.append(FAIL_STATE)

    state_to_i = {s: i for i, s in enumerate(idx)}
    n = len(idx)
    P = np.zeros((n, n))

    # GOAL absorbant
    goal_i = state_to_i[GOAL_STATE]
    P[goal_i, goal_i] = 1.0

    # FAIL absorbant
    if include_fail:
        fail_i = state_to_i[FAIL_STATE]
        P[fail_i, fail_i] = 1.0

    def cell_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 1

    def get_lateral(dr, dc):
        """Retourne les 2 directions latérales (perpendiculaires)."""
        return [(-dc, dr), (dc, -dr)]

    for cell in cell_states:
        if cell == goal:
            continue
        if cell not in policy:
            # Pas d'action → reste sur place
            i = state_to_i[cell]
            P[i, i] = 1.0
            continue

        i = state_to_i[cell]
        next_cell = policy[cell]
        dr = next_cell[0] - cell[0]
        dc = next_cell[1] - cell[1]

        laterals = get_lateral(dr, dc)

        # Action principale
        prob_main = 1.0 - epsilon
        prob_lat  = epsilon / 2.0

        def add_transition(from_i, target_cell, prob):
            if target_cell == goal:
                j = state_to_i[GOAL_STATE]
            elif include_fail and fail_cells and target_cell in fail_cells:
                j = state_to_i[FAIL_STATE]
            elif target_cell in state_to_i:
                j = state_to_i[target_cell]
            else:
                # Hors-grille ou obstacle : reste sur place
                j = from_i
            P[from_i, j] += prob

        # Transition principale
        tr, tc = next_cell
        if cell_valid(tr, tc):
            add_transition(i, next_cell, prob_main)
        else:
            P[i, i] += prob_main  # reste sur place

        # Transitions latérales
        for ldr, ldc in laterals:
            lr, lc = cell[0] + ldr, cell[1] + ldc
            lat_cell = (lr, lc)
            if cell_valid(lr, lc):
                add_transition(i, lat_cell, prob_lat)
            else:
                P[i, i] += prob_lat  # reste sur place

    # Vérification stochasticité
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), f"P non stochastique : {row_sums}"

    return P, idx


# ---------------------------------------------------------------------------
# Calcul de π^(n) = π^(0) · P^n
# ---------------------------------------------------------------------------

def distribution_at_step(pi0: np.ndarray, P: np.ndarray, n: int) -> np.ndarray:

    Pn = np.linalg.matrix_power(P, n)
    return pi0 @ Pn


def distribution_trajectory(pi0: np.ndarray, P: np.ndarray, n_steps: int) -> np.ndarray:

    n_states = P.shape[0]
    traj = np.zeros((n_steps + 1, n_states))
    traj[0] = pi0
    pi = pi0.copy()
    for t in range(1, n_steps + 1):
        pi = pi @ P
        traj[t] = pi
    return traj


# ---------------------------------------------------------------------------
# Analyse des classes de communication
# ---------------------------------------------------------------------------

def communication_classes(P: np.ndarray, idx: List) -> Dict:

    n = P.shape[0]
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-12:
                adj[i].add(j)

    # Kosaraju SCC
    visited = [False] * n
    finish_order = []

    def dfs1(v):
        stack = [(v, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                finish_order.append(node)
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            for nb in adj[node]:
                if not visited[nb]:
                    stack.append((nb, False))

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    # Graphe transposé
    adj_t = {i: set() for i in range(n)}
    for i in range(n):
        for j in adj[i]:
            adj_t[j].add(i)

    visited2 = [False] * n
    scc_id = [-1] * n
    scc_count = 0

    def dfs2(v, comp_id):
        stack = [v]
        while stack:
            node = stack.pop()
            if visited2[node]:
                continue
            visited2[node] = True
            scc_id[node] = comp_id
            for nb in adj_t[node]:
                if not visited2[nb]:
                    stack.append(nb)

    for v in reversed(finish_order):
        if not visited2[v]:
            dfs2(v, scc_count)
            scc_count += 1

    # Regrouper par SCC
    classes = {}
    for i in range(n):
        cid = scc_id[i]
        classes.setdefault(cid, set()).add(i)

    # Identifier recurrent vs transient
    # Une SCC est récurrente si elle n'a pas d'arc sortant vers une autre SCC
    scc_out = {cid: set() for cid in classes}
    for i in range(n):
        for j in adj[i]:
            if scc_id[i] != scc_id[j]:
                scc_out[scc_id[i]].add(scc_id[j])

    state_type = {}
    for cid, members in classes.items():
        t = "recurrent" if len(scc_out[cid]) == 0 else "transient"
        for m in members:
            state_type[idx[m]] = t

    classes_named = [frozenset(idx[m] for m in members) for members in classes.values()]

    return {
        "classes": classes_named,
        "type": state_type,
        "n_scc": scc_count,
    }


# ---------------------------------------------------------------------------
# Matrices d'absorption (Q, R, N)
# ---------------------------------------------------------------------------

def absorption_analysis(P: np.ndarray, idx: List) -> Optional[Dict]:

    absorbing = [i for i, s in enumerate(idx) if s in (GOAL_STATE, FAIL_STATE)]
    transient  = [i for i in range(len(idx)) if i not in absorbing]

    if not absorbing or not transient:
        return None

    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]

    try:
        N = np.linalg.inv(np.eye(len(transient)) - Q)
    except np.linalg.LinAlgError:
        return None

    t_mean = N @ np.ones(len(transient))   # temps moyen d'absorption
    B = N @ R                               # proba d'être absorbé par chaque état

    return {
        "N": N,
        "t_mean": {idx[transient[i]]: t_mean[i] for i in range(len(transient))},
        "B": {
            idx[transient[i]]: {idx[absorbing[k]]: B[i, k] for k in range(len(absorbing))}
            for i in range(len(transient))
        },
        "transient_states": [idx[i] for i in transient],
        "absorbing_states": [idx[i] for i in absorbing],
    }


# ---------------------------------------------------------------------------
# Simulation Monte-Carlo
# ---------------------------------------------------------------------------

def simulate_trajectories(
    P: np.ndarray,
    idx: List,
    start: Tuple[int, int],
    n_simulations: int = 1000,
    max_steps: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Dict:

    if rng is None:
        rng = np.random.default_rng(42)

    state_to_i = {s: i for i, s in enumerate(idx)}
    goal_i = state_to_i.get(GOAL_STATE)
    fail_i = state_to_i.get(FAIL_STATE)

    if start not in state_to_i:
        raise ValueError(f"État de départ {start} non trouvé dans l'index Markov.")
    start_i = state_to_i[start]

    n_states = P.shape[0]
    # Précalcul des CDF pour chaque état (sampling efficace)
    cdf = np.cumsum(P, axis=1)

    results_goal  = []
    results_fail  = []
    n_timeout = 0

    for _ in range(n_simulations):
        current = start_i
        absorbed = False
        for step in range(1, max_steps + 1):
            u = rng.random()
            current = int(np.searchsorted(cdf[current], u))
            if current == goal_i:
                results_goal.append(step)
                absorbed = True
                break
            if fail_i is not None and current == fail_i:
                results_fail.append(step)
                absorbed = True
                break
        if not absorbed:
            n_timeout += 1

    n_goal = len(results_goal)
    n_fail = len(results_fail)

    return {
        "n_simulations": n_simulations,
        "prob_goal": n_goal / n_simulations,
        "prob_fail": n_fail / n_simulations,
        "prob_timeout": n_timeout / n_simulations,
        "mean_time_goal": float(np.mean(results_goal)) if results_goal else None,
        "std_time_goal":  float(np.std(results_goal))  if results_goal else None,
        "times_goal": results_goal,
        "times_fail": results_fail,
    }
