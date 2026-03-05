"""
astar.py — Planification sur grille : UCS, Greedy Best-First, A*
"""

import heapq
import time
from typing import List, Tuple, Dict, Optional, Callable


# ---------------------------------------------------------------------------
# Représentation de la grille
# ---------------------------------------------------------------------------

def parse_grid(grid: List[List[int]]) -> Tuple[int, int]:
    """Retourne (rows, cols) de la grille."""
    return len(grid), len(grid[0])


def neighbors(state: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[Tuple[int, int], float]]:
    """
    Retourne les voisins (4-connexité) accessibles avec leur coût.
    grid[r][c] = 0 → libre, 1 → obstacle, ou coût > 1 si terrain variable.
    """
    r, c = state
    rows, cols = parse_grid(grid)
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            cost = 1 if grid[nr][nc] == 0 else float(grid[nr][nc])
            result.append(((nr, nc), cost))
    return result


# ---------------------------------------------------------------------------
# Heuristiques
# ---------------------------------------------------------------------------

def heuristic_manhattan(state: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def heuristic_zero(state, goal) -> float:
    """Heuristique nulle → UCS."""
    return 0.0


def heuristic_euclidean(state: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return ((state[0] - goal[0])**2 + (state[1] - goal[1])**2) ** 0.5


# ---------------------------------------------------------------------------
# Algorithme générique Best-First (UCS / Greedy / A*)
# ---------------------------------------------------------------------------

def best_first_search(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid: List[List[int]],
    heuristic: Callable = heuristic_manhattan,
    weight_g: float = 1.0,   # 0 → Greedy, 1 → A* / UCS
    weight_h: float = 1.0,   # 0 → UCS,    1 → A* / Greedy
) -> Dict:
    """
    Recherche Best-First générique.
    f(n) = weight_g * g(n) + weight_h * h(n)

    Returns dict :
        path        : liste de cellules (start..goal) ou [] si échec
        cost        : coût total du chemin
        nodes_exp   : nombre de nœuds développés (sorties de OPEN)
        open_max    : taille max de OPEN
        time_ms     : temps d'exécution en ms
    """
    t0 = time.perf_counter()

    # OPEN : (f, g, state)
    h0 = heuristic(start, goal)
    open_heap = [(weight_g * 0 + weight_h * h0, 0.0, start)]
    # g_cost[state] = meilleur g connu
    g_cost: Dict[Tuple[int, int], float] = {start: 0.0}
    # parent pour reconstruire le chemin
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    closed = set()
    nodes_exp = 0
    open_max = 1

    while open_heap:
        open_max = max(open_max, len(open_heap))
        f, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        nodes_exp += 1

        if current == goal:
            # Reconstruire le chemin
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "path": path,
                "cost": g_cost[goal],
                "nodes_exp": nodes_exp,
                "open_max": open_max,
                "time_ms": elapsed,
                "success": True,
            }

        for nb, move_cost in neighbors(current, grid):
            new_g = g + move_cost
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                parent[nb] = current
                h_val = heuristic(nb, goal)
                f_val = weight_g * new_g + weight_h * h_val
                heapq.heappush(open_heap, (f_val, new_g, nb))

    elapsed = (time.perf_counter() - t0) * 1000
    return {"path": [], "cost": float("inf"), "nodes_exp": nodes_exp,
            "open_max": open_max, "time_ms": elapsed, "success": False}


# ---------------------------------------------------------------------------
# Wrappers nommés
# ---------------------------------------------------------------------------

def ucs(start, goal, grid):
    return best_first_search(start, goal, grid, heuristic_zero, weight_g=1.0, weight_h=0.0)

def greedy(start, goal, grid):
    return best_first_search(start, goal, grid, heuristic_manhattan, weight_g=0.0, weight_h=1.0)

def astar(start, goal, grid):
    return best_first_search(start, goal, grid, heuristic_manhattan, weight_g=1.0, weight_h=1.0)

def astar_weighted(start, goal, grid, w=2.0):
    """A* pondéré : f = g + w*h (sous-optimal mais plus rapide)."""
    return best_first_search(start, goal, grid, heuristic_manhattan, weight_g=1.0, weight_h=w)


# ---------------------------------------------------------------------------
# Politique induite par un chemin
# ---------------------------------------------------------------------------

def path_to_policy(path: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    À partir d'un chemin, construit un dictionnaire
    policy[state] = next_state (action recommandée).
    Le dernier état (goal) n'a pas d'action.
    """
    policy = {}
    for i in range(len(path) - 1):
        policy[path[i]] = path[i + 1]
    return policy


# ---------------------------------------------------------------------------
# Utilitaire d'affichage
# ---------------------------------------------------------------------------

def print_grid(grid, path=None, start=None, goal=None):
    rows, cols = parse_grid(grid)
    path_set = set(path) if path else set()
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            cell = (r, c)
            if cell == start:
                row_str += "S "
            elif cell == goal:
                row_str += "G "
            elif grid[r][c] == 1:
                row_str += "# "
            elif cell in path_set:
                row_str += ". "
            else:
                row_str += "  "
        print(row_str)
