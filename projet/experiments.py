"""
experiments.py — Expériences, comparaisons et visualisations
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

from astar import astar, ucs, greedy, astar_weighted, path_to_policy, print_grid, heuristic_manhattan, heuristic_zero, heuristic_euclidean, best_first_search
from markov import (
    build_transition_matrix,
    distribution_trajectory,
    communication_classes,
    absorption_analysis,
    simulate_trajectories,
    GOAL_STATE, FAIL_STATE,
)

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Grilles de test
# ---------------------------------------------------------------------------

def make_grid_simple():
    """Grille 8x8 avec quelques obstacles."""
    g = [[0]*8 for _ in range(8)]
    for obs in [(1,2),(2,2),(3,2),(4,2),(3,4),(3,5),(3,6),(5,1),(5,2),(6,6),(7,6)]:
        g[obs[0]][obs[1]] = 1
    return g

def make_grid_maze():
    """Labyrinthe 10x10."""
    g = [
        [0,0,0,0,1,0,0,0,0,0],
        [1,1,0,1,1,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,0,1,0],
        [0,0,0,0,0,0,1,0,0,0],
        [1,1,1,1,0,1,1,1,1,0],
        [0,0,0,1,0,0,0,0,1,0],
        [0,1,0,1,1,1,0,1,1,0],
        [0,1,0,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,1,0,0],
    ]
    return g

def make_grid_weighted():
    """Grille 8x8 avec coûts variables (terrains)."""
    g = [
        [0,0,0,0,0,0,0,0],
        [0,1,1,0,0,3,3,0],
        [0,1,0,0,0,3,0,0],
        [0,1,0,3,3,3,0,0],
        [0,0,0,3,1,1,0,0],
        [0,0,0,0,1,0,0,0],
        [0,3,3,0,0,0,1,0],
        [0,0,0,0,0,0,0,0],
    ]
    return g


# ---------------------------------------------------------------------------
# Expérience 1 : Comparaison UCS / Greedy / A*
# ---------------------------------------------------------------------------

def experiment_algo_comparison(grid, start, goal, name="exp1"):
    print(f"\n{'='*55}")
    print(f"  Comparaison algorithmes — {name}")
    print(f"  Grille {len(grid)}x{len(grid[0])}, start={start}, goal={goal}")
    print(f"{'='*55}")

    algos = {
        "UCS"            : ucs(start, goal, grid),
        "Greedy"         : greedy(start, goal, grid),
        "A*"             : astar(start, goal, grid),
        "A* pondéré(w=2)": astar_weighted(start, goal, grid, w=2.0),
    }

    print(f"{'Algorithme':<20} {'Coût':>8} {'Nœuds exp.':>12} {'OPEN max':>10} {'Temps(ms)':>10}")
    print("-"*65)
    for algo_name, res in algos.items():
        if res["success"]:
            print(f"{algo_name:<20} {res['cost']:>8.2f} {res['nodes_exp']:>12} {res['open_max']:>10} {res['time_ms']:>10.3f}")
        else:
            print(f"{algo_name:<20} {'FAIL':>8}")

    # Visualisation du chemin A*
    astar_res = algos["A*"]
    if astar_res["success"]:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Comparaison algorithmes — {name}", fontsize=13, fontweight='bold')
        for ax, (algo_name, res) in zip(axes, algos.items()):
            _plot_grid(ax, grid, res.get("path", []), start, goal, algo_name, res)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{name}_algo_comparison.png", dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  → Figure sauvegardée : {name}_algo_comparison.png")

    return algos


# ---------------------------------------------------------------------------
# Expérience 2 : Impact de ε sur la robustesse (simulation Markov)
# ---------------------------------------------------------------------------

def experiment_epsilon_impact(grid, start, goal, epsilons=None, n_sim=2000, name="exp2"):
    """
    E.2 — Fixe A* et varie ε ∈ {0, 0.1, 0.2, 0.3}.
    Mesure : (i) coût prévu par A*, (ii) P(GOAL) réelle via Markov, (iii) E[T|GOAL].
    """
    if epsilons is None:
        epsilons = [0.0, 0.1, 0.2, 0.3]   # valeurs E.2 exactes

    print(f"\n{'='*60}")
    print(f"  E.2 Impact de ε — {name}")
    print(f"{'='*60}")

    astar_res = astar(start, goal, grid)
    if not astar_res["success"]:
        print("  A* échec — impossible de construire la politique.")
        return

    astar_cost = astar_res["cost"]
    policy = path_to_policy(astar_res["path"])

    print(f"  Coût A* (déterministe) : {astar_cost:.1f}  |  {len(astar_res['path'])} étapes")
    print(f"\n  {'ε':>5} | {'Coût A*':>8} | {'P(GOAL)':>8} | {'E[T|GOAL]':>10} | {'Timeout':>8}")
    print(f"  {'-'*55}")

    results = {}
    for eps in epsilons:
        P, idx = build_transition_matrix(policy, grid, goal, epsilon=eps, include_fail=False)
        sim = simulate_trajectories(P, idx, start, n_simulations=n_sim, max_steps=300)
        results[eps] = {**sim, "astar_cost": astar_cost}
        et = f"{sim['mean_time_goal']:.1f}" if sim['mean_time_goal'] else "N/A"
        print(f"  {eps:>5.2f} | {astar_cost:>8.1f} | {sim['prob_goal']:>8.3f} | "
              f"{et:>10} | {sim['prob_timeout']:>8.3f}")

    # --- Figure 3 sous-graphes : coût A* / P(GOAL) / E[T] ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"E.2 — A* fixé, variation de ε — {name}", fontsize=13, fontweight='bold')

    eps_list = list(results.keys())
    pg      = [results[e]["prob_goal"]        for e in eps_list]
    et_list = [results[e]["mean_time_goal"] or 0 for e in eps_list]
    costs   = [results[e]["astar_cost"]       for e in eps_list]

    # (i) Coût A* (constant, référence déterministe)
    axes[0].bar([str(e) for e in eps_list], costs, color="#78909C", alpha=0.85, edgecolor='white')
    axes[0].axhline(astar_cost, color="#F44336", lw=2, linestyle="--", label=f"Coût A* = {astar_cost:.1f}")
    axes[0].set_xlabel("ε", fontsize=11)
    axes[0].set_ylabel("Coût du chemin planifié", fontsize=11)
    axes[0].set_title("(i) Coût A* (déterministe)")
    axes[0].set_ylim(0, astar_cost * 1.5)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(costs):
        axes[0].text(i, v + 0.3, f"{v:.1f}", ha='center', fontsize=10, fontweight='bold')

    # (ii) P(GOAL) réelle via Markov
    axes[1].plot(eps_list, pg, "o-", color="#2196F3", lw=2.5, ms=9)
    axes[1].fill_between(eps_list, pg, alpha=0.15, color="#2196F3")
    axes[1].set_xlabel("ε (incertitude)", fontsize=11)
    axes[1].set_ylabel("P(atteindre GOAL)", fontsize=11)
    axes[1].set_title("(ii) P(GOAL) réelle — Markov")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(eps_list)
    axes[1].grid(True, alpha=0.3)
    for x, y in zip(eps_list, pg):
        axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

    # (iii) E[T|GOAL]
    axes[2].plot(eps_list, et_list, "s-", color="#E91E63", lw=2.5, ms=9)
    axes[2].fill_between(eps_list, et_list, alpha=0.15, color="#E91E63")
    axes[2].set_xlabel("ε (incertitude)", fontsize=11)
    axes[2].set_ylabel("E[T | succès] (pas)", fontsize=11)
    axes[2].set_title("(iii) Temps moyen d'atteinte")
    axes[2].set_xticks(eps_list)
    axes[2].grid(True, alpha=0.3)
    for x, y in zip(eps_list, et_list):
        axes[2].annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_epsilon_impact.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → Figure sauvegardée : {name}_epsilon_impact.png")
    return results


# ---------------------------------------------------------------------------
# Expérience 3 : Distribution Markov π^(n) et absorption
# ---------------------------------------------------------------------------

def experiment_markov_analysis(grid, start, goal, epsilon=0.1, n_steps=60, name="exp3"):
    print(f"\n{'='*55}")
    print(f"  Analyse Markov (ε={epsilon}) — {name}")
    print(f"{'='*55}")

    astar_res = astar(start, goal, grid)
    if not astar_res["success"]:
        print("  A* échec.")
        return

    policy = path_to_policy(astar_res["path"])
    P, idx = build_transition_matrix(policy, grid, goal, epsilon=epsilon, include_fail=False)

    n_states = P.shape[0]
    print(f"  Taille de P : {n_states}×{n_states}")
    print(f"  P stochastique : {np.allclose(P.sum(axis=1), 1.0)}")

    # Distribution initiale : départ certain depuis start
    state_to_i = {s: i for i, s in enumerate(idx)}
    pi0 = np.zeros(n_states)
    if start in state_to_i:
        pi0[state_to_i[start]] = 1.0
    else:
        print(f"  AVERTISSEMENT : {start} absent de l'index Markov.")
        return

    traj = distribution_trajectory(pi0, P, n_steps)
    goal_i = state_to_i.get(GOAL_STATE)
    prob_goal_t = traj[:, goal_i] if goal_i is not None else np.zeros(n_steps + 1)

    # Absorption
    absorb = absorption_analysis(P, idx)
    if absorb:
        t_mean_start = absorb["t_mean"].get(start)
        b_goal = absorb["B"].get(start, {}).get(GOAL_STATE, None)
        print(f"  Temps moyen théorique d'absorption depuis start : {t_mean_start:.2f}")
        print(f"  Probabilité théorique d'atteindre GOAL depuis start : {b_goal:.4f}")

    # Classes de communication
    cc = communication_classes(P, idx)
    print(f"  Nombre de classes de communication : {cc['n_scc']}")
    rec = sum(1 for t in cc["type"].values() if t == "recurrent")
    trans = sum(1 for t in cc["type"].values() if t == "transient")
    print(f"  États récurrents : {rec}, transitoires : {trans}")

    # Monte-Carlo
    sim = simulate_trajectories(P, idx, start, n_simulations=3000, max_steps=300)
    print(f"\n  Monte-Carlo (3000 sim) :")
    print(f"    P(GOAL)       = {sim['prob_goal']:.4f}")
    print(f"    E[T|GOAL]     = {sim['mean_time_goal']:.2f}")

    # Figure : π^(n) vers GOAL
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Analyse Markov — {name}  (ε={epsilon})", fontsize=12, fontweight='bold')

    t_axis = np.arange(n_steps + 1)
    axes[0].plot(t_axis, prob_goal_t, color="#4CAF50", lw=2.5, label="Matriciel $P^n$")

    # Courbe empirique depuis simulation
    if sim["times_goal"]:
        times_sorted = sorted(sim["times_goal"])
        empirical_cdf = np.arange(1, len(times_sorted) + 1) / sim["n_simulations"]
        axes[0].step(times_sorted, empirical_cdf, color="#FF5722", lw=1.5,
                     alpha=0.7, label="Monte-Carlo CDF")
    axes[0].set_xlabel("Pas de temps n", fontsize=11)
    axes[0].set_ylabel("P(être dans GOAL à l'instant n)", fontsize=11)
    axes[0].set_title("Convergence vers GOAL")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogramme des temps d'atteinte
    if sim["times_goal"]:
        axes[1].hist(sim["times_goal"], bins=30, color="#9C27B0", alpha=0.7, edgecolor='white')
        axes[1].axvline(sim["mean_time_goal"], color="red", lw=2, linestyle="--",
                        label=f"Moyenne = {sim['mean_time_goal']:.1f}")
        if absorb and t_mean_start:
            axes[1].axvline(t_mean_start, color="blue", lw=2, linestyle=":",
                            label=f"Théorique = {t_mean_start:.1f}")
        axes[1].set_xlabel("Temps d'atteinte (pas)", fontsize=11)
        axes[1].set_ylabel("Fréquence", fontsize=11)
        axes[1].set_title("Distribution du temps d'atteinte GOAL")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_markov_analysis.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → Figure sauvegardée : {name}_markov_analysis.png")

    return {"absorption": absorb, "sim": sim, "traj": traj}


# ---------------------------------------------------------------------------
# Utilitaire de visualisation de grille
# ---------------------------------------------------------------------------

def _plot_grid(ax, grid, path, start, goal, title, res):
    rows, cols = len(grid), len(grid[0])
    path_set = set(path)

    # Matrice de couleurs
    cmat = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if (r, c) == start:
                cmat[r, c] = 3
            elif (r, c) == goal:
                cmat[r, c] = 4
            elif grid[r][c] == 1:
                cmat[r, c] = 1
            elif (r, c) in path_set:
                cmat[r, c] = 2
            else:
                cmat[r, c] = 0

    cmap = ListedColormap(["#F5F5F5", "#424242", "#2196F3", "#4CAF50", "#F44336"])
    ax.imshow(cmat, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(f"{title}\ncoût={res.get('cost', '?'):.1f}  nœuds={res.get('nodes_exp', '?')}",
                 fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    patches = [
        mpatches.Patch(color="#F5F5F5", label="Libre"),
        mpatches.Patch(color="#424242", label="Obstacle"),
        mpatches.Patch(color="#2196F3", label="Chemin"),
        mpatches.Patch(color="#4CAF50", label="Start"),
        mpatches.Patch(color="#F44336", label="Goal"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=6, framealpha=0.8)


def visualize_grid_path(grid, path, start, goal, epsilon=0.1, title="Grille et chemin A*",
                         filename="grid_path.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    rows, cols = len(grid), len(grid[0])
    path_set = set(path)
    cmat = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if (r, c) == start:
                cmat[r, c] = 3
            elif (r, c) == goal:
                cmat[r, c] = 4
            elif grid[r][c] == 1:
                cmat[r, c] = 1
            elif (r, c) in path_set:
                cmat[r, c] = 2

    cmap = ListedColormap(["#F5F5F5", "#424242", "#2196F3", "#4CAF50", "#F44336"])
    ax.imshow(cmat, cmap=cmap, vmin=0, vmax=4)

    # Numéroter les étapes
    for i, (r, c) in enumerate(path):
        if (r, c) not in (start, goal):
            ax.text(c, r, str(i), ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax.set_title(f"{title}  (ε={epsilon})", fontsize=11, fontweight='bold')
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True, color='gray', lw=0.5)

    patches = [
        mpatches.Patch(color="#4CAF50", label="Start"),
        mpatches.Patch(color="#F44336", label="Goal"),
        mpatches.Patch(color="#2196F3", label="Chemin A*"),
        mpatches.Patch(color="#424242", label="Obstacle"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → Figure sauvegardée : {filename}")


# ---------------------------------------------------------------------------
# Expérience E.3 : Comparaison de deux heuristiques admissibles
# ---------------------------------------------------------------------------

def experiment_heuristic_comparison(grid, start, goal, name="exp_e3"):
    """
    Compare h=0 (UCS / heuristique nulle) vs h=Manhattan vs h=Euclidean
    sur la même grille, en mesurant le nombre de nœuds développés,
    la taille max de OPEN et le coût du chemin trouvé.
    """
    print(f"\n{'='*55}")
    print(f"  E.3 Comparaison heuristiques — {name}")
    print(f"  Grille {len(grid)}x{len(grid[0])}, start={start}, goal={goal}")
    print(f"{'='*55}")

    configs = {
        "h = 0  (UCS)"       : (heuristic_zero,      1.0, 0.0),
        "h = Manhattan (A*)" : (heuristic_manhattan,  1.0, 1.0),
        "h = Euclidean (A*)" : (heuristic_euclidean,  1.0, 1.0),
    }

    results = {}
    print(f"\n  {'Heuristique':<22} {'Admissible':>11} {'Coût':>8} {'Nœuds dév.':>12} {'OPEN max':>10} {'Temps(ms)':>10}")
    print(f"  {'-'*78}")

    for label, (h, wg, wh) in configs.items():
        res = best_first_search(start, goal, grid, heuristic=h, weight_g=wg, weight_h=wh)
        results[label] = res
        admissible = "Oui" if label != "h = 0  (UCS)" else "Oui (triviale)"
        if res["success"]:
            print(f"  {label:<22} {admissible:>11} {res['cost']:>8.2f} {res['nodes_exp']:>12} "
                  f"{res['open_max']:>10} {res['time_ms']:>10.3f}")
        else:
            print(f"  {label:<22} {'FAIL':>8}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"E.3 — Comparaison heuristiques admissibles — {name}", fontsize=13, fontweight='bold')

    colors = {"h = 0  (UCS)": "#FF9800", "h = Manhattan (A*)": "#2196F3", "h = Euclidean (A*)": "#4CAF50"}

    for ax, (label, res) in zip(axes, results.items()):
        _plot_grid(ax, grid, res.get("path", []), start, goal, label, res)

    # Sous-figure : barres nœuds développés
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    labels = list(results.keys())
    nodes  = [results[l]["nodes_exp"] for l in labels]
    open_m = [results[l]["open_max"]  for l in labels]
    x = range(len(labels))
    bars1 = ax2.bar([i - 0.2 for i in x], nodes,  width=0.4, label="Nœuds développés", color="#2196F3", alpha=0.85)
    bars2 = ax2.bar([i + 0.2 for i in x], open_m, width=0.4, label="OPEN max",          color="#FF5722", alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Nombre de nœuds", fontsize=11)
    ax2.set_title(f"Efficacité des heuristiques — {name}", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{name}_heuristic_paths.png",  dpi=120, bbox_inches='tight')
    fig2.savefig(f"{OUTPUT_DIR}/{name}_heuristic_bars.png",  dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"  → Figures sauvegardées : {name}_heuristic_paths.png / {name}_heuristic_bars.png")
    return results


# ---------------------------------------------------------------------------
# MAIN — Exécution de toutes les expériences
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MINI-PROJET : A* + Chaînes de Markov sur grille")
    print("="*60)

    # --- Grille simple ---
    grid1 = make_grid_simple()
    start1, goal1 = (0, 0), (7, 7)
    experiment_algo_comparison(grid1, start1, goal1, name="simple")
    r1 = astar(start1, goal1, grid1)
    if r1["success"]:
        visualize_grid_path(grid1, r1["path"], start1, goal1,
                            title="Grille simple — chemin A*", filename="simple_path.png")
    experiment_epsilon_impact(grid1, start1, goal1, name="simple")
    experiment_markov_analysis(grid1, start1, goal1, epsilon=0.1, name="simple")

    # --- Labyrinthe ---
    grid2 = make_grid_maze()
    start2, goal2 = (0, 0), (9, 9)
    experiment_algo_comparison(grid2, start2, goal2, name="maze")
    r2 = astar(start2, goal2, grid2)
    if r2["success"]:
        visualize_grid_path(grid2, r2["path"], start2, goal2,
                            title="Labyrinthe — chemin A*", filename="maze_path.png")
    experiment_markov_analysis(grid2, start2, goal2, epsilon=0.1, name="maze")

    # --- Grille pondérée ---
    grid3 = make_grid_weighted()
    start3, goal3 = (0, 0), (7, 7)
    experiment_algo_comparison(grid3, start3, goal3, name="weighted")

    # --- E.3 : Comparaison heuristiques admissibles ---
    experiment_heuristic_comparison(grid1, start1, goal1, name="simple")
    experiment_heuristic_comparison(grid2, start2, goal2, name="maze")

    print("\n" + "="*60)
    print("  Toutes les expériences terminées.")
    print(f"  Fichiers dans : {OUTPUT_DIR}")
    print("="*60)
