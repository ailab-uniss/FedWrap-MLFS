"""NSGA-II multi-objective evolutionary algorithm.

Implements the standard NSGA-II loop (Deb et al., 2002):
  1. Non-dominated sorting of the combined parent+offspring population.
  2. Crowding-distance assignment within each front.
  3. Binary-tournament selection, crossover, mutation.
  4. Environmental selection by rank then crowding.

The ``Variation`` protocol defines the genotype-specific operators
(crossover, mutate, repair).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


# ── Data types ────────────────────────────────────────────────────────

@dataclass
class Individual:
    """One member of the population."""
    genome: object
    objectives: np.ndarray   # shape (n_obj,), all minimised
    rank: int = 0
    crowding: float = 0.0
    meta: dict[str, object] | None = None


class Variation(Protocol):
    """Genotype-specific operators expected by :func:`nsga2`."""

    def crossover(
        self, a: object, b: object, rng: np.random.Generator,
    ) -> tuple[object, object]: ...

    def mutate(self, a: object, rng: np.random.Generator) -> object: ...

    def repair(self, a: object, rng: np.random.Generator) -> object:
        """Optional repair applied to every offspring.

        Default is identity (no-op); the flat-mask operators repair in place.
        """
        return a


# ── NSGA-II internals ─────────────────────────────────────────────────

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_nondominated_sort(objs: np.ndarray) -> list[list[int]]:
    """Assign Pareto fronts (front 0 = best) to *objs*."""
    n = objs.shape[0]
    s: list[list[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                s[p].append(q)
            elif _dominates(objs[q], objs[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in s[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(objs: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for one non-dominated front."""
    if not front:
        return np.array([], dtype=float)
    m = objs.shape[1]
    dist = np.zeros(len(front), dtype=float)
    fobjs = objs[front]
    for j in range(m):
        order = np.argsort(fobjs[:, j])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        minv = float(fobjs[order[0], j])
        maxv = float(fobjs[order[-1], j])
        if maxv == minv:
            continue
        for k in range(1, len(front) - 1):
            dist[order[k]] += (
                float(fobjs[order[k + 1], j]) - float(fobjs[order[k - 1], j])
            ) / (maxv - minv)
    return dist


def _risk(ind: Individual, tie_breaker) -> float:
    """Client-stability risk (lower is better), or 0 when unavailable."""
    if tie_breaker is None:
        return 0.0
    try:
        v = tie_breaker(ind)
        return float(v) if v is not None and np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def tournament_select(
    pop: list[Individual], rng: np.random.Generator, tie_breaker=None,
    crowd_tol: float = 0.05,
) -> Individual:
    """Binary tournament: prefer lower rank, then higher crowding; among equal rank with
    *similar* crowding, prefer lower client-stability risk (federation-aware tie-break)."""
    i, j = rng.integers(0, len(pop), size=2)
    a, b = pop[int(i)], pop[int(j)]
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    ca, cb = a.crowding, b.crowding
    if np.isfinite(ca) and np.isfinite(cb) and abs(ca - cb) <= crowd_tol * max(ca, cb, 1e-9):
        if tie_breaker is not None:
            ra, rb = _risk(a, tie_breaker), _risk(b, tie_breaker)
            if ra != rb:
                return a if ra < rb else b
        return a if rng.random() < 0.5 else b
    if ca != cb:
        return a if ca > cb else b
    return a if rng.random() < 0.5 else b


def _front_order(objs: np.ndarray, front: list[int], combined: list[Individual],
                 tie_breaker, stability_blend: float) -> np.ndarray:
    """Order indices within a front by crowding (desc); when a stability tie-breaker is given,
    blend a small standardized client-risk penalty so that near-equal-crowding solutions are
    ordered to prefer lower inter-client dispersion."""
    cd = crowding_distance(objs, front)
    if tie_breaker is None or stability_blend <= 0.0:
        return np.argsort(-cd), cd
    risk = np.array([_risk(combined[front[k]], tie_breaker) for k in range(len(front))])
    finite = cd[np.isfinite(cd)]
    scale = (finite.max() - finite.min()) if finite.size and finite.max() > finite.min() else 1.0
    rz = (risk - risk.mean()) / (risk.std() + 1e-9)
    key = cd - float(stability_blend) * scale * rz       # inf crowding stays on top
    return np.argsort(-key), cd


# ── Main loop ─────────────────────────────────────────────────────────

def nsga2(
    *,
    init_population: Callable[[np.random.Generator], list[object]],
    evaluate: Callable[[object], tuple[np.ndarray, dict[str, object]]],
    variation: Variation,
    pop_size: int,
    max_evals: int,
    crossover_prob: float,
    mutation_prob: float,
    seed: int,
    on_generation: (
        Callable[[int, list[Individual]], bool | None] | None
    ) = None,
    tie_breaker: Callable[[Individual], float] | None = None,
    stability_blend: float = 0.0,
) -> list[Individual]:
    """Run NSGA-II and return the final population."""
    rng = np.random.default_rng(seed)

    # ── Initialise ────────────────────────────────────────────────
    genomes = init_population(rng)
    pop: list[Individual] = []
    eval_count = 0
    for g in genomes:
        obj, meta = evaluate(g)
        pop.append(Individual(genome=g, objectives=obj, meta=meta))
        eval_count += 1

    gen = 0
    while True:
        if eval_count >= max_evals:
            break

        # Assign rank + crowding for parent selection.
        objs = np.stack([p.objectives for p in pop], axis=0)
        fronts = fast_nondominated_sort(objs)
        for rank, front in enumerate(fronts):
            cd = crowding_distance(objs, front)
            for idx_in_front, p_idx in enumerate(front):
                pop[p_idx].rank = rank
                pop[p_idx].crowding = float(cd[idx_in_front])

        # Callback (early stopping, logging).
        if on_generation is not None:
            stop = on_generation(gen, pop)
            if stop is True:
                break

        # ── Offspring generation ──────────────────────────────────
        offspring: list[Individual] = []
        child_genomes: list[object] = []
        while len(child_genomes) < pop_size:
            p1 = tournament_select(pop, rng, tie_breaker).genome
            p2 = tournament_select(pop, rng, tie_breaker).genome
            if rng.random() < crossover_prob:
                c1, c2 = variation.crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2

            mut1 = rng.random() < mutation_prob
            mut2 = rng.random() < mutation_prob
            if mut1:
                c1 = variation.mutate(c1, rng)
            if mut2:
                c2 = variation.mutate(c2, rng)

            # Repair only crossover-only children (mutated ones are
            # already repaired inside mutate).
            if not mut1:
                c1 = variation.repair(c1, rng)
            if not mut2:
                c2 = variation.repair(c2, rng)

            for c in (c1, c2):
                if len(child_genomes) >= pop_size:
                    break
                child_genomes.append(c)

        # ── Evaluate offspring ────────────────────────────────────
        for c in child_genomes:
            obj, meta = evaluate(c)
            offspring.append(Individual(genome=c, objectives=obj, meta=meta))
            eval_count += 1

        # ── Environmental selection ───────────────────────────────
        combined = pop + offspring
        objs = np.stack([p.objectives for p in combined], axis=0)
        fronts = fast_nondominated_sort(objs)
        new_pop: list[Individual] = []
        for rank, front in enumerate(fronts):
            order, cd = _front_order(objs, front, combined, tie_breaker, stability_blend)
            if len(new_pop) + len(front) <= pop_size:
                for oi in order:
                    idx = front[int(oi)]
                    ind = combined[idx]
                    ind.rank = rank
                    ind.crowding = float(cd[int(oi)])
                    new_pop.append(ind)
            else:
                for oi in order:
                    if len(new_pop) >= pop_size:
                        break
                    idx = front[int(oi)]
                    ind = combined[idx]
                    ind.rank = rank
                    ind.crowding = float(cd[int(oi)])
                    new_pop.append(ind)
                break
        pop = new_pop
        gen += 1

    return pop
