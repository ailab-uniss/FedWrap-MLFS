"""Experiment runner — the entry point for a single fold.

``run_experiment_from_config(config, fold_idx)`` performs:

1. Load pre-folded dataset (train / val / test).
2. Build the ML-kNN evaluator with tri-objective scoring.
3. Construct the flat binary-mask genotype.
4. Run NSGA-II with sliding-window early stopping on HV.
5. Save Pareto front, population masks, and test-set metrics.

The function is called by ``fedwrap.cli.run`` for each fold.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .config import Paths, get, load_config
from .datasets import load_dataset
from .genotypes import BitstringConfig, bitstring_crossover, bitstring_mutate, init_bitstring
from .metrics import hypervolume_3d, pareto_nondominated, pareto_nondominated_mask
from .ml_eval import EvalConfig, Evaluator
from .federated import make_evaluator, is_federated_enabled
from .federated.evaluator import FederatedEvaluator
from .federated.metrics import per_label_f1
from .fedaware import FedAwareConfig, FedAwareVariation, federated_relevance
from .nsga2 import Variation, nsga2
from .utils import JsonlLogger
from .logging_utils import setup_run_logger


# ═══════════════════════════════════════════════════════════════════
# Variation adapter (bridge genotype operators → NSGA-II protocol)
# ═══════════════════════════════════════════════════════════════════

class BitstringVariation(Variation):
    """Plain variation operator for the flat binary-mask genotype (federation-naive baseline).
    FedAware-NSGA-II replaces it with :class:`fedwrap.fedaware.FedAwareVariation`."""
    def __init__(self, cfg: BitstringConfig) -> None:
        self.cfg = cfg

    def crossover(self, a: object, b: object, rng: np.random.Generator) -> tuple[object, object]:
        return bitstring_crossover(np.asarray(a, dtype=bool), np.asarray(b, dtype=bool), rng)

    def mutate(self, a: object, rng: np.random.Generator) -> object:
        return bitstring_mutate(np.asarray(a, dtype=bool), self.cfg, rng)


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def run_experiment_from_config(config: dict[str, Any], fold_idx: int | None = None) -> Path:
    """Run a single experiment from a configuration dictionary.

    Args:
        config: Parsed YAML configuration.
        fold_idx: Fold index for cross-validation (0-based).

    Returns:
        Path to the output directory.
    """
    seed = int(config.get("seed", 42))
    paths = Paths.from_config(config)
    if fold_idx is not None:
        suffix = f"_fold{int(fold_idx)}"
        if not paths.out_dir.name.endswith(suffix):
            paths = Paths(out_dir=paths.out_dir.with_name(paths.out_dir.name + suffix))
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    runlog = setup_run_logger(paths.out_dir, name=f"fedwrap.run.{paths.out_dir.as_posix()}")
    log = runlog.logger
    logger = JsonlLogger(paths.out_dir / "history.jsonl")

    # Save the exact config used.
    try:
        import yaml
        with open(paths.out_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except Exception:
        pass

    log.info("run.start out_dir=%s seed=%s fold_idx=%s", paths.out_dir, seed, fold_idx)

    # ── Load dataset ──────────────────────────────────────────────
    # Natural-silo federated runs load the prepared dataset directly (aligned groups).
    _fed_groups = None
    _fed_groups_test = None
    if is_federated_enabled(config) and str(get(config, "federated.partition", "")) == "natural_silo":
        from .federated import load_fed_natural_split
        from .datasets import DatasetSplit as _DS
        d = load_fed_natural_split(
            root=str(get(config, "dataset.root", "data/fed_real")),
            name=str(get(config, "dataset.name", "")),
            seed=seed, val_size=float(get(config, "dataset.split.val_size", 0.25)),
        )
        ds = _DS(x_train=d["x_train"], y_train=d["y_train"], x_val=d["x_val"],
                 y_val=d["y_val"], x_test=d["x_test"], y_test=d["y_test"])
        _fed_groups = (d["groups_train"], d["groups_val"])
        # The test evaluator trains on x_full = vstack(x_train, x_val); its group vector
        # must therefore be in the SAME [train, val] row order, not the original trainval
        # order (groups_trainval), otherwise rows are assigned to the wrong silo.
        _fed_groups_test = (np.concatenate([d["groups_train"], d["groups_val"]]), d["groups_test"])
    else:
        ds = load_dataset(config, seed=seed, fold_idx=fold_idx)
    n_features = ds.x_train.shape[1]
    n_labels = ds.y_train.shape[1]
    log.info("dataset x_train=%s y_train=%s x_val=%s y_val=%s",
             ds.x_train.shape, ds.y_train.shape, ds.x_val.shape, ds.y_val.shape)

    # ── Build evaluator ───────────────────────────────────────────
    obj_names = config.get("objectives", {}).get("names")
    if obj_names and isinstance(obj_names, list):
        objective_names = [Evaluator._canonical(str(n)) for n in obj_names]
    else:
        objective_names = None

    eval_cfg = EvalConfig(
        kind=str(get(config, "model.kind", "mlknn")),
        primary_objective=objective_names[0] if objective_names else "one_minus_macro_f1",
        objective_names=objective_names,
        random_state=seed,
        k=int(get(config, "model.k", 5)),
        s=float(get(config, "model.s", 1.0)),
        mlknn_backend=str(get(config, "model.mlknn_backend", "auto")),
        mlknn_device=str(get(config, "model.mlknn_device", "auto")),
        cv_folds=int(get(config, "model.cv_folds", 1)),
    )
    evaluator = make_evaluator(ds.x_train, ds.y_train, ds.x_val, ds.y_val, eval_cfg, config, seed, groups=_fed_groups)
    federated = is_federated_enabled(config)
    log.info("evaluator kind=%s objectives=%s federated=%s", eval_cfg.kind, objective_names, federated)

    # ── Evolution parameters ──────────────────────────────────────
    evo_cfg = config.get("evolution", {})
    pop_size = int(evo_cfg.get("pop_size", 50))
    crossover_prob = float(evo_cfg.get("crossover_prob", 0.9))
    mutation_prob = float(evo_cfg.get("mutation_prob", 0.5))
    genotype_kind = str(evo_cfg.get("genotype", "bitstring"))
    log.info("evolution genotype=%s pop=%d cx=%.2f mut=%.2f",
             genotype_kind, pop_size, crossover_prob, mutation_prob)

    counts = {"real": 0}

    # ── FedAware-NSGA-II configuration (federation-aware search over flat masks) ──
    fa_cfg = FedAwareConfig(
        enabled=bool(get(config, "fedaware.enabled", False)),
        stability_tiebreak=bool(get(config, "fedaware.stability_tiebreak", True)),
        disagreement_mutation=bool(get(config, "fedaware.disagreement_mutation", True)),
        disagreement_prob=float(get(config, "fedaware.disagreement_prob", 0.5)),
        relevance_pool=int(get(config, "fedaware.relevance_pool", 20)),
        hardness_temperature=float(get(config, "fedaware.hardness_temperature", 0.5)),
        relevance_warmstart=bool(get(config, "fedaware.relevance_warmstart", True)),
        warmstart_frac=float(get(config, "fedaware.warmstart_frac", 0.3)),
        warmstart_jitter=float(get(config, "fedaware.warmstart_jitter", 0.10)),
        filter_seed=bool(get(config, "fedaware.filter_seed", False)),
        swap_prob=float(get(config, "fedaware.swap_prob", 0.0)),
    )
    fa_variation = None  # set below when fedaware is enabled on the bitstring genotype

    # ── Genotype setup ────────────────────────────────────────────
    if genotype_kind == "bitstring":
        bcfg = BitstringConfig(
            init_prob=float(get(config, "bitstring.init_prob", 0.1)),
            bitflip_prob=float(get(config, "bitstring.bitflip_prob", 1.0 / max(1, n_features))),
            bitflip_prob_on=get(config, "bitstring.bitflip_prob_on", None),
            bitflip_prob_off=get(config, "bitstring.bitflip_prob_off", None),
        )
        if fa_cfg.enabled and isinstance(evaluator, FederatedEvaluator):
            R, global_rel = federated_relevance(evaluator.clients, n_features, n_labels)
            fa_variation = FedAwareVariation(bcfg, fa_cfg, R, global_rel, n_labels)
            variation: Variation = fa_variation
            log.info("FedAware-NSGA-II: relevance sketch R=%s stability_tiebreak=%s disagreement=%s",
                     R.shape, fa_cfg.stability_tiebreak, fa_cfg.disagreement_mutation)
        else:
            variation = BitstringVariation(bcfg)

        def _filter_seeds(mr: float) -> list[np.ndarray]:
            """Strong federated-filter masks (fed-rank, local-top-k, top-frequency) at several
            sparsities, used as initialization priors so the wrapper refines them rather than
            rediscovering them (helps when local filters are strong, e.g. ExtraSensory)."""
            from .federated.baselines import (fed_rank_relevance, ranking_to_mask,
                                              local_topk_union, topk_frequency_scores)
            cl = evaluator.clients
            try:
                fr = fed_rank_relevance(cl)
            except Exception:
                return []
            seeds = []
            for r in sorted({0.05, 0.10, 0.20, float(mr)}):
                for mk in (ranking_to_mask(fr, r), local_topk_union(cl, r),
                           ranking_to_mask(topk_frequency_scores(cl, r), r)):
                    seeds.append(np.asarray(mk, dtype=bool))
            return seeds

        def init_pop(rng: np.random.Generator) -> list[object]:
            if fa_variation is not None and fa_cfg.relevance_warmstart:
                mr = float(get(config, "reporting.max_feature_ratio", 0.25))
                pop: list[object] = []
                if fa_cfg.filter_seed and isinstance(evaluator, FederatedEvaluator):
                    pop.extend(_filter_seeds(mr)[:pop_size])
                rest = max(0, pop_size - len(pop))
                pop.extend(fa_variation.seed_population(rest, rng, max_ratio=mr))
                rng.shuffle(pop)
                return pop[:pop_size]
            return [init_bitstring(n_features, bcfg, rng) for _ in range(pop_size)]

        def to_mask(genome: object) -> np.ndarray:
            m = np.asarray(genome, dtype=bool)
            if m.sum() == 0:
                m = m.copy(); m[0] = True
            return m

    else:
        raise ValueError(f"Unsupported genotype: {genotype_kind!r} (only 'bitstring' is supported)")

    # ── Evaluate function ─────────────────────────────────────────
    def evaluate(genome: object) -> tuple[np.ndarray, dict[str, object]]:
        mask = to_mask(genome)
        obj, ml = evaluator.evaluate_mask(mask)
        counts["real"] += 1
        meta = {"selected": int(mask.sum()),
                "feature_ratio": float(mask.sum() / mask.size),
                "ml": asdict(ml)}
        # FedAware: attach client-stability risk + per-label F1 (from the same sufficient
        # statistics) for the stability tie-break and disagreement-guided mutation.
        if fa_cfg.enabled and isinstance(evaluator, FederatedEvaluator):
            st = evaluator.client_metrics_for(mask)
            if st is not None:
                meta["client_risk"] = float(st.std_client_macro_f1)
                meta["worst_client"] = float(st.worst_client_macro_f1)
                meta["label_f1"] = per_label_f1(st.tp, st.fp, st.fn)
        return obj, meta

    # ── Early stopping (sliding-window on HV) ─────────────────────
    n_obj = len(objective_names) if objective_names else 2
    ref_list = [float(get(config, f"objectives.hv_ref.{i}", 1.0)) for i in range(n_obj)]
    ref = tuple(ref_list)

    es_cfg = evo_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    es_window = int(es_cfg.get("window", 10))
    es_rel_tol = float(es_cfg.get("rel_tol", 0.002))
    es_patience = int(es_cfg.get("patience", 2))

    hv_history: list[float] = []
    stagnant_count = 0
    last_gen = -1

    def on_generation(gen: int, pop: list[Any]) -> bool | None:
        nonlocal last_gen, stagnant_count
        last_gen = gen

        # FedAware: refocus disagreement-guided mutation on the labels the current
        # population handles worst (mean per-label F1 over individuals that carry it).
        if fa_variation is not None and fa_cfg.disagreement_mutation:
            lfs = [p.meta["label_f1"] for p in pop
                   if p.meta is not None and p.meta.get("label_f1") is not None]
            if lfs:
                fa_variation.update_hardness(np.mean(np.stack(lfs, axis=0), axis=0))

        objs = np.stack([p.objectives for p in pop], axis=0)
        nd = pareto_nondominated(objs)
        hv = hypervolume_3d(nd, ref=ref) if n_obj == 3 else float(np.prod(np.array(ref[:2]) - nd.min(axis=0)))

        stop = False
        if es_enabled:
            hv_history.append(hv)
            if len(hv_history) >= 2 * es_window:
                prev_w = np.mean(hv_history[-(2 * es_window):-es_window])
                curr_w = np.mean(hv_history[-es_window:])
                rel = (curr_w - prev_w) / max(abs(prev_w), 1e-12)
                if rel < es_rel_tol:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                if stagnant_count >= es_patience:
                    log.info("Early stopping at gen=%d (HV stagnant for %d checks)", gen, es_patience)
                    stop = True

        log.info("gen=%d hv=%.6f nd=%d evals=%d wait=%d/%d",
                 gen, hv, nd.shape[0], counts["real"], stagnant_count, es_patience)
        logger.log({"gen": gen, "hv": hv, "n_nd": nd.shape[0], "evals": counts["real"],
                     "stagnant": stagnant_count})
        return stop

    # ── Budget ────────────────────────────────────────────────────
    max_evals_pf = evo_cfg.get("max_evals_per_feature")
    if max_evals_pf is not None:
        max_evals = int(n_features * float(max_evals_pf))
    else:
        max_evals = int(evo_cfg.get("max_evals", n_features * 100))
    log.info("max_evals=%d", max_evals)

    # ── Run NSGA-II ───────────────────────────────────────────────
    final_pop = nsga2(
        init_population=init_pop,
        evaluate=evaluate,
        variation=variation,
        pop_size=pop_size,
        max_evals=max_evals,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        seed=seed,
        on_generation=on_generation,
        tie_breaker=(lambda ind: (ind.meta or {}).get("client_risk", 0.0))
                    if (fa_cfg.enabled and fa_cfg.stability_tiebreak) else None,
        stability_blend=float(get(config, "fedaware.stability_blend", 0.15))
                        if (fa_cfg.enabled and fa_cfg.stability_tiebreak) else 0.0,
    )

    # ═══════════════════════════════════════════════════════════════
    # Post-processing: save results
    # ═══════════════════════════════════════════════════════════════

    # Pareto front from surrogate objectives.
    objs = np.stack([p.objectives for p in final_pop], axis=0)
    nd = pareto_nondominated(objs)
    np.savetxt(paths.out_dir / "pareto_front.csv", nd, delimiter=",")

    # Real-evaluated Pareto front (validation set).
    masks: list[np.ndarray] = []
    val_objs_list: list[np.ndarray] = []
    val_mls: list[Any] = []
    for ind in final_pop:
        mask = to_mask(ind.genome)
        vo, vm = evaluator.evaluate_mask(mask)
        masks.append(mask)
        val_objs_list.append(vo)
        val_mls.append(vm)

    val_objs_arr = np.stack(val_objs_list) if val_objs_list else np.zeros((0, n_obj))
    nd_val = pareto_nondominated(val_objs_arr)
    np.savetxt(paths.out_dir / "pareto_front_real.csv", nd_val, delimiter=",")

    # Test-set evaluation (train = train ∪ val). This is a single held-out evaluation
    # (train+val -> test), never cross-validated: CV is only the search/selection objective.
    x_full = sparse.vstack([ds.x_train, ds.x_val]).tocsr()
    y_full = sparse.vstack([ds.y_train, ds.y_val]).tocsr()
    test_eval_cfg = replace(eval_cfg, cv_folds=1)
    test_eval = make_evaluator(x_full, y_full, ds.x_test, ds.y_test, test_eval_cfg, config, seed, groups=_fed_groups_test)

    test_objs_list: list[np.ndarray] = []
    test_mls: list[Any] = []
    for mask in masks:
        to, tm = test_eval.evaluate_mask(mask)
        test_objs_list.append(to)
        test_mls.append(tm)
    test_objs_arr = np.stack(test_objs_list) if test_objs_list else np.zeros((0, n_obj))
    nd_test = pareto_nondominated(test_objs_arr)
    np.savetxt(paths.out_dir / "pareto_front_test.csv", nd_test, delimiter=",")

    # Population masks (packed, for post-hoc analysis).
    try:
        masks_arr = np.stack(masks).astype(np.uint8)
        np.savez_compressed(
            paths.out_dir / "population_masks.npz",
            masks_packed=np.packbits(masks_arr, axis=1),
            n_features=n_features,
            val_objs=val_objs_arr.astype(np.float32),
            test_objs=test_objs_arr.astype(np.float32),
            pareto_val_mask=pareto_nondominated_mask(val_objs_arr).astype(np.uint8),
        )
    except Exception:
        pass

    # Human-readable Pareto solutions (JSON).
    nd_mask = pareto_nondominated_mask(val_objs_arr)
    pareto_solutions = []
    for i, keep in enumerate(nd_mask.tolist()):
        if not bool(keep):
            continue
        m = masks[i]
        pareto_solutions.append({
            "i": i,
            "objectives": val_objs_arr[i].tolist(),
            "selected": int(m.sum()),
            "selected_features": np.flatnonzero(m).astype(int).tolist(),
            "val": asdict(val_mls[i]),
            "test": asdict(test_mls[i]),
        })
    (paths.out_dir / "pareto_val_solutions.json").write_text(json.dumps({
        "train_for_test": "train+val",
        "model_kind": eval_cfg.kind,
        "fold_idx": fold_idx,
        "n_features": n_features,
        "solutions": pareto_solutions,
    }, indent=2))

    # Budget-selected solutions (select by val, report test).
    budget_mode = str(get(config, "reporting.budget_mode", "count"))
    budget_step = int(get(config, "reporting.budget_step", 1))
    max_ratio = float(get(config, "reporting.max_feature_ratio", 0.25))
    M = int(np.floor(max_ratio * n_features))
    budgets = list(range(1, max(1, M) + 1, budget_step))

    feat_col = int(objective_names.index("feature_ratio")) if (objective_names and "feature_ratio" in objective_names) else 1
    selected_counts = np.array([int(m.sum()) for m in masks], dtype=int)
    by_budget: list[dict] = []

    for budget in budgets:
        eligible = np.flatnonzero(selected_counts <= int(budget))
        if eligible.size == 0:
            continue
        best_i = int(min(eligible.tolist(),
                         key=lambda i: (float(val_objs_arr[i, 0]), float(val_objs_arr[i, feat_col]))))
        by_budget.append({
            "budget": int(budget),
            "selected": int(masks[best_i].sum()),
            "selected_features": np.flatnonzero(masks[best_i]).astype(int).tolist(),
            "val": asdict(val_mls[best_i]),
            "test": asdict(test_mls[best_i]),
        })
    (paths.out_dir / "test_selected_by_val.json").write_text(json.dumps({
        "train_for_test": "train+val",
        "model_kind": eval_cfg.kind,
        "fold_idx": fold_idx,
        "budget_mode": budget_mode,
        "by_budget": by_budget,
    }, indent=2))

    # Summary.
    hv_val = hypervolume_3d(nd_val, ref=ref) if n_obj == 3 else 0.0
    hv_test = hypervolume_3d(nd_test, ref=ref) if n_obj == 3 else 0.0
    summary_obj = {
        "seed": seed, "fold_idx": fold_idx,
        "dataset": config.get("dataset", {}),
        "evolution": config.get("evolution", {}),
        "final": {
            "gen": last_gen, "pareto_points_val": nd_val.shape[0],
            "hv_val": hv_val, "pareto_points_test": nd_test.shape[0],
            "hv_test": hv_test, "total_evals": counts["real"],
        },
    }
    if federated and isinstance(evaluator, FederatedEvaluator):
        fed_summary = evaluator.summary()
        summary_obj["federated"] = {
            "enabled": True,
            "n_clients": fed_summary.get("n_clients"),
            "partition": fed_summary.get("partition"),
            "counters": fed_summary.get("counters"),
            "communication": fed_summary.get("communication"),
        }
        if isinstance(test_eval, FederatedEvaluator) and pareto_solutions:
            best_i = min(pareto_solutions, key=lambda s: s["objectives"][0])["i"]
            ts = test_eval.client_metrics_for(masks[best_i])
            if ts is not None:
                summary_obj["final"]["worst_client_macro_f1"] = ts.worst_client_macro_f1
                summary_obj["final"]["std_client_macro_f1"] = ts.std_client_macro_f1
                summary_obj["final"]["macro_f1_best"] = float(test_mls[best_i].f1_macro)
                summary_obj["final"]["micro_f1_best"] = float(test_mls[best_i].f1_micro)
            comm = fed_summary.get("communication")
            if comm is not None:
                (paths.out_dir / "communication.json").write_text(json.dumps(comm, indent=2))
    (paths.out_dir / "summary.json").write_text(json.dumps(summary_obj, indent=2))

    log.info("run.done out_dir=%s pareto=%d", paths.out_dir, nd_val.shape[0])
    runlog.close()
    return paths.out_dir


def run_experiment(config_path: str | Path) -> Path:
    """Load YAML config and run."""
    return run_experiment_from_config(load_config(config_path))
