"""
CLI entry point for the KABO Bayesian Optimization pipeline.

The CLI is **task-aware**: pass ``--task <name>`` to select the active
system (default: ``co2rr``).  All downstream behaviour — feature schema,
product columns, default target, interactive prompts — is supplied by
the corresponding ``TaskBase`` implementation registered via
``kabo.task.register_task``.

Usage::

    python -m kabo                                   # CO2RR, default target
    python -m kabo --task co2rr --target-product HCOOH
    python -m kabo --task test --non-interactive     # minimal TestTask smoke
    python -m kabo --non-interactive                 # Demo mode
    python -m kabo --data data/data.csv              # Custom data
    python -m kabo --top-k 8 --beta 3.0             # Custom params
"""

from __future__ import annotations

import argparse

from kabo.config import load_config_file, merge_config_into_args
from kabo.task import TASK_REGISTRY, get_task

# NOTE: ``kabo.optimizer`` pulls torch / botorch at import time, so the
# heavy import is deferred into ``main()``.  This keeps ``build_parser``
# and ``parse_args`` importable in torch-free environments (unit tests,
# documentation builds, the WebUI project-editor validator).


def build_parser() -> argparse.ArgumentParser:
    """Build the KABO CLI argument parser.

    Exposed as a free function so that downstream tooling (e.g. the
    WebUI project-editor CLI helper, config-file merge tests) can
    reuse the same definitions without re-parsing ``sys.argv``.
    """
    registered_tasks = ", ".join(sorted(TASK_REGISTRY.keys()))

    parser = argparse.ArgumentParser(
        description=(
            "KABO Bayesian Optimization Pipeline "
            "(default: engineering-enhanced mode with RF feature selection)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Registered tasks:
  {registered_tasks}

Examples:
  python -m kabo                                     # Optimize CO (CO2RR default)
  python -m kabo --task co2rr --target-product HCOOH # Optimize formic acid
  python -m kabo --task test --non-interactive       # Minimal test task
  python -m kabo --non-interactive                   # Demo mode
  python -m kabo --data data/data.csv                # Custom data
  python -m kabo --top-k 8 --beta 3.0               # Custom params
  python run.py --candidates data/candidates.csv     # With discrete candidates
        """,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to a YAML / TOML / JSON configuration file.  Values from "
            "this file override argparse defaults but are themselves "
            "overridden by any flag explicitly passed on the command line."
        ),
    )
    parser.add_argument(
        "--task", type=str, default="co2rr",
        help=f"Active system task (default: co2rr). "
             f"Registered: {registered_tasks}",
    )
    parser.add_argument(
        "--data", type=str, default="data/data.csv",
        help="Path to input CSV dataset (default: data/data.csv)",
    )
    parser.add_argument(
        "--candidates", type=str, default="data/candidates.csv",
        help="Path to discrete candidates CSV (default: data/candidates.csv). "
             "Pass 'none' (or an empty string) to skip discrete candidates "
             "entirely — useful when switching tasks whose schema differs "
             "from the default CO2RR candidates file.",
    )
    parser.add_argument(
        "--target-product", type=str, default=None,
        help="Product to optimize (task-specific; defaults to the active "
             "task's default product).",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top features to select (default: 10)",
    )
    parser.add_argument(
        "--beta", type=float, default=2.0,
        help=(
            "UCB exploration parameter β (this implementation uses "
            "mu + beta * sigma; default: 2.0)"
        ),
    )
    parser.add_argument(
        "--beta-schedule", type=str,
        choices=["fixed", "theory", "theory-strict"],
        default="fixed",
        help=(
            "Beta schedule mode: fixed uses constant beta; theory uses "
            "a scaled time-varying beta_t; theory-strict uses pure "
            "theoretical beta_t without user scaling"
        ),
    )
    parser.add_argument(
        "--acq-strategy", type=str, choices=["ucb", "qnei"],
        default="ucb",
        help=(
            "Acquisition strategy: ucb (analytic UCB) or "
            "qnei (Monte Carlo noisy expected improvement)"
        ),
    )
    parser.add_argument(
        "--qnei-mc-samples", type=int, default=128,
        help=(
            "Number of QMC samples for qNEI Monte Carlo estimation "
            "(default: 128)"
        ),
    )
    parser.add_argument(
        "--kernel-type", type=str, default="matern",
        choices=["matern", "spectral_mixture"],
        help="Surrogate model kernel type (default: matern). See CatBOX literature.",
    )
    parser.add_argument(
        "--h2-penalty-weight", type=float, default=0.0,
        help=(
            "If > 0, optimize composite target: target - weight * Y_H2 "
            "to discourage HER (CO2RR-specific; default: 0.0)"
        ),
    )
    parser.add_argument(
        "--beta-delta", type=float, default=0.1,
        help=(
            "Confidence delta for theory beta_t schedule (0,1); "
            "ignored when --beta-schedule fixed"
        ),
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of BO iterations (default: 10)",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="Run in non-interactive demo mode (simulates user input)",
    )
    parser.add_argument(
        "--skip-feature-selection", action="store_true",
        help=(
            "Skip RF feature selection and use all task-declared features "
            "directly (paper-minimal BO mode)"
        ),
    )
    parser.add_argument(
        "--fs-method",
        choices=["random_forest", "permutation", "mutual_info", "shap"],
        default="random_forest",
        dest="feature_selection_method",
        help=(
            "Feature-ranking backend.  'random_forest' (legacy default) "
            "uses Gini importance; 'permutation' uses "
            "sklearn.inspection.permutation_importance on a fit RF; "
            "'mutual_info' uses sklearn's MI-regression; 'shap' uses "
            "TreeSHAP mean |phi| (requires optional `shap` package)."
        ),
    )
    parser.add_argument(
        "--permutation-repeats", type=int, default=5,
        help=(
            "Number of random shuffles per feature for "
            "--fs-method permutation (default 5)."
        ),
    )
    parser.add_argument(
        "--mutual-info-n-neighbors", type=int, default=3,
        help=(
            "n_neighbors for the k-NN density estimator used by "
            "--fs-method mutual_info (default 3)."
        ),
    )
    heatmap_group = parser.add_mutually_exclusive_group()
    heatmap_group.add_argument(
        "--correlation-heatmap", dest="correlation_heatmap",
        action="store_true", default=True,
        help="Emit a Pearson correlation heatmap of descriptors (default).",
    )
    heatmap_group.add_argument(
        "--no-correlation-heatmap", dest="correlation_heatmap",
        action="store_false",
        help="Suppress the correlation heatmap artifact.",
    )
    parser.add_argument(
        "--strict-training-schema", action="store_true",
        help=(
            "Require every task-declared descriptor column in training "
            "data (paper reproduction strict mode)"
        ),
    )
    parser.add_argument(
        "--pre-fill-before-choice", action="store_true",
        help=(
            "Pre-fill continuous non-selected feature values before "
            "candidate choice to show complete recipes"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Global random seed for reproducible runs "
            "(controls NumPy/Torch/Python RNGs)"
        ),
    )
    parser.add_argument(
        "--kabo-mode", action="store_true",
        help="Enable Knowledge-Augmented Bayesian Optimization mode",
    )
    parser.add_argument(
        "--lambda-p", type=float, default=1.0,
        help="Weight for preference model score in KABO (default: 1.0)",
    )
    parser.add_argument(
        "--lambda-k", type=float, default=1.0,
        help="Weight for expert prior score in KABO (default: 1.0)",
    )
    parser.add_argument(
        "--expert-prior-file", type=str, default=None,
        help="Path to JSON file containing expert priors (for KABO mode)",
    )
    parser.add_argument(
        "--diversity-weight", type=float, default=0.5,
        help=(
            "Weight for diversity in Top-N recommendation menu "
            "(0=pure score, 1=strong diversity; default: 0.5)"
        ),
    )
    parser.add_argument(
        "--pe-budget", type=int, default=0,
        help=(
            "Number of preference exploration queries per iteration "
            "(KABO mode; 0=disabled; default: 0)"
        ),
    )
    parser.add_argument(
        "--pe-pool-cap", type=int, default=None,
        help=(
            "Optional cap on the PE query candidate pool: when the pool "
            "exceeds this, a uniform random subsample is used for pair "
            "scoring (O(m^2) → O(cap^2)). Default: no cap."
        ),
    )
    parser.add_argument(
        "--pe-strategy",
        choices=["uncertainty", "random"],
        default="uncertainty",
        help=(
            "PE pair scoring strategy: 'uncertainty' (PEBO; var_i + var_j "
            "− |μ_i − μ_j|) or 'random' (distinct uniform pairs, useful "
            "for ablation and cold-start parity). Default: uncertainty."
        ),
    )
    parser.add_argument(
        "--lambda-v", type=float, default=0.0,
        help=(
            "Weight for approximate VOI term in KABO acquisition "
            "(0=disabled; default: 0.0)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for output files (default: output)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Torch device: 'auto', 'cpu', or 'cuda' (default: auto)",
    )
    parser.add_argument(
        "--generate-candidates-n", type=int, default=1000,
        help=(
            "Size of the dynamic candidate pool generated by "
            "Task.generate_candidates() when available (default: 1000). "
            "Only effective if the active Task implements the generator "
            "and --prefer-file-candidates is not set."
        ),
    )
    parser.add_argument(
        "--prefer-file-candidates", action="store_true",
        help=(
            "Force using the --candidates CSV file even when the active "
            "Task implements a dynamic generate_candidates() method. "
            "Useful for reproducing legacy runs."
        ),
    )
    parser.add_argument(
        "--discrete-strategy", type=str, default="acq",
        choices=["acq", "thompson"],
        help=(
            "Strategy for ranking the discrete candidate pool. "
            "'acq' (default): score every candidate with the acquisition "
            "function (legacy behaviour). "
            "'thompson': draw Top-N independent posterior samples and "
            "return their argmaxes — natural exploration diversity, "
            "recommended when the pool is large (e.g. dynamic generator)."
        ),
    )
    # ------------------------------------------------------------------ #
    #  v1.2 additions: batch recommendation + early stopping
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--q-batch", type=int, default=1,
        help=(
            "Number of continuous candidates to propose per iteration "
            "(default 1). q>1 uses joint batch optimization (qNEI) or "
            "sequential-greedy restarts (UCB); all candidates compete "
            "against the discrete pool in the Top-N ranking."
        ),
    )
    parser.add_argument(
        "--max-stagnation", type=int, default=0,
        help=(
            "Stop the BO loop early when the best target value has not "
            "improved by more than --stagnation-tol for N consecutive "
            "iterations.  0 (default) disables early stopping."
        ),
    )
    parser.add_argument(
        "--stagnation-tol", type=float, default=1e-4,
        help=(
            "Absolute improvement threshold used by --max-stagnation "
            "(default 1e-4)."
        ),
    )
    # ------------------------------------------------------------------ #
    #  v1.2 additions: multi-objective BO (qNEHVI)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--multi-objective", action="store_true",
        help=(
            "Enable multi-objective Bayesian Optimization via qNEHVI. "
            "Fits one GP per objective (ModelListGP) and optimises the "
            "Pareto front.  Objectives default to the active task's "
            "`multi_objectives()` preset unless overridden by "
            "--objectives.  Writes pareto_front.{csv,png} to the run "
            "output directory."
        ),
    )
    parser.add_argument(
        "--objectives", nargs="+", default=None, metavar="NAME",
        help=(
            "Explicit list of objective columns / short-names to feed "
            "into qNEHVI, e.g. `--objectives CO HCOOH`.  Implicitly "
            "enables --multi-objective when present.  Accepts task "
            "target short names (CO, HCOOH) or raw column names "
            "(Y_CO, Y_HCOOH).  All objectives are assumed maximise; "
            "use --objectives with an in-code ObjectiveSpec for minimise."
        ),
    )
    parser.add_argument(
        "--ref-point", nargs="+", type=float, default=None, metavar="VAL",
        help=(
            "Hypervolume reference point (raw scale, one value per "
            "objective in declared order).  When omitted, it is "
            "inferred from the observed data with a 10%% margin.  For "
            "'min' objectives, provide the raw upper bound — sign "
            "flipping is handled internally."
        ),
    )
    parser.add_argument(
        "--qnehvi-mc-samples", type=int, default=128,
        help=(
            "Number of Monte Carlo samples for qNEHVI posterior "
            "integration (default 128).  Higher = smoother / slower."
        ),
    )
    # ------------------------------------------------------------------ #
    #  v1.2 additions: variational / sparse GP for large datasets
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--gp-model",
        choices=["exact", "variational", "auto"],
        default="auto",
        dest="gp_model_type",
        help=(
            "GP backend for the surrogate.  'exact' uses the classic "
            "SingleTaskGP with an O(N^3) Cholesky (accurate for small "
            "N); 'variational' uses SingleTaskVariationalGP with "
            "inducing points and O(N m^2) cost (SVGP, recommended for "
            "N >~ 200); 'auto' (default) picks variational when the "
            "training set is large and exact otherwise."
        ),
    )
    parser.add_argument(
        "--num-inducing-points", type=int, default=None,
        help=(
            "Number of inducing points for the variational GP (ignored "
            "for 'exact').  Defaults to min(N, 100).  Increase for "
            "denser posteriors at the cost of O(N m^2) compute."
        ),
    )
    parser.add_argument(
        "--svgp-epochs", type=int, default=200,
        help=(
            "Adam epochs used to train the variational ELBO (default "
            "200).  Ignored for 'exact'."
        ),
    )
    parser.add_argument(
        "--svgp-lr", type=float, default=1e-2,
        help="Learning rate for the variational ELBO Adam loop (default 1e-2).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, merging in an optional ``--config``.

    Parameters
    ----------
    argv : list[str] or None
        Argument vector to parse; ``None`` means ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Final namespace after optional config-file merge.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "config", None):
        cfg = load_config_file(args.config)
        args = merge_config_into_args(args, parser, cfg)
    return args


def main() -> None:
    """Main entry point for the KABO optimization pipeline."""
    # parse_args() is torch-free and handles ``--help`` / ``--version`` by
    # itself, so we defer the heavy optimizer import until *after* argument
    # parsing succeeds.  This makes ``python -m kabo --help`` usable in
    # torch-free environments (e.g. from the WebUI config editor).
    args = parse_args()
    from kabo.optimizer import KABOOptimizer  # noqa: E402 — intentional late import

    task = get_task(args.task)
    # Let the task choose the default target when user did not specify.
    target_product = (
        args.target_product
        if args.target_product is not None
        else task.default_target()
    )

    # Resolve --candidates: 'none'/'' => skip discrete candidates entirely.
    candidates_path = args.candidates
    if candidates_path is not None and candidates_path.strip().lower() in {
        "", "none", "null"
    }:
        candidates_path = None

    optimizer = KABOOptimizer(
        data_path=args.data,
        task=task,
        target_product=target_product,
        top_k=args.top_k,
        beta=args.beta,
        beta_schedule=args.beta_schedule,
        beta_delta=args.beta_delta,
        acq_strategy=args.acq_strategy,
        qnei_mc_samples=args.qnei_mc_samples,
        kernel_type=args.kernel_type,
        h2_penalty_weight=args.h2_penalty_weight,
        candidates_path=candidates_path,
        skip_feature_selection=args.skip_feature_selection,
        feature_selection_method=args.feature_selection_method,
        correlation_heatmap=args.correlation_heatmap,
        permutation_repeats=args.permutation_repeats,
        mutual_info_n_neighbors=args.mutual_info_n_neighbors,
        strict_training_schema=args.strict_training_schema,
        pre_fill_before_choice=args.pre_fill_before_choice,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        kabo_mode=args.kabo_mode,
        lambda_p=args.lambda_p,
        lambda_k=args.lambda_k,
        expert_prior_file=args.expert_prior_file,
        diversity_weight=args.diversity_weight,
        pe_budget=args.pe_budget,
        pe_pool_cap=args.pe_pool_cap,
        pe_strategy=args.pe_strategy,
        lambda_v=args.lambda_v,
        generate_candidates_n=args.generate_candidates_n,
        prefer_file_candidates=args.prefer_file_candidates,
        discrete_strategy=args.discrete_strategy,
        q_batch=args.q_batch,
        max_stagnation=args.max_stagnation,
        stagnation_tol=args.stagnation_tol,
        multi_objective=args.multi_objective,
        objectives=args.objectives,
        ref_point=args.ref_point,
        qnehvi_mc_samples=args.qnehvi_mc_samples,
        gp_model_type=args.gp_model_type,
        num_inducing_points=args.num_inducing_points,
        svgp_epochs=args.svgp_epochs,
        svgp_lr=args.svgp_lr,
    )

    optimizer.run(
        n_iterations=args.iterations,
        interactive=not args.non_interactive,
    )


if __name__ == "__main__":
    main()
