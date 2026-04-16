"""
CLI entry point for the CO2RR Bayesian Optimization pipeline.

Usage::

    python -m co2rr_bo                                 # Interactive, optimize CO
    python -m co2rr_bo --target-product HCOOH          # Optimize formic acid
    python -m co2rr_bo --non-interactive                # Demo mode
    python -m co2rr_bo --data data/data.csv             # Custom data
    python -m co2rr_bo --top-k 8 --beta 3.0            # Custom params
"""

from __future__ import annotations

import argparse

from co2rr_bo.constants import DEFAULT_TARGET_PRODUCT, PRODUCT_COLUMNS
from co2rr_bo.optimizer import CO2RROptimizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    valid_products = ", ".join(PRODUCT_COLUMNS.keys())

    parser = argparse.ArgumentParser(
        description=(
            "CO2RR Bayesian Optimization Pipeline "
            "(default: engineering-enhanced mode with RF feature selection)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
CO2RR Products:
  {valid_products}

Examples:
  python -m co2rr_bo                                # Optimize CO (default)
  python -m co2rr_bo --target-product HCOOH         # Optimize formic acid
  python -m co2rr_bo --target-product CH4            # Optimize methane
  python -m co2rr_bo --non-interactive               # Demo mode
  python -m co2rr_bo --data data/data.csv            # Custom data
  python -m co2rr_bo --top-k 8 --beta 3.0           # Custom params
  python run.py --candidates data/candidates.csv     # With discrete candidates
        """,
    )
    parser.add_argument(
        "--data", type=str, default="data/data.csv",
        help="Path to input CSV dataset (default: data/data.csv)",
    )
    parser.add_argument(
        "--candidates", type=str, default="data/candidates.csv",
        help="Path to discrete candidates CSV (default: data/candidates.csv)",
    )
    parser.add_argument(
        "--target-product", type=str, default=DEFAULT_TARGET_PRODUCT,
        help=f"Product to optimize: {valid_products} "
             f"(default: {DEFAULT_TARGET_PRODUCT})",
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
            "to discourage HER (default: 0.0)"
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
            "Skip RF feature selection and use all 19 features directly "
            "(paper-minimal BO mode)"
        ),
    )
    parser.add_argument(
        "--strict-training-schema", action="store_true",
        help=(
            "Require all 19 descriptor columns in training data "
            "(paper reproduction strict mode)"
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
    return parser.parse_args()


def main() -> None:
    """Main entry point for the CO2RR optimization pipeline."""
    args = parse_args()

    optimizer = CO2RROptimizer(
        data_path=args.data,
        target_product=args.target_product,
        top_k=args.top_k,
        beta=args.beta,
        beta_schedule=args.beta_schedule,
        beta_delta=args.beta_delta,
        acq_strategy=args.acq_strategy,
        qnei_mc_samples=args.qnei_mc_samples,
        kernel_type=args.kernel_type,
        h2_penalty_weight=args.h2_penalty_weight,
        candidates_path=args.candidates,
        skip_feature_selection=args.skip_feature_selection,
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
        lambda_v=args.lambda_v,
    )

    optimizer.run(
        n_iterations=args.iterations,
        interactive=not args.non_interactive,
    )


if __name__ == "__main__":
    main()
