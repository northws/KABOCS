#!/usr/bin/env python
"""Prior Predictive Check & Validation Tool.

Validates a JSON prior file against the design space, checks parameter
sanity, and runs a prior predictive check by sampling from the specified
distributions.

Usage:
    python scripts/validate_prior.py priors/my_prior.json [--n-samples 1000]

Output is printed to stdout and optionally saved to output/prior_checks/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from co2rr_bo.constants import DESIGN_SPACE_BOUNDS


def validate_prior(config_path: str, n_samples: int = 1000) -> bool:
    """Validate a prior JSON file and run prior predictive check.

    Returns True if all checks pass, False otherwise.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return False

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    errors: list[str] = []
    warnings: list[str] = []
    feature_reports: list[dict] = []

    # Filter out meta keys
    feature_keys = [k for k in config if not k.startswith("_")]

    print(f"\n{'='*60}")
    print(f"  Prior Predictive Check: {path.name}")
    print(f"  Features with priors: {len(feature_keys)}")
    print(f"  Samples per feature: {n_samples}")
    print(f"{'='*60}\n")

    for feat in feature_keys:
        params = config[feat]
        ptype = params.get("type", "uniform").lower()
        confidence = params.get("confidence", "unspecified")
        evidence = params.get("evidence", "none")

        # Check if feature is in design space
        if feat not in DESIGN_SPACE_BOUNDS:
            warnings.append(f"  ⚠ '{feat}' not in DESIGN_SPACE_BOUNDS — ignored at runtime")
            continue

        lo, hi = DESIGN_SPACE_BOUNDS[feat]
        report = {"feature": feat, "type": ptype, "confidence": confidence}

        if ptype == "gaussian":
            mean = float(params.get("mean", 0.0))
            std = float(params.get("std", 1.0))

            if std <= 0:
                errors.append(f"  ❌ '{feat}': std must be > 0 (got {std})")
                continue

            samples = np.random.normal(mean, std, n_samples)
            in_bounds = np.sum((samples >= lo) & (samples <= hi)) / n_samples
            report["in_bounds_ratio"] = round(in_bounds, 3)

            if mean < lo or mean > hi:
                warnings.append(
                    f"  ⚠ '{feat}': Gaussian mean ({mean}) is outside "
                    f"design bounds [{lo}, {hi}]"
                )
            if in_bounds < 0.5:
                warnings.append(
                    f"  ⚠ '{feat}': Only {in_bounds:.1%} of samples fall "
                    f"within design bounds — prior may be too spread or misaligned"
                )

            print(f"  ✅ {feat}: Gaussian(μ={mean}, σ={std})")
            print(f"     Bounds: [{lo}, {hi}] | In-bounds: {in_bounds:.1%}")
            print(f"     Confidence: {confidence} | Evidence: {evidence}")

        elif ptype == "uniform":
            pmin = float(params.get("min", -np.inf))
            pmax = float(params.get("max", np.inf))

            if pmin >= pmax:
                errors.append(f"  ❌ '{feat}': min ({pmin}) >= max ({pmax})")
                continue

            overlap_lo = max(pmin, lo)
            overlap_hi = min(pmax, hi)
            if overlap_lo >= overlap_hi:
                warnings.append(
                    f"  ⚠ '{feat}': Uniform [{pmin}, {pmax}] has NO overlap "
                    f"with design bounds [{lo}, {hi}]"
                )
                report["overlap_ratio"] = 0.0
            else:
                overlap = (overlap_hi - overlap_lo) / (pmax - pmin)
                report["overlap_ratio"] = round(overlap, 3)

            print(f"  ✅ {feat}: Uniform[{pmin}, {pmax}]")
            print(f"     Bounds: [{lo}, {hi}] | Overlap: {report.get('overlap_ratio', 'N/A')}")
            print(f"     Confidence: {confidence} | Evidence: {evidence}")

        else:
            errors.append(f"  ❌ '{feat}': Unknown type '{ptype}'")
            continue

        feature_reports.append(report)
        print()

    # Summary
    print(f"{'='*60}")
    if errors:
        print(f"  ❌ FAILED — {len(errors)} error(s):")
        for e in errors:
            print(e)
    if warnings:
        print(f"  ⚠ {len(warnings)} warning(s):")
        for w in warnings:
            print(w)
    if not errors and not warnings:
        print("  ✅ ALL CHECKS PASSED")
    elif not errors:
        print("  ✅ PASSED with warnings (see above)")
    print(f"{'='*60}\n")

    # Save report
    output_dir = Path("output/prior_checks")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{path.stem}_check.json"
    report_data = {
        "source_file": str(path),
        "version": config.get("_version", "unknown"),
        "n_samples": n_samples,
        "errors": errors,
        "warnings": warnings,
        "features": feature_reports,
        "passed": len(errors) == 0,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {report_path}")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate expert prior JSON")
    parser.add_argument("config", help="Path to prior JSON file")
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of samples for prior predictive check (default: 1000)"
    )
    args = parser.parse_args()

    ok = validate_prior(args.config, args.n_samples)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
