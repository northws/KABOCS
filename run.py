#!/usr/bin/env python3
"""
KABO Bayesian Optimization — Entry Point
========================================

Convenience script to run the pipeline from the project root.

Usage:
    python run.py                              # Interactive, CO2RR default
    python run.py --task co2rr                 # Explicit CO2RR task
    python run.py --task test --non-interactive  # Minimal TestTask smoke
    python run.py --non-interactive            # Demo mode
    python run.py --data data/data.csv         # Custom data
    python run.py --top-k 8 --beta 3.0        # Custom parameters
"""

from kabo.cli import main

if __name__ == "__main__":
    main()
