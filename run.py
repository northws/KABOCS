#!/usr/bin/env python3
"""
CO2RR Bayesian Optimization — Entry Point
==========================================

Convenience script to run the pipeline from the project root.

Usage:
    python run.py                           # Interactive mode
    python run.py --non-interactive         # Demo mode
    python run.py --data data/data.csv      # Custom data
    python run.py --top-k 8 --beta 3.0     # Custom parameters
"""

from co2rr_bo.cli import main

if __name__ == "__main__":
    main()
