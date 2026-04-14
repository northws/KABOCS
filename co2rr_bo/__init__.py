"""
CO2RR Bayesian Optimization Package
====================================

A modular Python pipeline for optimizing photocatalytic CO2 reduction
reaction (CO2RR) systems using Bayesian Optimization.

The methodology follows arXiv:2604.01328v3:
    "Efficient and Principled Scientific Discovery through
     Bayesian Optimization: A Tutorial"

Typical usage::

    from co2rr_bo import CO2RROptimizer

    optimizer = CO2RROptimizer(data_path="data/data.csv")
    optimizer.run(n_iterations=10, interactive=True)
"""

from co2rr_bo.optimizer import CO2RROptimizer

__all__ = ["CO2RROptimizer"]
__version__ = "1.0.0"
