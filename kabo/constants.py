"""
Constants for the CO2RR Bayesian Optimization pipeline.

Defines all 19 physicochemical descriptors, CO2RR product columns,
explicit design space bounds, and default settings.
"""

# ===========================================================================
#  ALL 19 DESCRIPTORS (ordered as specified in the prompt)
# ===========================================================================
ALL_FEATURE_COLUMNS: list[str] = [
    # Amino Acid A
    "A_pI", "A_distance", "A_hbond_acceptors", "A_hbond_donors",
    # Amino Acid B
    "B_pI", "B_distance", "B_hbond_acceptors", "B_hbond_donors",
    # Porphyrin MOF
    "MOF_potential", "M_CO_binding_energy",
    # Photosensitizer
    "PS_absorption_wavelength", "PS_potential",
    # Solvent
    "Solvent_dielectric", "Solvent_hbond_acceptors", "Solvent_hbond_donors",
    "CO2_solubility",
    # Reaction Conditions
    "H2O_concentration", "Sacrificial_agent_potential",
    "Sacrificial_agent_concentration",
]

# ===========================================================================
#  DESIGN SPACE BOUNDS
#  Explicit physical bounds for each descriptor, defining the full
#  experimental design space that BO is allowed to explore.
#
#  These are NOT derived from training data — they represent the
#  physically feasible range for each feature. GP normalization and
#  acquisition function optimization use these bounds so that BO can
#  explore beyond observed data.
#
#  Format: {feature_name: (lower_bound, upper_bound)}
#  Update these if your experimental system has different constraints.
# ===========================================================================
DESIGN_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    # Amino Acid A  (common natural amino acids)
    "A_pI":                (2.7, 10.8),    # pH, range of 20 natural AAs
    "A_distance":          (1.5, 7.0),     # Å, side‐chain to metal node
    "A_hbond_acceptors":   (0, 6),         # count
    "A_hbond_donors":      (0, 4),         # count
    # Amino Acid B
    "B_pI":                (2.7, 10.8),    # pH
    "B_distance":          (1.5, 7.0),     # Å
    "B_hbond_acceptors":   (0, 6),         # count
    "B_hbond_donors":      (0, 4),         # count
    # Porphyrin MOF
    "MOF_potential":       (-1.5, -0.3),   # V vs. NHE
    "M_CO_binding_energy": (-2.5, -0.5),   # eV
    # Photosensitizer
    "PS_absorption_wavelength": (380, 700),  # nm (visible range)
    "PS_potential":        (0.8, 2.0),     # V vs. NHE (excited state)
    # Solvent
    "Solvent_dielectric":  (2.0, 110.0),   # ε_r (hexane → formamide)
    "Solvent_hbond_acceptors": (0, 6),     # count
    "Solvent_hbond_donors":    (0, 4),     # count
    "CO2_solubility":      (0.001, 0.3),   # mol/L
    # Reaction Conditions
    "H2O_concentration":           (0.0, 100.0),  # vol%
    "Sacrificial_agent_potential":  (-0.8, 0.0),   # V vs. NHE
    "Sacrificial_agent_concentration": (0.01, 1.0),  # mol/L
}

# ===========================================================================
#  CO2RR PRODUCT YIELD COLUMNS
#  Each column stores the Faradaic Efficiency (FE%) or yield (µmol/g·h)
#  of a specific product from photocatalytic CO2 reduction.
# ===========================================================================
PRODUCT_COLUMNS: dict[str, str] = {
    "CO":     "Y_CO",       # Carbon monoxide
    "HCOOH":  "Y_HCOOH",    # Formic acid / formate
    "CH4":    "Y_CH4",      # Methane
    "C2H4":   "Y_C2H4",     # Ethylene
    "CH3OH":  "Y_CH3OH",    # Methanol
    "C2H5OH": "Y_C2H5OH",   # Ethanol
    "H2":     "Y_H2",       # Hydrogen (competing HER)
}

# Ordered list of all product yield column names
ALL_PRODUCT_COLUMNS: list[str] = list(PRODUCT_COLUMNS.values())

# Human-readable product names → column name mapping
PRODUCT_NAMES: dict[str, str] = {v: k for k, v in PRODUCT_COLUMNS.items()}

# Default target product for optimization (CO is the most common
# dominant product for porphyrin MOF-based CO2RR systems)
DEFAULT_TARGET_PRODUCT: str = "CO"

# Legacy single-target column (kept for backward compatibility)
TARGET_COLUMN: str = "Y"
