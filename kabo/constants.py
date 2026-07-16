"""
Constants for the CO2RR Bayesian Optimization pipeline.

Two CO2RR variants are described here, each with its own 19-descriptor
schema, product columns, explicit design space bounds and defaults:

* **Photocatalytic** (``ALL_FEATURE_COLUMNS`` / ``PRODUCT_COLUMNS``) —
  read by ``CO2RRTask``.  Yields are reported per product as ``Y_*``.
* **Electrocatalytic** (``ECO2RR_*``) — read by ``ECO2RRTask``.  Products
  are reported as Faradaic efficiencies ``FE_*`` (percent), which sum to
  at most 100% across all products of a single measurement.

Both variants share the same competing side reaction (HER → H₂), but the
electrocatalytic schema is driven by applied potential rather than
photosensitizer excitation, and declares three *categorical* descriptors
(metal, cation, cell type) that the photocatalytic schema does not have.
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


# ===========================================================================
#  ELECTROCATALYTIC CO2RR — ALL 19 DESCRIPTORS
#
#  Mirrors the photocatalytic schema above in size and spirit, but the
#  driving force is an applied electrode potential rather than a
#  photosensitizer, so the descriptor groups differ:
#
#      Catalyst (6) · Electrode/GDE (3) · Electrolyte (5) · Cell (5)
#
#  Three descriptors are CATEGORICAL (metal, cation, cell type) and two
#  are INTEGER (d-electron count, cation charge); see ECO2RR_FEATURE_TYPES.
# ===========================================================================
ECO2RR_FEATURE_COLUMNS: list[str] = [
    # Catalyst
    "Metal_identity", "Metal_CO_binding_energy", "Metal_H_binding_energy",
    "Metal_d_electron_count", "Particle_size", "Roughness_factor",
    # Electrode / gas-diffusion electrode
    "Catalyst_loading", "Ionomer_loading", "Catalyst_layer_thickness",
    # Electrolyte
    "Cation", "Cation_charge", "Electrolyte_concentration",
    "Electrolyte_pH", "Electrolyte_conductivity",
    # Cell & operating conditions
    "Cell_type", "Applied_potential", "CO2_partial_pressure",
    "CO2_flow_rate", "Temperature",
]

# ===========================================================================
#  ELECTROCATALYTIC CO2RR — CATEGORICAL VALUE SETS
#
#  Categorical descriptors are ORDINAL-ENCODED in the DataFrame: a value
#  is the index of the category in these lists (Cu → 0, Ag → 1, …).  The
#  design-space bounds below are therefore (0, len(values) - 1), and the
#  engine snaps these dims back onto that integer grid after continuous
#  acquisition relaxation.
#
#  Metals are ordered by the classic Hori product-selectivity grouping:
#  C2+-capable (Cu) → CO-selective (Ag, Au, Zn) → formate-selective
#  (Sn, Bi).  The order carries no numeric meaning for the GP, which
#  applies a CategoricalKernel (Hamming-style) to these dims.
# ===========================================================================
ECO2RR_CATEGORICAL_VALUES: dict[str, list[str]] = {
    "Metal_identity": ["Cu", "Ag", "Au", "Zn", "Sn", "Bi"],
    "Cation":         ["Li", "Na", "K", "Cs"],
    "Cell_type":      ["H-cell", "flow-cell", "MEA"],
}

# ===========================================================================
#  ELECTROCATALYTIC CO2RR — FEATURE TYPES
#  Consumed by ECO2RRTask.feature_types(); the engine routes categorical
#  dims to MixedSingleTaskGP(cat_dims=...) and snaps integer dims to grid.
# ===========================================================================
ECO2RR_INTEGER_FEATURES: frozenset[str] = frozenset({
    "Metal_d_electron_count",   # count of d electrons (Zn/Cd d10 → Ni d8)
    "Cation_charge",            # +1 (alkali) … +3 (Al3+)
})

# ===========================================================================
#  ELECTROCATALYTIC CO2RR — DESIGN SPACE BOUNDS
#  As with the photocatalytic schema, these are physical feasibility
#  ranges — NOT data-derived — so BO may explore beyond observed data.
#  Categorical dims use (0, n_categories - 1) in ordinal-encoded space.
# ===========================================================================
ECO2RR_DESIGN_SPACE_BOUNDS: dict[str, tuple[float, float]] = {
    # ---- Catalyst ----
    "Metal_identity":          (0, 5),        # ordinal code, see values above
    "Metal_CO_binding_energy": (-2.0, 0.5),   # eV, *CO adsorption (Cu ≈ -0.5)
    "Metal_H_binding_energy":  (-1.0, 0.8),   # eV, *H adsorption (HER descriptor)
    "Metal_d_electron_count":  (5, 10),       # count (Mn d5 → Cu/Zn d10)
    "Particle_size":           (1.0, 100.0),  # nm
    "Roughness_factor":        (1.0, 200.0),  # ECSA / geometric area
    # ---- Electrode / GDE ----
    "Catalyst_loading":        (0.1, 5.0),    # mg/cm²
    "Ionomer_loading":         (0.0, 40.0),   # wt% (Nafion / Sustainion)
    "Catalyst_layer_thickness": (0.1, 50.0),  # µm
    # ---- Electrolyte ----
    "Cation":                  (0, 3),        # ordinal code, see values above
    "Cation_charge":           (1, 3),        # elementary charges
    "Electrolyte_concentration": (0.05, 3.0), # mol/L (e.g. KHCO3, KOH)
    "Electrolyte_pH":          (6.0, 15.0),   # pH (bicarbonate → strong alkaline)
    "Electrolyte_conductivity": (1.0, 300.0), # mS/cm
    # ---- Cell & operating conditions ----
    "Cell_type":               (0, 2),        # ordinal code, see values above
    "Applied_potential":       (-1.8, -0.2),  # V vs. RHE (cathodic)
    "CO2_partial_pressure":    (0.1, 10.0),   # bar
    "CO2_flow_rate":           (1.0, 100.0),  # sccm
    "Temperature":             (10.0, 80.0),  # °C
}

# ===========================================================================
#  ELECTROCATALYTIC CO2RR PRODUCT COLUMNS
#  Each column stores the Faradaic Efficiency (FE%) of a product.  Unlike
#  photocatalytic yields, FE values of one measurement are constrained:
#  they are non-negative and sum to <= 100% (the remainder is unaccounted
#  charge).  ECO2RRTask enforces this in prompts and simulation.
#
#  C3H7OH (n-propanol) has no photocatalytic counterpart; it is included
#  because Cu electrodes reach measurable C3 selectivity at high
#  overpotential.
# ===========================================================================
ECO2RR_PRODUCT_COLUMNS: dict[str, str] = {
    "CO":     "FE_CO",       # Carbon monoxide
    "HCOOH":  "FE_HCOOH",    # Formic acid / formate
    "CH4":    "FE_CH4",      # Methane
    "C2H4":   "FE_C2H4",     # Ethylene
    "CH3OH":  "FE_CH3OH",    # Methanol
    "C2H5OH": "FE_C2H5OH",   # Ethanol
    "C3H7OH": "FE_C3H7OH",   # n-Propanol
    "H2":     "FE_H2",       # Hydrogen (competing HER)
}

# Ordered list of all Faradaic-efficiency column names
ECO2RR_ALL_PRODUCT_COLUMNS: list[str] = list(ECO2RR_PRODUCT_COLUMNS.values())

# Human-readable product names → column name mapping
ECO2RR_PRODUCT_NAMES: dict[str, str] = {
    v: k for k, v in ECO2RR_PRODUCT_COLUMNS.items()
}

# Default target product (CO is the canonical two-electron benchmark and
# the dominant product on Ag / Au / Zn cathodes)
ECO2RR_DEFAULT_TARGET_PRODUCT: str = "CO"

# Total Faradaic efficiency cap (%) used for validation / simulation.
ECO2RR_FE_TOTAL_MAX: float = 100.0
