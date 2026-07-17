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


# ===========================================================================
#  AMINO-ACID DESCRIPTOR TABLE
#
#  Standard tabulated physicochemical descriptors for the 20 canonical
#  residues.  These exist so a Task can encode an amino-acid / peptide
#  ligand as CONTINUOUS descriptors rather than as a categorical identity.
#
#  Why that matters: a categorical feature carries no metric between its
#  levels, so a GP can never say anything about a residue it has not seen.
#  Descriptors give every residue — tested or not — a coordinate, which is
#  what lets BO extrapolate to untested residues at all.  (``CO2RRTask``
#  already applies the same idea via ``A_pI`` / ``A_hbond_*``.)
#
#  Scales:
#    pI       — isoelectric point
#    kd       — Kyte-Doolittle hydropathy (negative = hydrophilic)
#    volume   — Zamyatnin residue volume (Å^3)
#    hbd/hba  — side-chain H-bond donors / acceptors (counts)
#    charge   — net side-chain charge at pH 7 (His ~+0.1, pKa ~6.0)
# ===========================================================================
AA_DESCRIPTOR_NAMES: list[str] = ["pI", "kd", "volume", "hbd", "hba", "charge"]

# residue -> (pI, kd, volume, hbd, hba, charge@pH7)
AMINO_ACID_DESCRIPTORS: dict[str, tuple[float, float, float, float, float, float]] = {
    "Ala": (6.00,  1.8,  88.6, 0, 0,  0.0),
    "Arg": (10.76, -4.5, 173.4, 5, 0,  1.0),
    "Asn": (5.41, -3.5, 114.1, 2, 1,  0.0),
    "Asp": (2.77, -3.5, 111.1, 0, 2, -1.0),
    "Cys": (5.07,  2.5, 108.5, 1, 1,  0.0),
    "Gln": (5.65, -3.5, 143.8, 2, 1,  0.0),
    "Glu": (3.22, -3.5, 138.4, 0, 2, -1.0),
    "Gly": (5.97, -0.4,  60.1, 0, 0,  0.0),
    "His": (7.59, -3.2, 153.2, 1, 1,  0.1),
    "Ile": (6.02,  4.5, 166.7, 0, 0,  0.0),
    "Leu": (5.98,  3.8, 166.7, 0, 0,  0.0),
    "Lys": (9.74, -3.9, 168.6, 3, 0,  1.0),
    "Met": (5.74,  1.9, 162.9, 0, 1,  0.0),
    "Phe": (5.48,  2.8, 189.9, 0, 0,  0.0),
    "Pro": (6.30, -1.6, 112.7, 0, 0,  0.0),
    "Ser": (5.68, -0.8,  89.0, 1, 1,  0.0),
    "Thr": (5.60, -0.7, 116.1, 1, 1,  0.0),
    "Trp": (5.89, -0.9, 227.8, 1, 0,  0.0),
    "Tyr": (5.66, -1.3, 193.6, 1, 1,  0.0),
    "Val": (5.96,  4.2, 140.0, 0, 0,  0.0),
}

# Common three-letter aliases seen in lab notebooks / spreadsheets.
AA_ALIASES: dict[str, str] = {
    "gln": "Gln", "met": "Met", "his": "His", "arg": "Arg", "lys": "Lys",
    "asp": "Asp", "glu": "Glu", "gly": "Gly", "ala": "Ala", "ser": "Ser",
    "thr": "Thr", "cys": "Cys", "asn": "Asn", "pro": "Pro", "val": "Val",
    "ile": "Ile", "leu": "Leu", "phe": "Phe", "trp": "Trp", "tyr": "Tyr",
}


def aa_descriptor_bounds() -> dict[str, tuple[float, float]]:
    """Descriptor bounds spanning ALL 20 canonical residues.

    Derived from the table rather than hard-coded so that the design space
    provably contains every residue — an untested one must never fall
    outside the bounds the surrogate normalises with, or it would be
    silently clipped onto a different residue's coordinate.
    """
    # strict=: a silently truncated zip here would mis-align descriptor
    # names with their columns, i.e. hand back another descriptor's bounds.
    cols = list(zip(*AMINO_ACID_DESCRIPTORS.values(), strict=True))
    return {
        name: (float(min(col)), float(max(col)))
        for name, col in zip(AA_DESCRIPTOR_NAMES, cols, strict=True)
    }


# ===========================================================================
#  PEPTIDE-LIGATED ELECTROCATALYTIC CO2RR  (PeptideECO2RRTask)
#
#  A metal centre (Fe) carrying an amino-acid / peptide ligand, swept over
#  applied potential.  The ligand enters the design space ONLY through its
#  averaged residue descriptors above — there is deliberately no
#  categorical "which catalyst" dimension, because that is exactly what
#  would prevent BO from proposing a residue that has never been tested.
#
#  Products are gas-phase only (GC-quantified); there is no liquid-product
#  column here, which is why this task does not reuse ECO2RR_PRODUCT_COLUMNS.
# ===========================================================================
PEPTIDE_LIGAND_PREFIX: str = "Ligand_"

PEPTIDE_FEATURE_COLUMNS: list[str] = [
    "Ligand_pI", "Ligand_hydropathy", "Ligand_volume",
    "Ligand_hbond_donors", "Ligand_hbond_acceptors", "Ligand_charge",
    "Applied_potential",
]

# Maps a feature column back onto its entry in AA_DESCRIPTOR_NAMES.
PEPTIDE_LIGAND_DESCRIPTOR_MAP: dict[str, str] = {
    "Ligand_pI": "pI",
    "Ligand_hydropathy": "kd",
    "Ligand_volume": "volume",
    "Ligand_hbond_donors": "hbd",
    "Ligand_hbond_acceptors": "hba",
    "Ligand_charge": "charge",
}

# Applied potential sweep window (V vs. reference), from the measured range.
PEPTIDE_POTENTIAL_BOUNDS: tuple[float, float] = (-2.4, -1.0)

PEPTIDE_PRODUCT_COLUMNS: dict[str, str] = {
    "CO":    "FE_CO",      # Carbon monoxide (2 e-)
    "CH4":   "FE_CH4",     # Methane
    "C2H2":  "FE_C2H2",    # Acetylene
    "C2H4":  "FE_C2H4",    # Ethylene
    "C2H6":  "FE_C2H6",    # Ethane
    "C3H4":  "FE_C3H4",    # Propyne / allene
    "C3H6":  "FE_C3H6",    # Propylene
    "C3H8":  "FE_C3H8",    # Propane
    "C4H10": "FE_C4H10",   # Butane
    "H2":    "FE_H2",      # Hydrogen (competing HER)
}

PEPTIDE_ALL_PRODUCT_COLUMNS: list[str] = list(PEPTIDE_PRODUCT_COLUMNS.values())
PEPTIDE_PRODUCT_NAMES: dict[str, str] = {
    v: k for k, v in PEPTIDE_PRODUCT_COLUMNS.items()
}
PEPTIDE_DEFAULT_TARGET_PRODUCT: str = "CO"
