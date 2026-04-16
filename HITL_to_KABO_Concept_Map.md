# HITL to KABO Evolution Concept Map

## Conceptual Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            FROM HITL to KABO: Knowledge Internalization Pathway             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │  Human      │
                              │  Expert     │
                              └──────┬──────┘
                                     │
                                     ▼
         ┌───────────────────────────────────────────────────┐
         │              PHASE 1: HITL-BO                      │
         │   (Human-in-the-Loop Bayesian Optimization)       │
         │                                                   │
         │   ┌────────────────┐    ┌────────────────┐        │
         │   │ Preference     │    │ Interactive    │        │
         │   │ Learning       │◀───│ Query          │        │
         │   │ (Astudillo     │    │ (Lin 2022)     │        │
         │   │ 2019)          │    └────────────────┘        │
         │   └────────┬───────┘                              │
         │            │                                      │
         │            ▼                                      │
         │   ┌────────────────┐                              │
         │   │ Expert         │    Key Papers:               │
         │   │ Guidance       │    - arXiv:1911.05934        │
         │   │ (Dewancker     │    - arXiv:2203.11382        │
         │   │ 2018)          │    - arXiv:1801.02788        │
         │   └────────────────┘                              │
         └───────────────────────────┬───────────────────────┘
                                     │
                                     │ Knowledge Capture
                                     │ (Preferences, Constraints,
                                     │  Regions, Trajectories)
                                     ▼
         ┌───────────────────────────────────────────────────┐
         │              PHASE 2: Knowledge Encoding          │
         │                                                   │
         │   ┌────────────────┐    ┌────────────────┐        │
         │   │ Prior          │    │ Constraint     │        │
         │   │ Elicitation    │    │ Encoding       │        │
         │   │ (Mikkola       │    │ (Safe BO       │        │
         │   │ 2021)          │    │ literature)    │        │
         │   └────────┬───────┘    └────────┬───────┘        │
         │            │                     │                │
         │            ▼                     ▼                │
         │   ┌────────────────┐    ┌────────────────┐        │
         │   │ Knowledge      │    │ Preference     │        │
         │   │ Gradient       │    │ Model          │        │
         │   │ (Wu 2016)      │    │ (Bradley-Terry)│        │
         │   │ - Quantifies   │    │ - BT model     │        │
         │   │   knowledge    │    │ - Thurstone    │        │
         │   │   value        │    └────────────────┘        │
         │   └────────────────┘                              │
         │                                                   │
         │   Key Papers:                                     │
         │   - arXiv:1606.04414 (KG)                         │
         │   - arXiv:2002.11256 (Expert Prior)               │
         │   - arXiv:2112.01380 (Prior Elicitation)          │
         └───────────────────────────┬───────────────────────┘
                                     │
                                     │ Knowledge Representation
                                     │ (PDF, Prior Distribution,
                                     │  Acquisition Bias)
                                     ▼
         ┌───────────────────────────────────────────────────┐
         │              PHASE 3: KABO                        │
         │   (Knowledge-Augmented Bayesian Optimization)    │
         │                                                   │
         │   ┌────────────────────────────────────────┐      │
         │   │   KABO Acquisition Function           │      │
         │   │                                        │      │
         │   │   α_KABO(x) = α_base(x)               │      │
         │   │              + λ_K · KG_expert(x)     │      │
         │   │              + λ_P · Preference(x)    │      │
         │   │                                        │      │
         │   │   where:                               │      │
         │   │   - α_base: EI/UCB base              │      │
         │   │   - KG_expert: Knowledge gradient    │      │
         │   │   - Preference: Preference score     │      │
         │   │   - λ: adaptive weights              │      │
         │   └────────────────────────────────────────┘      │
         │                                                   │
         │   ┌────────────────┐    ┌────────────────┐        │
         │   │ Prior-Enhanced │    │ Self-Evolving  │        │
         │   │ Surrogate      │    │ Knowledge      │        │
         │   │ Model          │    │ Base           │        │
         │   └────────────────┘    └────────────────┘        │
         │                                                   │
         │   Features:                                       │
         │   ✓ Reduced human interaction burden             │
         │   ✓ Knowledge persistence across tasks           │
         │   ✓ Adaptive knowledge weighting                 │
         │   ✓ Domain transferability                       │
         └───────────────────────────────────────────────────┘
                                     │
                                     │ Continuous Learning
                                     │ (New expert feedback
                                     │  updates knowledge)
                                     ▼
                              ┌─────────────┐
                              │ Knowledge   │
                              │ Accumulation│
                              │ & Refinement│
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │  KABO      │
                              │  Mature    │
                              │  System    │
                              └─────────────┘
```

## Key Technical Transitions

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   HITL Phase    │────▶│ Encoding Phase  │────▶│   KABO Phase    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ - Preference    │     │ - Prior PDF     │     │ - Augmented     │
│   queries       │     │ - Constraint    │     │   acquisition   │
│ - Interactive   │     │   functions     │     │ - Prior-aware   │
│   settings      │     │ - Knowledge     │     │   surrogate     │
│ - Expert        │     │   gradient      │     │ - Self-adaptive │
│   guidance      │     │ - Preference    │     │   weights       │
└─────────────────┘     │   model         │     └─────────────────┘
                        └─────────────────┘
```

## Knowledge Type Encoding Map

```
┌────────────────────┬───────────────────────┬───────────────────────┐
│  Knowledge Type    │  Encoding Method      │  BO Integration       │
├────────────────────┼───────────────────────┼───────────────────────┤
│ Preference data    │ Bradley-Terry model   │ Preference-scored     │
│ (pairwise)         │ Thurstone model       │ acquisition           │
├────────────────────┼───────────────────────┼───────────────────────┤
│ Region knowledge   │ Prior distribution    │ Region-biased         │
│ (promising areas)  │ (e.g., truncated)     │ sampling              │
├────────────────────┼───────────────────────┼───────────────────────┤
│ Constraint rules   │ Constraint function   │ Safe BO methods       │
│ (feasibility)      │ Implicit constraint   │ Constrained BO        │
├────────────────────┼───────────────────────┼───────────────────────┤
│ Trajectory data    │ Expert trajectory     │ Imitation learning    │
│ (expert decisions) │ encoding              │ + acquisition bias    │
├────────────────────┼───────────────────────┼───────────────────────┤
│ Hyperparameter     │ Hyperprior            │ Hierarchical BO       │
│ knowledge          │ Meta-prior            │ Meta-learning         │
└────────────────────┴───────────────────────┴───────────────────────┘
```

## Implementation Roadmap for ml-co2rr

```
Phase 2 Enhancement (Current HITL)
    │
    ├── Add Preference Learning Module
    │   ├── Bradley-Terry preference model
    │   ├── Preference query generation
    │   └ Preference-based acquisition bias
    │
    ├── Expert Prior Integration
    │   ├── Product distribution prior
    │   ├── Parameter range constraints
    │   └── Feasibility region encoding
    │
Phase 3 Evolution (KABO)
    │
    ├── Knowledge-Augmented Acquisition
    │   ├── α_KABO(x) = α_UCB(x) + λ·KG_expert(x)
    │   ├── Dynamic weight adjustment
    │   └── Knowledge contribution tracking
    │
    ├── Knowledge Accumulation System
    │   ├── Historical preference storage
    │   ├── Task-specific knowledge bank
    │   ├── Cross-session knowledge persistence
    │
    └── Self-Evolution Mechanism
        ├── Meta-learning for λ adaptation
        ├── Knowledge transfer across products
        ├── Online knowledge refinement
```

---
Generated: 2026-04-16
Purpose: Visualize the pathway from HITL to KABO for ml-co2rr project