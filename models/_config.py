"""Central configuration module for models4 pipeline."""

# =============================================================================
# DATA PATHS & FILES
# =============================================================================
FAKE_CSV     = "data/Fake.csv"
TRUE_CSV     = "data/True.csv"
VALIDATION_SIZE = 60   # Global validation pool size


# =============================================================================
# NLI MODEL CONFIGURATION (nli-deberta-v3-small)
# Label indices: CONTRADICTION=0, ENTAILMENT=1, NEUTRAL=2
NLI_CONTRADICTION = 0
NLI_ENTAILMENT    = 1
NLI_NEUTRAL       = 2


# =============================================================================
# SENTENCE ROLE CONFIGURATION (zero-shot NLI hypotheses)
# role_names[ROLE_INDEX] - maps integer to string name
ROLE_CLAIM      = "claim"     # factual assertions/evidence statements
ROLE_EVIDENCE   = "evidence"  # data/quotes/citations
ROLE_ANALYSIS   = "analysis"  # interpretation/opinion (KEY signal)
ROLE_BACKGROUND = "background"# general context/facts
ROLE_NAMES       = [           # in order of ROLE_INDEX + 1 (offset by -1 from claim index below)
    'claim',
    'evidence',
    'analysis',
    'background'
]

# =============================================================================
# EDGE TYPE CONFIGURATION
EDGE_NEUTRAL        = 0   # no semantic relation identified
EDGE_ENTAILMENT     = 1   # premise entails hypothesis (supports)
EDGE_CONTRADICTION  = 2   # premise contradicts hypothesis


# =============================================================================
# NODE FEATURE DIMENSION (includes role one-hot + cross-source signals)
FEATURE_DIM         = 20
