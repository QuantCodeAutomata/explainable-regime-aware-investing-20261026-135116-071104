"""
Configuration for Explainable Regime-Aware Investing replication.

Paper: "Explainable Regime-Aware Investing" (Amine Boukardagha, Columbia University)

Parameters marked [paper] are confirmed from the paper text (pages cited).
Parameters marked [assumed] are not specified in the paper; defaults are noted.
"""

# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

# Yahoo Finance tickers for the five-asset universe
# [paper p.4]: SPX (S&P 500 proxy), BOND, GOLD, OIL, USD — names only, no tickers given
# [assumed] Standard Yahoo Finance proxies:
TICKERS = {
    "SPX":  "^GSPC",   # S&P 500 Index
    "BOND": "TLT",     # iShares 20+ Year Treasury Bond ETF (broad bond proxy)
    "GOLD": "GLD",     # SPDR Gold Shares
    "OIL":  "USO",     # United States Oil Fund LP
    "USD":  "UUP",     # Invesco DB US Dollar Index Bullish Fund
}

ASSET_NAMES = list(TICKERS.keys())  # ["SPX", "BOND", "GOLD", "OIL", "USD"]

# [paper p.4]: "daily adjusted close prices from Yahoo Finance from 2005 to 2026"
DATA_START = "2005-01-01"
DATA_END   = "2026-01-31"

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

# [paper p.4]: "60-day rolling per-asset volatility"
VOL_WINDOW = 60

# [paper p.4]: "20-day rolling mean return"
MOM_WINDOW = 20

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

# [assumed] Not specified in paper; figures show OOS starts 2023-05; p.12 table starts 2023-05
TRAIN_TEST_SPLIT = "2023-05-01"

# Minimum warm-up period before OOS begins (need enough data for features)
WARMUP_DAYS = max(VOL_WINDOW, MOM_WINDOW) + 5

# ---------------------------------------------------------------------------
# Parametric Wasserstein HMM parameters
# ---------------------------------------------------------------------------

# [assumed] G=5 templates — from regime labels A, B, C, D, E visible in paper results (p.14-16)
N_TEMPLATES = 5

# [assumed] Candidate HMM orders — paper says K_min,...,K_max without values
K_MIN = 2
K_MAX = 3  # 2 candidate orders (K=2,3) for feasible runtime

# [paper p.7 alg.1]: "order selection frequency F_K (e.g., weekly)"
# [assumed] F_K=10 trading days (biweekly) — reduces HMM fits by 2x vs weekly
ORDER_SELECTION_FREQ = 10

# [assumed] Maximum training window for HMM fitting (rolling cap on expanding history)
# Paper says "expanding window"; capped at 750 days (~3 years) for feasible runtime.
MAX_TRAIN_WINDOW = 750

# [assumed] HMM covariance type — "diag" for feasible runtime vs. paper's implied "full"
# Diagonal covariance dramatically reduces EM computation while preserving regime structure.
HMM_COVAR_TYPE = "diag"

# [assumed] Validation slice length |V|; paper does not specify
VALIDATION_SLICE_LEN = 60

# [assumed] Complexity penalty lambda_K * K; paper says "proportional to K"
LAMBDA_K = 1.0

# [assumed] Template smoothing rate eta; paper does not specify
TEMPLATE_ETA = 0.05

# [assumed] Initial calibration window for template initialization
INIT_CALIBRATION_WINDOW = 252

# HMM fitting settings [assumed]
HMM_N_ITER = 30   # reduced from 100 for feasible runtime
HMM_TOL    = 1e-4
HMM_RANDOM_STATE = 42
HMM_N_INIT = 1    # reduced from 3 (single restart) for feasible runtime

# ---------------------------------------------------------------------------
# MVO optimization parameters
# ---------------------------------------------------------------------------

# [assumed] Risk aversion coefficient gamma; not specified in paper
GAMMA = 1.0

# [assumed] Transaction cost tau; paper reports ~0.79 bps avg turnover → suggests small tau
TAU = 0.001

# [assumed] Maximum weight per asset w_max
W_MAX = 0.5

# ---------------------------------------------------------------------------
# KNN parameters (exp_2)
# ---------------------------------------------------------------------------

# [assumed] Number of neighbors K; not specified in paper
KNN_K = 20

# [assumed] Minimum lookback L before KNN starts
KNN_LOOKBACK = 252

# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

# [assumed] Annualization convention: 252 trading days
ANNUALIZATION_FACTOR = 252

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
DATA_DIR    = "data"
