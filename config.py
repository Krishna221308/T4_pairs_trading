# config.py

# Universe of pairs
UNIVERSE = [("GLD", "GDX"), ("KO", "PEP"), ("XOM", "CVX"), ("JPM", "BAC"), ("MSFT", "AAPL"), ("HD", "LOW"), ("WMT", "COST"), ("UNH", "CI"), ("V", "MA"), ("PG", "CL")]

# Backtest Timeframe
START_DATE = "2019-01-01"
END_DATE = "2024-06-01"

# Cointegration significance threshold
COINT_ALPHA = 0.05

# Half life bounds for pairs screening
HALF_LIFE_MIN = 5
HALF_LIFE_MAX = 60

# Kalman filter noise params 
KALMAN_Q = 1e-4             # Process noise (how fast beta drifts)
KALMAN_R = 1e-3             # Measurement noise (price observation noise)

# Z-Score signal params
ZSCORE_WINDOW = 60     # Window for z-score normalisation
ENTRY_Z = 2.0          # enter when |z| >= this
EXIT_Z = 00.5          # exit when |z| <= this

# Walk forward params
TRAIN_FRAC = 0.70       #70% train and 30% test (chronologically)
REFIT_INTERVAL = 60     # Re-fit ML model every N bars
LOOKBACK_COINT = 120    # To check for cointegration breakdown

# TCA params
SLIPPAGE_BPS = 5        # Half-spread in basis points
IMPACT_GAMMA = 0.5      # Market impact scaling constant

# Model persistance
MODEL_OUTPUT_DIR = "output/models"