"""
Agent-NEE FinAI — Configuration
All constants in one place. Imported by every module.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
DAILY_DIR = DATA_DIR / "daily"
REPLAY_DIR = DATA_DIR / "replay_buffer"
MODELS_DIR = BASE_DIR / "models" / "adapters"
CHARTS_DIR = BASE_DIR / "charts"
STATIC_BASELINE_PATH = REPLAY_DIR / "static_baseline.parquet"

# ─── Tickers ─────────────────────────────────────────────────────────
TICKER_SYMBOLS = [
    "NSE:RELIANCE", "NSE:TCS", "NSE:HDFCBANK", "NSE:INFY", "NSE:ICICIBANK",
    "NSE:SBIN", "NSE:BHARTIARTL", "NSE:ITC", "NSE:WIPRO", "NSE:AXISBANK",
]

# ─── CSV Simulation ──────────────────────────────────────────────────
CSV_ROWS_PER_CYCLE = 1
SIMULATION_LOOP = True
SIMULATION_SPEED = 1.0

# Base prices (used by generate_csv.py)
MOCK_BASE_PRICES = {
    "NSE:RELIANCE": 2845.0,
    "NSE:TCS": 3920.0,
    "NSE:HDFCBANK": 1650.0,
    "NSE:INFY": 1480.0,
    "NSE:ICICIBANK": 1120.0,
    "NSE:SBIN": 780.0,
    "NSE:BHARTIARTL": 1250.0,
    "NSE:ITC": 480.0,
    "NSE:WIPRO": 510.0,
    "NSE:AXISBANK": 1080.0,
}

# ─── Ollama ──────────────────────────────────────────────────────────
MODEL_NAME = "phi3:mini"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_TAGS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_TIMEOUT = 30

# ─── Inference Parameters ────────────────────────────────────────────
INFERENCE_TEMPERATURE = 0.3
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 64
N_CTX = 4096
AGENT_MAX_TOKENS = 512
SYNTH_MAX_TOKENS = 64
SYNTHESIS_TEMPERATURE = 0.1

# ─── Time Windows (IST) ─────────────────────────────────────────────
PREDICTION_START = "09:25"
PREDICTION_END = "15:15"
FRIDAY_EARLY_CLOSE = "15:00"
PREMARKET_WARMUP_TIME = "09:15"
INTERVAL_MINUTES = 5
INTERVAL_SECONDS = INTERVAL_MINUTES * 60

# ─── Indicators ──────────────────────────────────────────────────────
CANDLE_HISTORY_DAYS = 5
MIN_INDICATOR_ROWS = 30
SIDEWAYS_THRESHOLD_PCT = 0.25

# ─── Hardware ────────────────────────────────────────────────────────
CPU_TICKER_LIMIT = 3
GPU_TICKER_LIMIT = 10
TRAINING_VRAM_MIN_GB = 12
TRAINING_CUDA_MIN_VERSION = "12.1"

# ─── Training (LoRA SFT) ────────────────────────────────────────────
MIN_TRAINING_ROWS = 20
REPLAY_BUFFER_DAYS = 30
VALIDATION_SPLIT = 0.1
ROLLBACK_IF_VAL_LOSS_INCREASES = True
MAX_ADAPTERS_TO_KEEP = 10
STATIC_BASELINE_SPLIT = 0.3

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

SFT_LEARNING_RATE = 2e-5
SFT_NUM_EPOCHS = 3
SFT_PER_DEVICE_BATCH_SIZE = 2
SFT_GRADIENT_ACCUMULATION_STEPS = 4
SFT_EARLY_STOPPING_PATIENCE = 3
SFT_MAX_SEQ_LENGTH = 512

# ─── Agents ──────────────────────────────────────────────────────────
N_AGENTS = 3
AGENT_ROLES_TO_RUN = ["technical_analyst", "volatility_analyst", "volume_analyst"]
BAND_ENABLED = True

# ─── Web Server ──────────────────────────────────────────────────────
WEB_HOST = "127.0.0.1"
WEB_PORT = 8080
WEB_AUTO_OPEN_BROWSER = True
WEB_REFRESH_INTERVAL = 1.0

# ─── Candlestick ─────────────────────────────────────────────────────
CANDLESTICK_HISTORY_CANDLES = 100
CANDLESTICK_UP_COLOR = "#00ff41"
CANDLESTICK_DOWN_COLOR = "#ff3355"
CANDLESTICK_VOLUME_HEIGHT = 0.2

# ─── Logging ─────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "bifas_nexus.log"
LOG_MAX_DAYS = 30
