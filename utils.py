"""
Agent-NEE FinAI — Utilities
Logging, custom exceptions, retry decorators.
"""

import logging
import time
import functools
from logging.handlers import TimedRotatingFileHandler

import config


# ─── Custom Exceptions ───────────────────────────────────────────────

class BIFASError(Exception):
    """Base exception for Agent-NEE."""
    pass

class ConfigError(BIFASError):
    """Configuration or environment variable error."""
    pass

class DataSourceError(BIFASError):
    """Data source connection or data error."""
    pass

class OllamaError(BIFASError):
    """Ollama API connection or inference error."""
    pass

class LedgerError(BIFASError):
    """Parquet ledger read/write error."""
    pass

class TrainingError(BIFASError):
    """LoRA SFT training error."""
    pass

class AgentError(BIFASError):
    """Multi-agent squad execution error."""
    pass


# ─── Logging Setup ───────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure file + console logging with daily rotation."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("agent_nee")
    logger.setLevel(logging.DEBUG)

    # File handler — daily rotation, 30 days retention
    file_handler = TimedRotatingFileHandler(
        config.LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=config.LOG_MAX_DAYS,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    # Console handler — INFO level, simplified
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)s: %(message)s"
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ─── Retry Decorator ─────────────────────────────────────────────────

def retry_with_backoff(max_retries=3, base_delay=1, backoff_factor=2, max_delay=60):
    """Exponential backoff retry decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(min(delay, max_delay))
                        delay *= backoff_factor
            raise last_exception
        return wrapper
    return decorator
