"""
Agent-NEE FinAI — LoRA SFT Training Module
Standalone subprocess for supervised fine-tuning on correct predictions.
Runs as: python learn.py --date YYYY-MM-DD

Training dependencies (torch, transformers, peft) are ONLY imported here.
"""

import sys
import os
import json
import logging
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
import utils

logger = logging.getLogger("agent_nee")


def check_training_deps() -> bool:
    """Check if training dependencies are available."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Training requires NVIDIA GPU.")
            return False
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        if vram_gb < config.TRAINING_VRAM_MIN_GB:
            logger.warning(f"VRAM {vram_gb:.1f} GB < {config.TRAINING_VRAM_MIN_GB} GB required")
            return False
        cuda_version = torch.version.cuda
        if cuda_version and cuda_version < config.TRAINING_CUDA_MIN_VERSION:
            logger.warning(f"CUDA {cuda_version} < {config.TRAINING_CUDA_MIN_VERSION} required")
            return False
        return True
    except ImportError:
        logger.warning("Training deps not installed. Run: pip install torch transformers peft")
        return False


def load_training_data() -> list[dict]:
    """Load replay buffer and filter for training-eligible rows."""
    import pandas as pd

    buffer_path = config.REPLAY_DIR / "replay_buffer.parquet"
    if not buffer_path.exists():
        logger.warning("No replay buffer found")
        return []

    df = pd.read_parquet(buffer_path)

    # Filter: resolved, correct predictions, not yet used
    eligible = df[
        (df["prediction_accuracy"].eq(True)) &
        (df["used_in_training"].ne(True)) &
        (df["actual_direction"].notna())
    ]

    if len(eligible) < config.MIN_TRAINING_ROWS:
        logger.info(f"Insufficient rows: {len(eligible)} < {config.MIN_TRAINING_ROWS}")
        return []

    return eligible.to_dict("records")


def format_training_pair(row: dict) -> dict:
    """Format a resolved row as prompt-completion pair for SFT."""
    prompt = f"""Ticker: {row.get('ticker', 'NSE:UNKNOWN')}
Price: O={row.get('open', 0):.2f} H={row.get('high', 0):.2f} L={row.get('low', 0):.2f} C={row.get('close', 0):.2f}
Volume: {row.get('volume', 0):.0f}
VWAP: {row.get('vwap', 0):.2f}
RSI(14): {row.get('rsi_14', 0):.2f}
MACD: {row.get('macd', 0):.2f} (signal: {row.get('macd_signal', 0):.2f})
Bollinger Bands: {row.get('bb_lower', 0):.2f} - {row.get('bb_middle', 0):.2f} - {row.get('bb_upper', 0):.2f}
ATR(14): {row.get('atr_14', 0):.2f}

Predict direction, return percentage, and confidence."""

    completion = json.dumps({
        "direction": row.get("actual_direction", "SIDEWAYS"),
        "target_return_pct": round(row.get("actual_return_pct", 0.0), 2),
        "confidence": row.get("prediction_confidence", "MED"),
    })

    return {"prompt": prompt, "completion": completion, "row_id": row.get("row_id", "")}


def balance_classes(rows: list[dict]) -> list[dict]:
    """Balance UP/DOWN/SIDEWAYS distribution."""
    import random

    by_class = {"UP": [], "DOWN": [], "SIDEWAYS": []}
    for row in rows:
        direction = row.get("actual_direction", "SIDEWAYS")
        if direction in by_class:
            by_class[direction].append(row)

    # Filter out empty classes
    non_empty = [v for v in by_class.values() if v]
    if not non_empty:
        return []

    min_count = min(len(v) for v in non_empty)
    if min_count == 0:
        return []

    balanced = []
    for cls_rows in by_class.values():
        if cls_rows:
            balanced.extend(random.sample(cls_rows, min(min_count, len(cls_rows))))

    random.shuffle(balanced)
    return balanced


def get_next_version() -> int:
    """Get next adapter version number."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in config.MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not existing:
        return 1
    versions = [int(v[1:]) for v in existing if v[1:].isdigit()]
    return max(versions) + 1 if versions else 1


def get_previous_val_loss() -> float:
    """Load previous validation loss for rollback comparison."""
    loss_file = config.MODELS_DIR / "last_val_loss.txt"
    if loss_file.exists():
        try:
            return float(loss_file.read_text().strip())
        except (ValueError, OSError):
            pass
    return float("inf")


def save_val_loss(loss: float):
    """Save current validation loss."""
    loss_file = config.MODELS_DIR / "last_val_loss.txt"
    loss_file.write_text(str(loss))


def update_version_files(version_dir: Path):
    """Update version tracking files."""
    current_file = config.MODELS_DIR / "current_version.txt"
    previous_file = config.MODELS_DIR / "previous_version.txt"
    latest_file = config.MODELS_DIR / "latest.txt"

    if current_file.exists():
        previous_file.write_text(current_file.read_text().strip())

    current_file.write_text(str(version_dir))
    latest_file.write_text(str(version_dir))

    cleanup_old_adapters()


def cleanup_old_adapters():
    """Keep only MAX_ADAPTERS_TO_KEEP most recent adapters."""
    adapters = sorted(
        [d for d in config.MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0
    )
    while len(adapters) > config.MAX_ADAPTERS_TO_KEEP:
        oldest = adapters.pop(0)
        shutil.rmtree(oldest)
        logger.info(f"Removed old adapter: {oldest.name}")


def train(pairs: list[dict], date_str: str) -> bool:
    """Run LoRA SFT training on the given pairs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    logger.info(f"Starting LoRA SFT training with {len(pairs)} pairs")

    model_name = "microsoft/Phi-3-mini-4k-instruct"
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",

    )

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(examples):
        full_text = [p + "\n" + c for p, c in zip(examples["prompt"], examples["completion"])]
        tokens = tokenizer(full_text, truncation=True, max_length=config.SFT_MAX_SEQ_LENGTH, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(pairs)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt", "completion", "row_id"])

    split = tokenized.train_test_split(test_size=config.VALIDATION_SPLIT, seed=42)

    version_dir = config.MODELS_DIR / f"v{get_next_version()}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Load previous val loss for rollback
    prev_val_loss = get_previous_val_loss()

    # Training arguments
    batch_size = config.SFT_PER_DEVICE_BATCH_SIZE
    grad_accum = config.SFT_GRADIENT_ACCUMULATION_STEPS

    def make_trainer(bs, ga):
        args = TrainingArguments(
            output_dir=str(version_dir),
            num_train_epochs=config.SFT_NUM_EPOCHS,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            learning_rate=config.SFT_LEARNING_RATE,
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        )
        return Trainer(model=model, args=args,
                      train_dataset=split["train"], eval_dataset=split["test"])

    # Train with OOM recovery
    trainer = make_trainer(batch_size, grad_accum)
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("CUDA OOM. Retrying with batch_size=1...")
            torch.cuda.empty_cache()
            trainer = make_trainer(1, grad_accum * 2)
            try:
                trainer.train()
            except RuntimeError as e2:
                logger.error(f"Training failed even at batch_size=1: {e2}")
                shutil.rmtree(version_dir, ignore_errors=True)
                return False
        else:
            shutil.rmtree(version_dir, ignore_errors=True)
            raise

    # Evaluate
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss", float("inf"))
    logger.info(f"Validation loss: {val_loss:.4f} (previous: {prev_val_loss:.4f})")

    # Validation rollback check
    if config.ROLLBACK_IF_VAL_LOSS_INCREASES and val_loss > prev_val_loss:
        logger.warning(f"Val loss increased ({val_loss:.4f} > {prev_val_loss:.4f}). Rolling back.")
        shutil.rmtree(version_dir, ignore_errors=True)
        # Keep the previous adapter
        return True  # Not a failure, just no improvement

    # Save adapter (only if no rollback)
    model.save_pretrained(str(version_dir))
    tokenizer.save_pretrained(str(version_dir))
    save_val_loss(val_loss)
    update_version_files(version_dir)

    logger.info(f"Training complete. Adapter saved to {version_dir}")
    return True


def mark_rows_trained(pairs: list[dict]):
    """Mark only the specific rows used in training as used_in_training=True."""
    import pandas as pd

    row_ids = {p["row_id"] for p in pairs if p.get("row_id")}
    if not row_ids:
        return

    buffer_path = config.REPLAY_DIR / "replay_buffer.parquet"
    if not buffer_path.exists():
        return

    df = pd.read_parquet(buffer_path)
    mask = df["row_id"].isin(row_ids)
    df.loc[mask, "used_in_training"] = True
    df.to_parquet(buffer_path, index=False)
    logger.info(f"Marked {mask.sum()} rows as used_in_training")


def main():
    parser = argparse.ArgumentParser(description="Agent-NEE LoRA SFT Training")
    parser.add_argument("--date", default=None, help="Training date (YYYY-MM-DD)")
    args = parser.parse_args()

    utils.setup_logging()
    date_str = args.date or "today"
    logger.info(f"Training run for {date_str}")

    if not check_training_deps():
        logger.warning("Training deps check failed. Exiting.")
        sys.exit(0)

    rows = load_training_data()
    if not rows:
        logger.info("No training data available. Exiting.")
        sys.exit(0)

    balanced = balance_classes(rows)
    logger.info(f"Training data: {len(balanced)} balanced rows")

    pairs = [format_training_pair(row) for row in balanced]

    success = train(pairs, date_str)

    if success:
        mark_rows_trained(pairs)
        logger.info("Training pipeline complete")
    else:
        logger.warning("Training pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
