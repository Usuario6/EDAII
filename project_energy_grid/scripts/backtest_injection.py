"""Run rolling-origin backtesting for hourly distribution-grid injection."""

from scripts.backtesting_common import run_dataset_backtest
from scripts.train_baseline_injection import FEATURES
from src.config import GOLD_DATA_DIR, REPORTS_DIR, configure_logging


def main():
    configure_logging()
    return run_dataset_backtest(GOLD_DATA_DIR / "gold_injection_hourly.parquet", "total_injection", FEATURES, REPORTS_DIR / "backtesting", "injection")


if __name__ == "__main__":
    main()
