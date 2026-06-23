"""Run rolling-origin backtesting for hourly national consumption."""

from scripts.backtesting_common import run_dataset_backtest
from scripts.train_baseline_consumption import FEATURES
from src.config import GOLD_DATA_DIR, REPORTS_DIR, configure_logging


def main():
    configure_logging()
    return run_dataset_backtest(GOLD_DATA_DIR / "gold_consumption_hourly.parquet", "total", FEATURES, REPORTS_DIR / "backtesting", "consumption")


if __name__ == "__main__":
    main()
