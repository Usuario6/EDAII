"""Train and compare chronological consumption baselines."""

from src.config import GOLD_DATA_DIR, REPORTS_DIR, configure_logging
from scripts.modeling_common import run_target_baselines

FEATURES = ["hour", "dayofweek", "month", "is_weekend", "total_lag_1", "total_lag_24", "total_lag_168", "total_rollmean_24", "total_rollmean_168"]


def main():
    configure_logging()
    return run_target_baselines(GOLD_DATA_DIR / "gold_consumption_hourly.parquet", "total", FEATURES, REPORTS_DIR / "models/consumption_model_comparison.csv", REPORTS_DIR / "models/consumption_feature_importance.csv")


if __name__ == "__main__":
    main()
