"""Train and compare chronological grid-injection baselines."""

from src.config import GOLD_DATA_DIR, REPORTS_DIR, configure_logging
from scripts.modeling_common import run_target_baselines

FEATURES = ["hour", "dayofweek", "month", "is_weekend", "total_injection_lag_1", "total_injection_lag_24", "total_injection_lag_168", "total_injection_rollmean_24", "total_injection_rollmean_168"]


def main():
    configure_logging()
    return run_target_baselines(GOLD_DATA_DIR / "gold_injection_hourly.parquet", "total_injection", FEATURES, REPORTS_DIR / "models/injection_model_comparison.csv", REPORTS_DIR / "models/injection_feature_importance.csv")


if __name__ == "__main__":
    main()
