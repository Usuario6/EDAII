"""Run direct multi-step experiments for distribution-grid injection."""

from scripts.multistep_common import run_multistep_dataset
from src.config import GOLD_DATA_DIR, REPORTS_DIR, configure_logging


def main():
    configure_logging()
    return run_multistep_dataset(GOLD_DATA_DIR / "gold_injection_enriched.parquet", "total_injection", "injection", REPORTS_DIR / "multistep")


if __name__ == "__main__":
    main()
