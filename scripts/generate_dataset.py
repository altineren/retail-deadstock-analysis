"""Generate processed dataset with engineered features and labels."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is on path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_pipeline import PreparedData, prepare_dataset


RAW_DATA_PATH = Path("retail_store_inventory.csv")
OUTPUT_PATH = Path("data/processed_inventory.csv")


def main() -> None:
    prepared: PreparedData = prepare_dataset(RAW_DATA_PATH)
    output_df = prepared.raw
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved processed dataset with {len(output_df)} rows to {OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()
