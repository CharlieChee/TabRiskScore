import json
from typing import Any, Dict, List

import pandas as pd
from sdmetrics.single_table.privacy import DisclosureProtection


def evaluate_privacy(
    data: Dict[str, Any], config: Dict[str, Any]
) -> List[Dict[str, Any]]:

    real = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/real.csv"
    )
    synth = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/synthetic.csv"
    )
    with open(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/metadata.json",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    score = DisclosureProtection.compute(
        real_data=real,
        synthetic_data=synth,
        metadata=metadata,
        known_column_names=config["known_cols"],
        sensitive_column_names=config["sensitive_cols"],
        num_rows_subsample=config.get("num_rows_subsample", 1000),
    )

    return [
        {
            "name": "DisclosureProtection",
            "value": float(score),
            "details": {
                "known_cols": config["known_cols"],
                "sensitive_cols": config["sensitive_cols"],
                "num_rows_subsample": config.get("num_rows_subsample", 1000),
            },
        }
    ]
