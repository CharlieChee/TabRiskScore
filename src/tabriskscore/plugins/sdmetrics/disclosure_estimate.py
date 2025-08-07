import json
from typing import Any, Dict, List

import pandas as pd
from sdmetrics.single_table.privacy import DisclosureProtectionEstimate


def evaluate_privacy(
    data: Dict[str, Any], config: Dict[str, Any]
) -> List[Dict[str, Any]]:

    real = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/real.csv"
    )
    synth = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/synthetic.csv"
    )

    score = DisclosureProtectionEstimate.compute(
        real_data=real,
        synthetic_data=synth,
        known_column_names=config["known_cols"],
        sensitive_column_names=config["sensitive_cols"],
        num_rows_subsample=config.get("num_rows_subsample", 2500),
        num_iterations=config.get("num_iterations", 100),
        verbose=config.get("verbose", False),
    )

    return [
        {
            "name": "DisclosureProtectionEstimate",
            "value": float(score),
            "details": {
                "known_cols": config["known_cols"],
                "sensitive_cols": config["sensitive_cols"],
                "num_rows_subsample": config.get("num_rows_subsample", 2500),
                "num_iterations": config.get("num_iterations", 100),
            },
        }
    ]
