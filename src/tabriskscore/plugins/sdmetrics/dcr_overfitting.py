import json
from typing import Any, Dict, List

import pandas as pd
from sdmetrics.single_table.privacy import DCROverfittingProtection


def evaluate_privacy(
    data: Dict[str, Any], config: Dict[str, Any]
) -> List[Dict[str, Any]]:

    real_train = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/real_train.csv"
    )
    valid = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/real_valid.csv"
    )
    synth = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/synthetic.csv"
    )
    with open(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/metadata.json",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    score = DCROverfittingProtection.compute(
        real_data=real_train,
        validation_data=valid,
        synthetic_data=synth,
        metadata=metadata,
        num_rows_subsample=config.get("num_rows_subsample", 1000),
        k=config.get("k", 5),
    )

    return [
        {
            "name": "DCROverfittingProtection",
            "value": float(score),
            "details": {
                "num_rows_subsample": config.get("num_rows_subsample", 1000),
                "k": config.get("k", 5),
            },
        }
    ]
