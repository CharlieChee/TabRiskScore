from typing import Any, Dict, List

import pandas as pd
from sdmetrics.single_table.privacy import CategoricalCAP


def evaluate_privacy(
    data: Dict[str, Any], config: Dict[str, Any]
) -> List[Dict[str, Any]]:

    real = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/real.csv"
    )
    synth = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/synthetic.csv"
    )

    score = CategoricalCAP.compute(
        real_data=real,
        synthetic_data=synth,
        key_fields=config["key_fields"],  # 攻击者已知列
        sensitive_fields=config["sensitive_fields"],  # 要推断的列
        variant=config.get("variant", "cap"),  # cap / zero_cap / generalized_cap
    )

    return [
        {
            "name": "CategoricalCAP",
            "value": float(score),
            "details": {
                "key_fields": config["key_fields"],
                "sensitive_fields": config["sensitive_fields"],
                "variant": config.get("variant", "cap"),
            },
        }
    ]
