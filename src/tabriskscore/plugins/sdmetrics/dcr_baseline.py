# src/tabriskscore/plugins/sdmetrics_privacy/dcr_baseline.py
import json
from typing import Any, Dict, List

import pandas as pd
from sdmetrics.single_table.privacy import DCRBaselineProtection


def evaluate_privacy(
    data: Dict[str, Any], config: Dict[str, Any]
) -> List[Dict[str, Any]]:

    real = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/real.csv"
    )  # 按需指定日期列
    synth = pd.read_csv(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/synthetic.csv"
    )
    with open(
        "/Users/changlong.ji/Desktop/project/TabRiskScore/data/sdmetrics/metadata.json",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    score = DCRBaselineProtection.compute(real, synth, metadata)  # 核心只这一行

    return [
        {
            "name": "DCRBaselineProtection",
            "value": score,
            "details": {
                # 你希望暴露给前端 / 报告的额外信息
                "num_rows_subsample": config.get("num_rows_subsample", 1000),
                "num_iterations": config.get("num_iterations", 1),
            },
        }
    ]
