# src/tabriskscore/plugins/row_count.py

from typing import Any, Dict


def evaluate_privacy(data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
    """
    统计每个可用文件（如 original、synthetic）的行数，
    并把结果写入 details，返回一个统一的 row_count 指标。
    """
    counts: Dict[str, int] = {}
    for name, df in data.items():
        try:
            counts[name] = df.shape[0]
        except Exception:
            counts[name] = None

    return {
        "name": "row_count",
        "value": None,  # 主值留空，用 details 展示
        "details": counts,  # {"original": 123, "synthetic": 456}
    }
