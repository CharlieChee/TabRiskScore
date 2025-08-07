# src/tabriskscore/plugins/similar_check.py

from typing import Any, Dict

import pandas as pd


def evaluate_privacy(data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
    """
    如果 data 中同时包含 'Xsyn' 和 'Xtrain' 两个 DataFrame，
    则统计它们**完全相同**的行有多少条，返回相应指标；否则跳过。
    """
    df_syn = data.get("Xsyn")
    df_train = data.get("Xtrain")

    # 只有两个表都存在时才执行
    if df_syn is None or df_train is None:
        return None

    # 方法：把每行当成一个 tuple，用集合交集来算完全相同的行数
    try:
        set_syn = set(map(tuple, df_syn.values))
        set_train = set(map(tuple, df_train.values))
        common = set_syn.intersection(set_train)
        count = len(common)
    except Exception:
        # 如果中途失败，也不要整个流程挂掉，返回一个标记
        return {
            "name": "similar_check",
            "value": None,
            "details": {"error": "无法比较行内容"},
        }

    return {"name": "similar_check", "value": count, "details": {"common_rows": count}}
