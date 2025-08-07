# src/tabriskscore/plugins/lnb/lnb_plugin.py

import os
import time
from typing import Any, Dict, List

import nest_asyncio
import typer

from .data_prep import load_data, split_data
from .distance import compute_achilles, top_n_vulnerable_records
from .mia import mia

# 假设下面这些函数都在同一个目录下（lnb/），并且你已经把 LNB 里对应的源码拷贝过来了：
#   - load_data(path_to_csv, path_to_metadata) -> (DataFrame, categorical_cols, continuous_cols, meta_data)
#   - split_data(df, path_to_indices_pickle) -> (df_aux, df_eval, df_target)
#   - compute_achilles(df_target, categorical_cols, continuous_cols, meta_data, k) -> all_dists
#   - top_n_vulnerable_records(all_dists, n) -> List（排好序的最脆弱记录索引）
#   - mia(path_to_data, path_to_metadata, path_to_data_split, target_records, generator_name, n_original, n_synth, n_datasets, epsilon, output_path)
#
# 具体的实现文件可能放在 compute.py、data_prep.py、distance.py、mia.py 等，根据你的目录结构自行修改导入路径。


# 由于 LNB 某些内部可能用了 asyncio，需要先 apply
nest_asyncio.apply()


def evaluate_privacy(data: Dict[str, Any], config: Dict) -> List[Dict[str, Any]]:
    """
    LNB 演示版插件：
    - 假设项目根目录下有 data/adult/ 文件夹，里面放：
        • Adult_dataset.csv
        • Adult_metadata_discretized.json
        • 1000_indices.pickle
    - 先用 load_data + split_data 拆分数据
    - 然后 compute_achilles，取 top 100 脆弱记录
    - 最后对最脆弱记录列表执行一次 mia()
    - 返回三个指标：Achilles 计算用时、最脆弱记录数、MIA 用时

    参数：
        data:   框架传入的字典（key->DataFrame）。本插件不直接使用它，
                而是从硬编码的文件路径里加载“Adult”数据。
        config: 框架上下文，用户可在 config 中额外传入 LNB 参数，
                比如 {"adult_dir": "data/adult", "k": 5, "top_n": 100, "mia_params": {...}}
    返回：
        List[Dict[str, Any]]：每个 dict 包含 {"name": str, "value": 任意, "details": dict}
    """
    # 如果用户在 config 中提供了自定义的 data 目录，优先用它；否则用默认
    adult_dir = config.get("adult_dir", "data/adult")
    # LNB 参数：k（计算 Achilles 时的 KNN 半径），默认为 5
    k = config.get("k", 5)
    # MIA 参数（除了路径外的参数可以放在这里）
    mia_params = config.get("mia_params", {})

    # 构造相关文件路径
    path_to_data = os.path.join(adult_dir, "Adult_dataset.csv")
    path_to_metadata = os.path.join(adult_dir, "Adult_metadata_discretized.json")
    path_to_indices = os.path.join(adult_dir, "1000_indices.pickle")

    # 先检查这些文件是否存在
    missing = []
    for path in (path_to_data, path_to_metadata, path_to_indices):
        if not os.path.isfile(path):
            missing.append(path)
    if missing:
        typer.secho(f"❌ LNB 插件找不到必要文件：{missing}", fg=typer.colors.RED)
        return []

    metrics: List[Dict[str, Any]] = []

    # ——— 步骤 1：读取并拆分数据 ———
    try:
        df, categorical_cols, continuous_cols, meta_data = load_data(
            path_to_data, path_to_metadata
        )
        # split_data 会根据 pickle 中存好的 1000 条索引，把 df 划分成三个部分
        df_aux, df_eval, df_target = split_data(df, path_to_indices)
    except Exception as e:
        typer.secho(
            f"❌ LNB 插件在 load_data / split_data 环节出错：{e}", fg=typer.colors.RED
        )
        return [
            {
                "name": "lnb_status",
                "value": None,
                "details": {"error": f"load_data/split_data 失败: {e}"},
            }
        ]

    # ——— 步骤 2：计算 Achilles scores ———
    try:
        t0 = time.time()
        # compute_achilles 返回一个距离矩阵或字典，表示每个 target-record 到 training set 的 K 最近距离
        all_dists = compute_achilles(
            df_target, categorical_cols, continuous_cols, meta_data, k
        )
        t1 = time.time()
        achilles_time = t1 - t0
    except Exception as e:
        typer.secho(
            f"❌ LNB 插件在 compute_achilles 环节出错：{e}", fg=typer.colors.RED
        )
        return [
            {
                "name": "lnb_status",
                "value": None,
                "details": {"error": f"compute_achilles 失败: {e}"},
            }
        ]

    # 选出前 top_n 脆弱记录，默认 top_n=100
    top_n = config.get("top_n", 100)
    try:
        top_n_records = top_n_vulnerable_records(all_dists, top_n)
        num_vulnerable = len(top_n_records)
    except Exception as e:
        typer.secho(
            f"❌ LNB 插件在 top_n_vulnerable_records 环节出错：{e}", fg=typer.colors.RED
        )
        num_vulnerable = None

    # 把 Achilles 用时和脆弱记录数量加入 metrics
    metrics.append(
        {"name": "achilles_time", "value": achilles_time, "details": {"k": k}}
    )
    metrics.append(
        {
            "name": "num_vulnerable_records",
            "value": num_vulnerable,
            "details": {"top_n": top_n},
        }
    )

    # ——— 步骤 3：对最脆弱记录运行一次 MIA ———
    try:
        t2 = time.time()
        mia_results = mia(
            path_to_data=path_to_data,
            path_to_metadata=path_to_metadata,
            path_to_data_split=path_to_indices,
            target_records=top_n_records[0:2],  # 这里只示范第 1 条脆弱记录
            generator_name=mia_params.get("generator_name", "SYNTHPOP"),
            n_synth=mia_params.get("n_synth", 1000),
            n_datasets=mia_params.get("n_datasets", 10),
            epsilon=mia_params.get("epsilon", 0.0),
            output_path=mia_params.get("output_path", "./output/files/"),
        )

        t3 = time.time()
        mia_time = t3 - t2
    except Exception as e:
        typer.secho(f"❌ LNB 插件在 MIA 环节出错：{e}", fg=typer.colors.RED)
        mia_time = None

    metrics.append(
        {
            "name": "mia_result",
            "value": mia_results,
            "details": {
                "generator_name": mia_params.get("generator_name", "SYNTHPOP"),
                "n_original": mia_params.get("n_original", 1000),
                "n_synth": mia_params.get("n_synth", 1000),
                "n_datasets": mia_params.get("n_datasets", 10),
                "epsilon": mia_params.get("epsilon", 0.0),
                "mia_time": mia_time,
            },
        }
    )

    return metrics
