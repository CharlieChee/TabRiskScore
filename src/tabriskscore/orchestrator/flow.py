# src/tabriskscore/orchestrator/flow.py

import importlib
import os
import pkgutil
import sys
import types

import typer

from tabriskscore.adapters.csv import load_csv
from tabriskscore.reports.render import render_html


def discover_plugins():
    """
    自动发现并 import 所有 plugins 目录下除了 hooks.py 以外的模块
    """
    import tabriskscore.plugins as plug_pkg

    for finder, name, _ in pkgutil.iter_modules(plug_pkg.__path__):
        if name == "hooks":
            continue
        importlib.import_module(f"tabriskscore.plugins.{name}")


def run_all(data_dict: dict[str, any], config: dict = None) -> list[dict]:
    """
    遍历已加载的插件模块，调用它们的 evaluate_privacy，
    支持插件返回单个 dict 或者 list[dict]，统一扁平化到一个列表。
    """
    config = config or {}
    results: list[dict] = []

    for mod in list(sys.modules.values()):
        if (
            isinstance(mod, types.ModuleType)
            and mod.__name__.startswith("tabriskscore.plugins.")
            and mod.__name__ != "tabriskscore.plugins.hooks"
            and hasattr(mod, "evaluate_privacy")
        ):
            try:
                # 这里是关键！！
                res = mod.evaluate_privacy(data_dict, config)
                if isinstance(res, list):
                    results.extend(res)
                elif isinstance(res, dict):
                    results.append(res)
                else:
                    typer.secho(
                        f"⚠️ 插件 {mod.__name__} 返回类型不支持: {type(res)}，已跳过",
                        fg=typer.colors.YELLOW,
                    )
            except Exception as e:
                typer.secho(
                    f"⚠️ 插件 {mod.__name__} 执行失败：{e}", fg=typer.colors.YELLOW
                )
    return results


def privacy_flow(data_dir: str, output_path: str):
    """
    1) 扫描 data_dir 下所有 .csv
    2) 加载成 pandas.DataFrame 存入 data_dict
    3) 发现并调用所有插件
    4) 一律渲染 HTML 报告
    """
    # 1) 加载 CSV
    data_dict: dict[str, any] = {}
    if not os.path.isdir(data_dir):
        typer.secho(f"❌ data 目录不存在：{data_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".csv"):
            continue
        key = os.path.splitext(fname)[0]
        path = os.path.join(data_dir, fname)
        try:
            data_dict[key] = load_csv(path)
            typer.secho(
                f"ℹ️ 读取文件 {fname} → key='{key}'，{data_dict[key].shape[0]} 行",
                fg=typer.colors.BLUE,
            )
        except Exception as e:
            typer.secho(f"❌ 读取 {fname} 失败：{e}", fg=typer.colors.RED)

    if not data_dict:
        typer.secho("❌ data 目录下没有可用的 CSV 文件，退出。", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 2) 发现 & 加载插件
    discover_plugins()

    # 3) 执行所有度量
    # 这里的data_dict 是个字典，里面有3个key, 分别为文件名Xtrain，Xsyn，Xcontrol
    metrics = run_all(data_dict)

    # 4) 渲染 HTML
    if not output_path.lower().endswith(".html"):
        output_path = f"{output_path}.html"
    render_html(metrics, output_path)
