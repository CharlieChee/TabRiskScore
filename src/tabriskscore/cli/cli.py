# src/tabriskscore/cli/cli.py


import os

import typer
import yaml

from tabriskscore.orchestrator.flow import privacy_flow

app = typer.Typer(help="TabRiskScore 隐私度量 CLI")


@app.command("evaluate")
def evaluate(
    config_file: str = typer.Option(
        "config.yaml", "--config", "-c", help="配置文件路径"
    ),
    output_file: str = typer.Option(
        "report.html", "-o", "--output", help="报告输出路径（.html 或 .pdf）"
    ),
):
    """
    自动读取根目录下的 data/ 文件夹里的 original.csv、synthetic.csv，
    存在就执行对应算法，最后生成报告。
    """
    # 确保在项目根目录下运行
    project_root = os.getcwd()
    data_dir = os.path.join(project_root, "data")
    if not os.path.isdir(data_dir):
        typer.secho(f"❌ data 文件夹不存在：{data_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)

    privacy_flow(data_dir, output_file)
    typer.secho(f"✅ 报告生成完成：{output_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
