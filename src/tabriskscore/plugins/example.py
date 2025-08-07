# plugins/example.py
from .hooks import evaluate_privacy


def evaluate_privacy(data, config):
    # 一个 demo 插件，总是输出 0.42
    return {"name": "example_metric", "value": 0.42}
