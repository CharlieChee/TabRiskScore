# plugins/hooks.py
from typing import Any, Dict

# 使用最简单的 callable 接口，未来可换成 pluggy
PrivacyResult = Dict[str, Any]


def evaluate_privacy(data: Any, config: Dict) -> PrivacyResult:
    """所有插件都要实现这个函数签名"""
    raise NotImplementedError
