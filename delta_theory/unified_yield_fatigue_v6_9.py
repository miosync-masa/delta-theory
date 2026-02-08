"""
Backward compatibility shim: unified_yield_fatigue_v6_9 → v10

他のモジュール（unified_flc_v8_1等）が
  from .unified_yield_fatigue_v6_9 import ...
で参照している場合の互換レイヤー。

v10.0 SSOC版の全エクスポートを再公開する。
"""
# v10.0 から全て再エクスポート
from .unified_yield_fatigue_v10 import *  # noqa: F401, F403
