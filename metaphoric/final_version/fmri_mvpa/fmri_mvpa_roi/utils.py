"""
utils.py

用途
- ROI-MVPA 模块的小工具集合（目前主要是统一的 log 输出）。
"""

from __future__ import annotations

import logging


def log(message, config=None):
    logging.info(message)
    print(message)
