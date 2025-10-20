# utils.py
# 工具函数模块

import gc
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log(message, config=None):
    """增强的日志函数"""
    logging.info(message)
    
    # 可选：保存到日志文件
    if config and hasattr(config, 'results_dir'):
        log_file = config.results_dir / "analysis.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [MVPA] {message}\n")

def memory_cleanup():
    """内存清理"""
    gc.collect()