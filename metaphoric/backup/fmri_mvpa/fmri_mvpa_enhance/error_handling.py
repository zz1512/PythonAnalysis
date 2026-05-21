#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理和容错性模块
提供统一的错误处理、日志记录和恢复机制
"""

import os
import sys
import traceback
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, Dict, List
import warnings

class MVPAError(Exception):
    """MVPA分析基础异常类"""
    pass

class DataLoadError(MVPAError):
    """数据加载异常"""
    pass

class ConfigError(MVPAError):
    """配置异常"""
    pass

class ProcessingError(MVPAError):
    """处理异常"""
    pass

class QualityError(MVPAError):
    """质量控制异常"""
    pass

class AnalysisError(MVPAError):
    """分析异常"""
    pass

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, log_dir: str = "./logs", debug_mode: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_mode = debug_mode
        
        # 设置日志
        self.setup_logging()
        
        # 错误统计
        self.error_counts = {
            'data_load': 0,
            'config': 0,
            'processing': 0,
            'quality': 0,
            'analysis': 0,
            'unknown': 0
        }
        
        # 恢复点存储
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """设置日志系统"""
        # 创建logger
        self.logger = logging.getLogger('MVPA_Enhanced')
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        
        # 清除现有handlers
        self.logger.handlers.clear()
        
        # 文件handler
        log_file = self.log_dir / f"mvpa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"日志系统已初始化，日志文件: {log_file}")
    
    def log_error(self, error: Exception, context: str = "", error_type: str = "unknown"):
        """记录错误"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_class': error.__class__.__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc() if self.debug_mode else None
        }
        
        # 记录到日志
        self.logger.error(f"[{error_type.upper()}] {context}: {error}")
        if self.debug_mode:
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # 保存错误详情
        error_file = self.log_dir / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        
        return error_info
    
    def save_checkpoint(self, data: Any, checkpoint_name: str, step: str = ""):
        """保存检查点"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.pkl"
            
            checkpoint_data = {
                'timestamp': timestamp,
                'step': step,
                'data': data
            }
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"检查点已保存: {checkpoint_file}")
            return checkpoint_file
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Any]:
        """加载检查点"""
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.logger.info(f"检查点已加载: {checkpoint_file}")
            return checkpoint_data['data']
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return None
    
    def list_checkpoints(self, checkpoint_name: str = "") -> List[str]:
        """列出可用的检查点"""
        pattern = f"{checkpoint_name}_*.pkl" if checkpoint_name else "*.pkl"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        return sorted([str(cp) for cp in checkpoints])
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'error_breakdown': self.error_counts.copy(),
            'error_rate': {k: v/total_errors if total_errors > 0 else 0 
                          for k, v in self.error_counts.items()}
        }

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    exceptions: tuple = (Exception,)):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"第{attempt + 1}次尝试失败，{delay}秒后重试: {e}")
                        import time
                        time.sleep(delay)
                    else:
                        print(f"所有重试均失败，抛出异常")
                        raise last_exception
            
            raise last_exception
        return wrapper
    return decorator

def safe_execute(func: Callable, error_handler: ErrorHandler, 
                context: str = "", error_type: str = "unknown",
                default_return: Any = None, raise_on_error: bool = False):
    """安全执行函数"""
    try:
        return func()
    except Exception as e:
        error_handler.log_error(e, context, error_type)
        if raise_on_error:
            raise
        return default_return

def validate_data_quality(data: Any, checks: Dict[str, Callable], 
                         error_handler: ErrorHandler) -> bool:
    """数据质量验证"""
    passed_checks = []
    failed_checks = []
    
    for check_name, check_func in checks.items():
        try:
            result = check_func(data)
            if result:
                passed_checks.append(check_name)
            else:
                failed_checks.append(check_name)
                error_handler.logger.warning(f"质量检查失败: {check_name}")
        except Exception as e:
            failed_checks.append(check_name)
            error_handler.log_error(e, f"质量检查: {check_name}", "quality")
    
    error_handler.logger.info(f"质量检查结果: 通过 {len(passed_checks)}, 失败 {len(failed_checks)}")
    
    return len(failed_checks) == 0

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int, error_handler: ErrorHandler):
        self.total_steps = total_steps
        self.current_step = 0
        self.error_handler = error_handler
        self.start_time = datetime.now()
        self.step_times = []
        
    def update(self, step_name: str = "", save_checkpoint: bool = False, 
              checkpoint_data: Any = None):
        """更新进度"""
        self.current_step += 1
        current_time = datetime.now()
        self.step_times.append(current_time)
        
        # 计算进度
        progress = (self.current_step / self.total_steps) * 100
        elapsed = current_time - self.start_time
        
        # 估算剩余时间
        if self.current_step > 1:
            avg_step_time = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = avg_step_time * remaining_steps
            eta_str = str(eta).split('.')[0]  # 去掉微秒
        else:
            eta_str = "未知"
        
        # 记录进度
        progress_msg = f"进度: {self.current_step}/{self.total_steps} ({progress:.1f}%) - {step_name} - ETA: {eta_str}"
        self.error_handler.logger.info(progress_msg)
        
        # 保存检查点
        if save_checkpoint and checkpoint_data is not None:
            checkpoint_name = f"step_{self.current_step:03d}_{step_name.replace(' ', '_')}"
            self.error_handler.save_checkpoint(checkpoint_data, checkpoint_name, step_name)
    
    def is_complete(self) -> bool:
        """检查是否完成"""
        return self.current_step >= self.total_steps
    
    def get_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        elapsed = datetime.now() - self.start_time
        return {
            'total_steps': self.total_steps,
            'completed_steps': self.current_step,
            'progress_percent': (self.current_step / self.total_steps) * 100,
            'elapsed_time': str(elapsed).split('.')[0],
            'is_complete': self.is_complete()
        }

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def check_file_exists(filepath: str) -> bool:
        """检查文件是否存在"""
        return os.path.exists(filepath)
    
    @staticmethod
    def check_file_size(filepath: str, min_size_mb: float = 0.1) -> bool:
        """检查文件大小"""
        if not os.path.exists(filepath):
            return False
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        return size_mb >= min_size_mb
    
    @staticmethod
    def check_nifti_integrity(filepath: str) -> bool:
        """检查NIfTI文件完整性"""
        try:
            import nibabel as nib
            img = nib.load(filepath)
            data = img.get_fdata()
            return data.size > 0 and not np.all(np.isnan(data))
        except Exception:
            return False
    
    @staticmethod
    def check_csv_integrity(filepath: str, required_columns: List[str] = None) -> bool:
        """检查CSV文件完整性"""
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            if required_columns:
                return all(col in df.columns for col in required_columns)
            return len(df) > 0
        except Exception:
            return False
    
    @staticmethod
    def check_memory_usage(max_usage_gb: float = 16.0) -> bool:
        """检查内存使用量"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_gb = memory.used / (1024**3)
            return used_gb < max_usage_gb
        except Exception:
            return True  # 如果无法检查，假设正常
    
    @staticmethod
    def check_disk_space(path: str, min_space_gb: float = 5.0) -> bool:
        """检查磁盘空间"""
        try:
            import shutil
            free_bytes = shutil.disk_usage(path).free
            free_gb = free_bytes / (1024**3)
            return free_gb >= min_space_gb
        except Exception:
            return True  # 如果无法检查，假设正常

# 全局错误处理器实例
global_error_handler = None

def get_error_handler(log_dir: str = "./logs", debug_mode: bool = False) -> ErrorHandler:
    """获取全局错误处理器"""
    global global_error_handler
    if global_error_handler is None:
        global_error_handler = ErrorHandler(log_dir, debug_mode)
    return global_error_handler

# 上下文管理器
class ErrorContext:
    """错误处理上下文管理器"""
    
    def __init__(self, context_name: str, error_handler: ErrorHandler = None,
                 error_type: str = "unknown", raise_on_error: bool = True):
        self.context_name = context_name
        self.error_handler = error_handler or get_error_handler()
        self.error_type = error_type
        self.raise_on_error = raise_on_error
    
    def __enter__(self):
        self.error_handler.logger.info(f"开始执行: {self.context_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.log_error(exc_val, self.context_name, self.error_type)
            if not self.raise_on_error:
                return True  # 抑制异常
        else:
            self.error_handler.logger.info(f"成功完成: {self.context_name}")
        return False

if __name__ == "__main__":
    # 测试错误处理系统
    handler = ErrorHandler(debug_mode=True)
    
    # 测试错误记录
    try:
        raise DataLoadError("测试数据加载错误")
    except Exception as e:
        handler.log_error(e, "测试上下文", "data_load")
    
    # 测试检查点
    test_data = {"step": 1, "results": [1, 2, 3]}
    checkpoint_file = handler.save_checkpoint(test_data, "test_checkpoint", "测试步骤")
    
    # 测试进度跟踪
    tracker = ProgressTracker(5, handler)
    for i in range(5):
        tracker.update(f"步骤 {i+1}", save_checkpoint=True, checkpoint_data={"step": i+1})
    
    print("错误处理系统测试完成")
    print(handler.get_error_summary())
    print(tracker.get_summary())