#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据流优化模块
实现MVPA分析流程的自动化管道、缓存机制和数据质量检查
"""

import os
import sys
import hashlib
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import warnings

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .global_config import GlobalConfig
from .error_handling import ErrorHandler, ErrorContext, ProgressTracker, DataValidator

@dataclass
class PipelineStep:
    """流水线步骤定义"""
    name: str
    function: Callable
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str] = None
    cache_enabled: bool = True
    quality_checks: Dict[str, Callable] = None
    retry_count: int = 3
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.quality_checks is None:
            self.quality_checks = {}

class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "./cache", max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_gb = max_size_gb
        self.index_file = self.cache_dir / "cache_index.json"
        self.load_index()
    
    def load_index(self):
        """加载缓存索引"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}
        except Exception:
            self.index = {}
    
    def save_index(self):
        """保存缓存索引"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            warnings.warn(f"保存缓存索引失败: {e}")
    
    def _get_cache_key(self, step_name: str, inputs: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 创建输入的哈希值
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        return f"{step_name}_{input_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def has_cache(self, step_name: str, inputs: Dict[str, Any]) -> bool:
        """检查是否有缓存"""
        cache_key = self._get_cache_key(step_name, inputs)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_key in self.index and cache_path.exists():
            # 检查缓存是否过期（可选）
            cache_info = self.index[cache_key]
            return True
        return False
    
    def get_cache(self, step_name: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """获取缓存数据"""
        if not self.has_cache(step_name, inputs):
            return None
        
        cache_key = self._get_cache_key(step_name, inputs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # 更新访问时间
            self.index[cache_key]['last_accessed'] = datetime.now().isoformat()
            self.save_index()
            
            return data
        except Exception as e:
            warnings.warn(f"读取缓存失败: {e}")
            return None
    
    def set_cache(self, step_name: str, inputs: Dict[str, Any], data: Any):
        """设置缓存数据"""
        cache_key = self._get_cache_key(step_name, inputs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # 检查缓存大小限制
            self._cleanup_if_needed()
            
            # 保存数据
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # 更新索引
            file_size = cache_path.stat().st_size
            self.index[cache_key] = {
                'step_name': step_name,
                'created': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'file_size': file_size,
                'file_path': str(cache_path)
            }
            self.save_index()
            
        except Exception as e:
            warnings.warn(f"保存缓存失败: {e}")
    
    def _cleanup_if_needed(self):
        """清理缓存以释放空间"""
        total_size = sum(info['file_size'] for info in self.index.values())
        max_size_bytes = self.max_size_gb * 1024**3
        
        if total_size > max_size_bytes:
            # 按最后访问时间排序，删除最旧的缓存
            sorted_items = sorted(
                self.index.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            for cache_key, info in sorted_items:
                if total_size <= max_size_bytes * 0.8:  # 清理到80%
                    break
                
                cache_path = Path(info['file_path'])
                if cache_path.exists():
                    cache_path.unlink()
                    total_size -= info['file_size']
                
                del self.index[cache_key]
            
            self.save_index()
    
    def clear_cache(self, step_name: Optional[str] = None):
        """清理缓存"""
        if step_name is None:
            # 清理所有缓存
            for cache_key, info in self.index.items():
                cache_path = Path(info['file_path'])
                if cache_path.exists():
                    cache_path.unlink()
            self.index = {}
        else:
            # 清理特定步骤的缓存
            to_remove = []
            for cache_key, info in self.index.items():
                if info['step_name'] == step_name:
                    cache_path = Path(info['file_path'])
                    if cache_path.exists():
                        cache_path.unlink()
                    to_remove.append(cache_key)
            
            for cache_key in to_remove:
                del self.index[cache_key]
        
        self.save_index()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(info['file_size'] for info in self.index.values())
        step_counts = {}
        
        for info in self.index.values():
            step_name = info['step_name']
            step_counts[step_name] = step_counts.get(step_name, 0) + 1
        
        return {
            'total_entries': len(self.index),
            'total_size_mb': total_size / (1024**2),
            'max_size_gb': self.max_size_gb,
            'usage_percent': (total_size / (self.max_size_gb * 1024**3)) * 100,
            'step_counts': step_counts
        }

class MVPAPipeline:
    """MVPA分析流水线"""
    
    def __init__(self, config: GlobalConfig, cache_enabled: bool = True):
        self.config = config
        self.cache_enabled = cache_enabled
        self.steps = {}
        self.execution_order = []
        
        # 初始化组件
        self.error_handler = ErrorHandler(
            log_dir=str(config.log_dir),
            debug_mode=config.debug_mode
        )
        
        if cache_enabled:
            cache_dir = config.base_dir / "cache"
            self.cache = DataCache(str(cache_dir))
        else:
            self.cache = None
        
        self.validator = DataValidator()
        
        # 执行状态
        self.execution_state = {
            'current_step': None,
            'completed_steps': [],
            'failed_steps': [],
            'step_results': {},
            'start_time': None,
            'end_time': None
        }
    
    def add_step(self, step: PipelineStep):
        """添加流水线步骤"""
        self.steps[step.name] = step
        self._update_execution_order()
    
    def _update_execution_order(self):
        """更新执行顺序（拓扑排序）"""
        # 简单的拓扑排序实现
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_name):
            if step_name in temp_visited:
                raise ValueError(f"检测到循环依赖: {step_name}")
            if step_name in visited:
                return
            
            temp_visited.add(step_name)
            
            if step_name in self.steps:
                for dep in self.steps[step_name].dependencies:
                    visit(dep)
            
            temp_visited.remove(step_name)
            visited.add(step_name)
            order.append(step_name)
        
        for step_name in self.steps.keys():
            if step_name not in visited:
                visit(step_name)
        
        self.execution_order = order
    
    def _check_dependencies(self, step_name: str) -> bool:
        """检查步骤依赖是否满足"""
        step = self.steps[step_name]
        for dep in step.dependencies:
            if dep not in self.execution_state['completed_steps']:
                return False
        return True
    
    def _run_quality_checks(self, step: PipelineStep, data: Any) -> bool:
        """运行质量检查"""
        if not step.quality_checks:
            return True
        
        all_passed = True
        for check_name, check_func in step.quality_checks.items():
            try:
                result = check_func(data)
                if not result:
                    self.error_handler.logger.warning(
                        f"质量检查失败: {step.name}.{check_name}"
                    )
                    all_passed = False
            except Exception as e:
                self.error_handler.log_error(
                    e, f"质量检查: {step.name}.{check_name}", "quality"
                )
                all_passed = False
        
        return all_passed
    
    def _execute_step(self, step_name: str, inputs: Dict[str, Any]) -> Any:
        """执行单个步骤"""
        step = self.steps[step_name]
        
        # 检查缓存
        if self.cache_enabled and step.cache_enabled and self.cache:
            cached_result = self.cache.get_cache(step_name, inputs)
            if cached_result is not None:
                self.error_handler.logger.info(f"使用缓存结果: {step_name}")
                return cached_result
        
        # 执行步骤
        with ErrorContext(f"执行步骤: {step_name}", self.error_handler, "processing"):
            result = step.function(**inputs)
            
            # 质量检查
            if not self._run_quality_checks(step, result):
                raise QualityError(f"步骤 {step_name} 质量检查失败")
            
            # 保存到缓存
            if self.cache_enabled and step.cache_enabled and self.cache:
                self.cache.set_cache(step_name, inputs, result)
            
            return result
    
    def run(self, start_step: Optional[str] = None, end_step: Optional[str] = None,
           force_rerun: bool = False) -> Dict[str, Any]:
        """运行流水线"""
        self.execution_state['start_time'] = datetime.now()
        
        # 确定执行范围
        if start_step:
            start_idx = self.execution_order.index(start_step)
        else:
            start_idx = 0
        
        if end_step:
            end_idx = self.execution_order.index(end_step) + 1
        else:
            end_idx = len(self.execution_order)
        
        execution_steps = self.execution_order[start_idx:end_idx]
        
        # 初始化进度跟踪
        tracker = ProgressTracker(len(execution_steps), self.error_handler)
        
        self.error_handler.logger.info(
            f"开始执行流水线: {len(execution_steps)} 个步骤"
        )
        
        try:
            for step_name in execution_steps:
                if step_name not in self.steps:
                    raise ValueError(f"未知步骤: {step_name}")
                
                step = self.steps[step_name]
                self.execution_state['current_step'] = step_name
                
                # 检查依赖
                if not self._check_dependencies(step_name):
                    raise ValueError(f"步骤 {step_name} 的依赖未满足")
                
                # 准备输入
                inputs = {}
                for input_name in step.inputs:
                    if input_name in self.execution_state['step_results']:
                        inputs[input_name] = self.execution_state['step_results'][input_name]
                    else:
                        # 尝试从配置中获取
                        if hasattr(self.config, input_name):
                            inputs[input_name] = getattr(self.config, input_name)
                        else:
                            raise ValueError(f"无法找到输入: {input_name}")
                
                # 执行步骤
                try:
                    result = self._execute_step(step_name, inputs)
                    
                    # 保存结果
                    for i, output_name in enumerate(step.outputs):
                        if isinstance(result, (list, tuple)):
                            self.execution_state['step_results'][output_name] = result[i]
                        else:
                            self.execution_state['step_results'][output_name] = result
                    
                    self.execution_state['completed_steps'].append(step_name)
                    
                    # 更新进度
                    tracker.update(
                        step_name,
                        save_checkpoint=True,
                        checkpoint_data={
                            'step_name': step_name,
                            'result': result,
                            'execution_state': self.execution_state.copy()
                        }
                    )
                    
                except Exception as e:
                    self.execution_state['failed_steps'].append(step_name)
                    self.error_handler.log_error(e, f"执行步骤: {step_name}", "processing")
                    
                    if not force_rerun:
                        raise
            
            self.execution_state['end_time'] = datetime.now()
            
            self.error_handler.logger.info(
                f"流水线执行完成: {len(self.execution_state['completed_steps'])} 个步骤成功, "
                f"{len(self.execution_state['failed_steps'])} 个步骤失败"
            )
            
            return self.execution_state['step_results']
            
        except Exception as e:
            self.execution_state['end_time'] = datetime.now()
            self.error_handler.log_error(e, "流水线执行", "processing")
            raise
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        if self.execution_state['start_time'] and self.execution_state['end_time']:
            duration = self.execution_state['end_time'] - self.execution_state['start_time']
        else:
            duration = None
        
        return {
            'total_steps': len(self.steps),
            'completed_steps': len(self.execution_state['completed_steps']),
            'failed_steps': len(self.execution_state['failed_steps']),
            'success_rate': len(self.execution_state['completed_steps']) / len(self.steps) * 100,
            'duration': str(duration).split('.')[0] if duration else None,
            'current_step': self.execution_state['current_step'],
            'cache_stats': self.cache.get_cache_stats() if self.cache else None
        }
    
    def resume_from_checkpoint(self, checkpoint_file: str):
        """从检查点恢复执行"""
        checkpoint_data = self.error_handler.load_checkpoint(checkpoint_file)
        if checkpoint_data:
            self.execution_state = checkpoint_data['execution_state']
            self.error_handler.logger.info(f"从检查点恢复: {checkpoint_data['step_name']}")
            return True
        return False

def create_lss_pipeline_step(config: GlobalConfig) -> PipelineStep:
    """创建LSS分析步骤"""
    def run_lss_analysis(subjects, runs, **kwargs):
        # 这里会调用优化后的LSS分析函数
        from .lss_enhanced import run_lss_analysis_enhanced
        return run_lss_analysis_enhanced(subjects, runs, config)
    
    return PipelineStep(
        name="lss_analysis",
        function=run_lss_analysis,
        inputs=["subjects", "runs"],
        outputs=["lss_results"],
        quality_checks={
            "check_output_files": lambda x: len(x.get('output_files', [])) > 0,
            "check_beta_images": lambda x: all(
                DataValidator.check_nifti_integrity(f) 
                for f in x.get('beta_files', [])
            )
        }
    )

def create_roi_pipeline_step(config: GlobalConfig) -> PipelineStep:
    """创建ROI生成步骤"""
    def run_roi_generation(**kwargs):
        from .roi_enhanced import create_roi_masks_enhanced
        return create_roi_masks_enhanced(config)
    
    return PipelineStep(
        name="roi_generation",
        function=run_roi_generation,
        inputs=[],
        outputs=["roi_masks"],
        dependencies=[],
        quality_checks={
            "check_roi_count": lambda x: len(x.get('roi_files', [])) > 0,
            "check_roi_files": lambda x: all(
                DataValidator.check_nifti_integrity(f) 
                for f in x.get('roi_files', [])
            )
        }
    )

def create_mvpa_pipeline_step(config: GlobalConfig) -> PipelineStep:
    """创建MVPA分析步骤"""
    def run_mvpa_analysis(lss_results, roi_masks, **kwargs):
        from .mvpa_enhanced import run_mvpa_analysis_enhanced
        return run_mvpa_analysis_enhanced(lss_results, roi_masks, config)
    
    return PipelineStep(
        name="mvpa_analysis",
        function=run_mvpa_analysis,
        inputs=["lss_results", "roi_masks"],
        outputs=["mvpa_results"],
        dependencies=["lss_analysis", "roi_generation"],
        quality_checks={
            "check_results_format": lambda x: isinstance(x, dict) and 'group_results' in x,
            "check_statistics": lambda x: 'statistics' in x and len(x['statistics']) > 0
        }
    )

def create_full_pipeline(config: GlobalConfig) -> MVPAPipeline:
    """创建完整的MVPA分析流水线"""
    pipeline = MVPAPipeline(config)
    
    # 添加步骤
    pipeline.add_step(create_lss_pipeline_step(config))
    pipeline.add_step(create_roi_pipeline_step(config))
    pipeline.add_step(create_mvpa_pipeline_step(config))
    
    return pipeline

if __name__ == "__main__":
    # 测试数据流系统
    from .global_config import create_config
    
    config = create_config()
    config.lss.subjects = ['sub-01', 'sub-02']
    config.lss.runs = [1, 2, 3]
    
    # 创建流水线
    pipeline = create_full_pipeline(config)
    
    print("流水线步骤:")
    for step_name in pipeline.execution_order:
        step = pipeline.steps[step_name]
        print(f"  {step_name}: {step.inputs} -> {step.outputs}")
    
    print("\n数据流系统测试完成")