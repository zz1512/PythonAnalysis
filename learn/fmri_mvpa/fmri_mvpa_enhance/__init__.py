"""增强版MVPA分析流水线

这是一个功能完整、高度优化的fMRI多体素模式分析(MVPA)流水线，
集成了数据流优化、配置管理、错误处理、质量控制和可视化等多个增强功能。

Enhanced MVPA Analysis Pipeline

A comprehensive and highly optimized fMRI Multi-Voxel Pattern Analysis (MVPA) pipeline
with integrated data flow optimization, configuration management, error handling,
quality control, and visualization enhancements.
"""

__version__ = "1.0.0"
__author__ = "MVPA Enhancement Team"
__email__ = "mvpa@example.com"
__description__ = "Enhanced fMRI MVPA Analysis Pipeline"

# 核心模块导入 (Core Module Imports)
try:
    from .enhanced_pipeline import (
        EnhancedMVPAPipeline,
        create_enhanced_pipeline,
        run_quick_analysis
    )
    from .global_config import (
        GlobalConfig,
        LSSConfig,
        ROIConfig,
        MVPAConfig
    )
    from .error_handling import (
        MVPAError,
        DataError,
        ConfigError,
        ProcessingError,
        ErrorHandler,
        retry_on_failure
    )
    from .data_flow import (
        MVPAPipeline,
        DataCache,
        PipelineStep
    )
    from .lss_enhanced import LSSAnalysisEnhanced
    from .quality_control import (
        PerformanceMonitor,
        DataQualityAssessor,
        QualityReporter,
        run_comprehensive_quality_check
    )
    from .visualization import (
        MVPAVisualizer,
        ReportGenerator
    )
except ImportError as e:
    import warnings
    warnings.warn(
        f"Some modules could not be imported: {e}. "
        "Please check that all dependencies are installed.",
        ImportWarning
    )

# 便捷导入 (Convenience Imports)
__all__ = [
    # 版本信息 (Version Info)
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    
    # 主要类和函数 (Main Classes and Functions)
    "EnhancedMVPAPipeline",
    "create_enhanced_pipeline",
    "run_quick_analysis",
    
    # 配置管理 (Configuration Management)
    "GlobalConfig",
    "LSSConfig",
    "ROIConfig",
    "MVPAConfig",
    
    # 错误处理 (Error Handling)
    "MVPAError",
    "DataError",
    "ConfigError",
    "ProcessingError",
    "ErrorHandler",
    "retry_on_failure",
    
    # 数据流 (Data Flow)
    "MVPAPipeline",
    "DataCache",
    "PipelineStep",
    
    # 分析模块 (Analysis Modules)
    "LSSAnalysisEnhanced",
    
    # 质量控制 (Quality Control)
    "PerformanceMonitor",
    "DataQualityAssessor",
    "QualityReporter",
    "run_comprehensive_quality_check",
    
    # 可视化 (Visualization)
    "MVPAVisualizer",
    "ReportGenerator",
]

# 模块级别的便捷函数 (Module-level Convenience Functions)
def get_version():
    """获取版本信息"""
    return __version__

def get_info():
    """获取包信息"""
    return {
        "name": "fmri_mvpa_enhance",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__
    }

def check_dependencies():
    """检查依赖包是否正确安装"""
    import importlib
    
    required_packages = [
        "numpy", "scipy", "pandas", "nibabel", "nilearn",
        "sklearn", "matplotlib", "seaborn", "plotly",
        "joblib", "psutil", "tqdm"
    ]
    
    missing_packages = []
    installed_packages = {}
    
    for package in required_packages:
        try:
            if package == "sklearn":
                module = importlib.import_module("sklearn")
            else:
                module = importlib.import_module(package)
            
            version = getattr(module, "__version__", "unknown")
            installed_packages[package] = version
        except ImportError:
            missing_packages.append(package)
    
    return {
        "installed": installed_packages,
        "missing": missing_packages,
        "all_installed": len(missing_packages) == 0
    }

def print_dependency_status():
    """打印依赖包状态"""
    status = check_dependencies()
    
    print("=== 增强版MVPA流水线依赖包状态 ===")
    print(f"包版本: {__version__}")
    print()
    
    if status["all_installed"]:
        print("✅ 所有必需依赖包已正确安装")
    else:
        print("❌ 缺少以下依赖包:")
        for package in status["missing"]:
            print(f"  - {package}")
        print()
        print("请运行以下命令安装缺少的包:")
        print(f"pip install {' '.join(status['missing'])}")
    
    print()
    print("已安装的包:")
    for package, version in status["installed"].items():
        print(f"  {package}: {version}")

# 包初始化时的检查 (Package Initialization Checks)
def _initialize_package():
    """包初始化函数"""
    import warnings
    
    # 检查Python版本
    import sys
    if sys.version_info < (3, 7):
        warnings.warn(
            "Python 3.7+ is recommended for optimal performance. "
            f"Current version: {sys.version}",
            UserWarning
        )
    
    # 检查关键依赖
    try:
        import numpy
        import nibabel
        import sklearn
    except ImportError as e:
        warnings.warn(
            f"Critical dependency missing: {e}. "
            "Please install all required packages using: "
            "pip install -r requirements.txt",
            ImportWarning
        )

# 执行包初始化
_initialize_package()

# 设置默认的警告过滤器
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="nilearn")

# 包级别的配置
DEFAULT_CONFIG = {
    "enable_cache": True,
    "enable_quality_control": True,
    "enable_visualization": True,
    "n_jobs": -1,
    "verbose": True
}

# 导出默认配置
__all__.append("DEFAULT_CONFIG")