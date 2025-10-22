# report_generator.py
# HTML报告生成模块 - Searchlight版本

import base64
import numpy as np
import pandas as pd
from pathlib import Path

def log(message, config):
    """简单的日志函数"""
    print(f"[LOG] {message}")

def image_to_base64(image_path):
    """将图片转换为base64编码，增强错误处理"""
    try:
        # 检查文件是否存在
        if not Path(image_path).exists():
            print(f"警告：图片文件不存在: {image_path}")
            return None
        
        # 检查文件大小，避免加载过大的文件
        file_size = Path(image_path).stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB限制
            print(f"警告：图片文件过大 ({file_size/1024/1024:.1f}MB): {image_path}")
            return None
        
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            print(f"成功编码图片: {image_path} ({file_size/1024:.1f}KB)")
            return encoded
            
    except PermissionError:
        print(f"权限错误：无法读取图片文件: {image_path}")
        return None
    except MemoryError:
        print(f"内存错误：图片文件过大无法加载: {image_path}")
        return None
    except Exception as e:
        print(f"图片编码失败: {image_path}, 错误类型: {type(e).__name__}, 详情: {e}")
        return None

def generate_image_html(image_b64, alt_text):
    """
    生成图片HTML代码，处理加载失败的情况
    
    Args:
        image_b64: base64编码的图片数据
        alt_text: 图片的alt文本
    
    Returns:
        str: HTML代码字符串
    """
    if image_b64:
        return f'<img src="data:image/png;base64,{image_b64}" alt="{alt_text}" style="max-width: 100%; height: auto; margin: 10px 0;">'
    else:
        return f'<div class="image-placeholder" style="background-color: #f0f0f0; padding: 20px; text-align: center; margin: 10px 0; border: 1px dashed #ccc;">图片加载失败: {alt_text}</div>'

def generate_html_report(group_df, stats_df, config, main_viz_b64=None, individual_viz_b64=None):
    """
    生成Searchlight MVPA分析的HTML报告
    
    Args:
        group_df: 个体结果DataFrame
        stats_df: 组水平统计结果DataFrame
        config: 配置对象
        main_viz_b64: 主要可视化图片的base64编码
        individual_viz_b64: 个体结果可视化图片的base64编码
    
    Returns:
        Path: 生成的HTML报告文件路径
    """
    
    # 生成报告内容
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>fMRI Searchlight MVPA 分析报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }}
        .header h1 {{
            color: #007acc;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.2em;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #007acc;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .section h3 {{
            color: #333;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .info-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        .info-card h4 {{
            margin-top: 0;
            color: #007acc;
        }}
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007acc;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .significant {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .not-significant {{
            background-color: #f8d7da !important;
        }}
        .metric {{
            font-size: 1.1em;
            font-weight: bold;
        }}
        .accuracy-high {{
            color: #28a745;
        }}
        .accuracy-medium {{
            color: #ffc107;
        }}
        .accuracy-low {{
            color: #dc3545;
        }}
        .visualization {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-placeholder {{
            background-color: #f0f0f0;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            border: 2px dashed #ccc;
            border-radius: 8px;
            color: #666;
        }}
        .summary-stats {{
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .config-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        .config-item {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #007acc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 fMRI Searchlight MVPA 分析报告</h1>
            <div class="subtitle">多体素模式分析 - Searchlight方法</div>
            <div class="subtitle">生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <!-- 分析概览 -->
        <div class="section">
            <h2>📊 分析概览</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h4>被试信息</h4>
                    <p><strong>总被试数:</strong> {len(group_df['subject'].unique()) if not group_df.empty else 0}</p>
                    <p><strong>有效被试:</strong> {', '.join(sorted(group_df['subject'].unique())) if not group_df.empty else '无'}</p>
                </div>
                <div class="info-card">
                    <h4>分析参数</h4>
                    <p><strong>Searchlight半径:</strong> {config.searchlight_radius} 体素</p>
                    <p><strong>交叉验证:</strong> {config.cv_folds} fold</p>
                    <p><strong>处理Mask:</strong> {config.process_mask}</p>
                </div>
                <div class="info-card">
                    <h4>对比条件</h4>
                    <p><strong>对比数量:</strong> {len(config.contrasts)}</p>
                    <p><strong>条件:</strong> {', '.join([f"{c[1]} vs {c[2]}" for c in config.contrasts])}</p>
                </div>
                <div class="info-card">
                    <h4>结果文件</h4>
                    <p><strong>结果目录:</strong> {config.results_dir.name}</p>
                    <p><strong>总分析数:</strong> {len(group_df) if not group_df.empty else 0}</p>
                </div>
            </div>
        </div>

        <!-- 主要结果可视化 -->
        <div class="section">
            <h2>📈 主要结果可视化</h2>
            <div class="visualization">
                {generate_image_html(main_viz_b64, "主要结果可视化")}
            </div>
        </div>

        <!-- 组水平统计结果 -->
        <div class="section">
            <h2>🔬 组水平统计结果</h2>
            {generate_stats_table(stats_df) if not stats_df.empty else '<p>无统计结果</p>'}
        </div>

        <!-- 个体结果详情 -->
        <div class="section">
            <h2>👤 个体结果详情</h2>
            <div class="visualization">
                {generate_image_html(individual_viz_b64, "个体结果可视化")}
            </div>
            {generate_individual_table(group_df) if not group_df.empty else '<p>无个体结果</p>'}
        </div>

        <!-- 分析配置 -->
        <div class="section">
            <h2>⚙️ 分析配置</h2>
            {generate_config_section(config)}
        </div>

        <div class="footer">
            <p>报告由 fMRI Searchlight MVPA Pipeline 自动生成</p>
            <p>如有问题，请检查日志文件: {config.log_file}</p>
        </div>
    </div>
</body>
</html>
"""

    # 保存HTML报告
    report_path = config.results_dir / "searchlight_mvpa_report.html"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML报告已生成: {report_path}")
    except Exception as e:
        print(f"生成HTML报告失败: {str(e)}")
        return None
    
    return report_path

def generate_stats_table(stats_df):
    """生成组水平统计结果表格"""
    if stats_df.empty:
        return "<p>无统计结果数据</p>"
    
    table_html = """
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>对比条件</th>
                    <th>被试数</th>
                    <th>平均准确率</th>
                    <th>平均机会水平</th>
                    <th>平均差值</th>
                    <th>差值SEM</th>
                    <th>差值95% CI</th>
                    <th>t统计量(Δ)</th>
                    <th>符号翻转p值</th>
                    <th>置换次数</th>
                    <th>效应量(d)</th>
                    <th>显著性</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in stats_df.iterrows():
        # 判断显著性
        is_significant = row.get('significant_permutation', False) or row.get('significant_t_test', False)
        row_class = 'significant' if is_significant else 'not-significant'
        
        # 准确率颜色
        acc = row['mean_accuracy']
        if acc >= 0.6:
            acc_class = 'accuracy-high'
        elif acc >= 0.55:
            acc_class = 'accuracy-medium'
        else:
            acc_class = 'accuracy-low'
        
        significance_symbol = '✓' if is_significant else '✗'
        
        ci_lower_delta = row.get('ci_lower_delta', np.nan)
        ci_upper_delta = row.get('ci_upper_delta', np.nan)
        sem_delta = row.get('sem_delta_accuracy', row.get('sem_accuracy', np.nan))
        permutations = row.get('n_group_permutations', np.nan)

        table_html += f"""
                <tr class="{row_class}">
                    <td><strong>{row['contrast']}</strong></td>
                    <td>{row['n_subjects']}</td>
                    <td class="{acc_class} metric">{row['mean_accuracy']:.4f}</td>
                    <td>{row.get('mean_chance_accuracy', np.nan):.4f}</td>
                    <td>{row.get('mean_delta_accuracy', np.nan):.4f}</td>
                    <td>{sem_delta:.4f}</td>
                    <td>[{ci_lower_delta:.3f}, {ci_upper_delta:.3f}]</td>
                    <td>{row['t_statistic']:.3f}</td>
                    <td>{row.get('sign_flip_pvalue', row.get('permutation_pvalue', np.nan)):.4f}</td>
                    <td>{int(permutations) if np.isfinite(permutations) else 'N/A'}</td>
                    <td>{row['cohens_d']:.3f}</td>
                    <td>{significance_symbol}</td>
                </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    </div>
    
    <div class="summary-stats">
        <h4>📋 结果解读</h4>
        <ul>
            <li><strong>显著性标准:</strong> p < 0.05 (置换检验或t检验)</li>
            <li><strong>准确率解读:</strong> 
                <span class="accuracy-high">≥60% (高)</span>, 
                <span class="accuracy-medium">55-60% (中等)</span>, 
                <span class="accuracy-low">&lt;55% (低)</span>
            </li>
            <li><strong>效应量解读:</strong> |d| ≥ 0.8 (大), 0.5-0.8 (中等), 0.2-0.5 (小), &lt;0.2 (微小)</li>
            <li><strong>两步校正:</strong> 个体层面置换估计机会水平，组层面符号翻转并结合max-T控制FWER</li>
        </ul>
    </div>
    """
    
    return table_html

def generate_individual_table(group_df):
    """生成个体结果表格"""
    if group_df.empty:
        return "<p>无个体结果数据</p>"

    table_html = """
    <div class="table-container">
        <h3>个体分析结果</h3>
        <table>
            <thead>
                <tr>
                    <th>被试</th>
                    <th>对比条件</th>
                    <th>条件1样本数</th>
                    <th>条件2样本数</th>
                    <th>平均准确率</th>
                    <th>平均机会水平</th>
                    <th>平均差值</th>
                    <th>准确率图</th>
                    <th>机会水平图</th>
                    <th>差值图</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in group_df.iterrows():
        acc = row.get('mean_accuracy', row.get('accuracy', np.nan))
        if acc >= 0.6:
            acc_class = 'accuracy-high'
        elif acc >= 0.55:
            acc_class = 'accuracy-medium'
        else:
            acc_class = 'accuracy-low'

        accuracy_map = row.get('accuracy_map_path')
        chance_map = row.get('chance_map_path')
        delta_map = row.get('delta_map_path')

        def make_link(path):
            if path:
                name = Path(path).name
                return f'<a href="{path}">{name}</a>'
            return '缺失'

        table_html += f"""
                <tr>
                    <td><strong>{row.get('subject', 'N/A')}</strong></td>
                    <td>{row.get('contrast', 'N/A')}</td>
                    <td>{row.get('n_trials_cond1', 'N/A')}</td>
                    <td>{row.get('n_trials_cond2', 'N/A')}</td>
                    <td class="{acc_class} metric">{acc:.4f}</td>
                    <td>{row.get('mean_chance_accuracy', np.nan):.4f}</td>
                    <td>{row.get('mean_delta_accuracy', np.nan):.4f}</td>
                    <td>{make_link(accuracy_map)}</td>
                    <td>{make_link(chance_map)}</td>
                    <td>{make_link(delta_map)}</td>
                </tr>
        """

    table_html += """
            </tbody>
        </table>
    </div>
    """

    return table_html

def generate_config_section(config):
    """生成配置信息部分"""
    config_html = """
    <div class="config-section">
        <div class="config-grid">
            <div class="config-item">
                <h4>🎯 分析参数</h4>
                <p><strong>Searchlight半径:</strong> {radius} 体素</p>
                <p><strong>最小区域大小:</strong> {min_size} 体素</p>
                <p><strong>交叉验证折数:</strong> {cv_folds}</p>
                <p><strong>处理Mask:</strong> {mask}</p>
            </div>
            <div class="config-item">
                <h4>📁 数据路径</h4>
                <p><strong>LSS根目录:</strong> {lss_root}</p>
                <p><strong>Mask目录:</strong> {mask_dir}</p>
                <p><strong>结果目录:</strong> {results_dir}</p>
            </div>
            <div class="config-item">
                <h4>🔧 算法设置</h4>
                <p><strong>分类器:</strong> SVM (线性核)</p>
                <p><strong>并行进程:</strong> {n_jobs}</p>
            </div>
            <div class="config-item">
                <h4>📊 统计设置</h4>
                <p><strong>置换次数:</strong> {n_perm}</p>
                <p><strong>被试内置换:</strong> {within_perm}</p>
                <p><strong>精确符号翻转阈值:</strong> {max_exact} 名被试</p>
                <p><strong>显著性水平:</strong> {alpha}</p>
                <p><strong>随机种子:</strong> {random_state}</p>
            </div>
        </div>
    </div>
    """.format(
        radius=config.searchlight_radius,
        min_size=config.min_region_size,
        cv_folds=config.cv_folds,
        mask=config.process_mask,
        lss_root=config.lss_root.name,
        mask_dir=config.mask_dir.name,
        results_dir=config.results_dir.name,
        n_jobs=config.n_jobs,
        n_perm=config.n_permutations,
        within_perm=getattr(config, 'within_subject_permutations', 'N/A'),
        max_exact=getattr(config, 'max_exact_sign_flips_subjects', 'N/A'),
        alpha=config.alpha_level,
        random_state=config.cv_random_state
    )

    return config_html
