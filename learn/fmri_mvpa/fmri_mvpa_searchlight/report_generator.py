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


def describe_cross_validation(config):
    """根据配置对象生成交叉验证描述。"""

    strategy = getattr(config, 'cv_strategy', None)
    if strategy:
        return str(strategy)

    cv_folds = getattr(config, 'cv_folds', None)
    if cv_folds:
        return f"{cv_folds}-fold StratifiedKFold"

    return "未指定"


def describe_classifier(config):
    """生成分类器描述。"""

    params = getattr(config, 'svm_params', {}) or {}
    kernel = params.get('kernel', 'linear')
    c_val = params.get('C', '1.0')
    class_weight = params.get('class_weight', 'balanced')

    if kernel == 'linear':
        return f"线性SVM (C={c_val}, class_weight={class_weight})"

    return f"SVM (kernel={kernel}, C={c_val})"


def generate_methodology_section(config):
    """生成方法概述部分。"""

    chance_level = getattr(config, 'chance_level', 0.5)
    balance_enabled = getattr(config, 'balance_trials', False)
    balance_text = (
        "在每个条件内随机下采样至相同trial数，控制类别先验。"
        if balance_enabled
        else "使用全部可用trial，保留原始分布。"
    )

    permutation_strategy = getattr(config, 'permutation_strategy', 'analytic')
    within_perm = getattr(config, 'within_subject_permutations', 'N/A')

    items = [
        f"Searchlight半径 {getattr(config, 'searchlight_radius', 'N/A')} 体素，分析范围由 {getattr(config, 'process_mask', 'N/A')} 限定。",
        f"分类器: {describe_classifier(config)}，每个球体内特征进行z分数标准化。",
        f"交叉验证: {describe_cross_validation(config)}。",
        f"样本平衡: {'已启用' if balance_enabled else '未启用'}，{balance_text}",
        (
            f"机会水平设置为 {chance_level:.2f}，"\
            "组水平对准确率-机会值差异执行单样本t检验与符号翻转置换。"
        ),
        (
            f"被试内机会水平估计采用 {permutation_strategy} 策略"
            f" (置换次数 {within_perm})。"
        ),
    ]

    list_items = "".join(f"<li>{item}</li>" for item in items)

    return (
        "<div class=\"summary-stats\">"
        "<ul>"
        f"{list_items}"
        "</ul>"
        "</div>"
    )

def generate_html_report(
    group_df,
    stats_df,
    config,
    main_viz_b64=None,
    individual_viz_b64=None,
    map_outputs=None,
    cluster_df=None,
    cluster_summary_path=None,
    significance_summary_df=None,
    significance_summary_path=None,
    significance_viz_b64=None,
):
    """
    生成Searchlight MVPA分析的HTML报告
    
    Args:
        group_df: 个体结果DataFrame
        stats_df: 组水平统计结果DataFrame
        config: 配置对象
        main_viz_b64: 主要可视化图片的base64编码
        individual_viz_b64: 个体结果可视化图片的base64编码
        map_outputs: 每个对比的全脑图路径信息
        cluster_df: 显著脑区的汇总DataFrame
        cluster_summary_path: 显著脑区汇总表路径
        significance_summary_df: 分类阈值筛选后的显著体素汇总表
        significance_summary_path: 分类阈值显著体素汇总表路径
        significance_viz_b64: 分类阈值显著性图的base64编码
    
    Returns:
        Path: 生成的HTML报告文件路径
    """
    
    maps_section_html = generate_map_outputs_section(map_outputs, config) if map_outputs else "<p>未生成组水平全脑图。</p>"
    cluster_section_html = generate_cluster_table(cluster_df) if cluster_df is not None and not cluster_df.empty else "<p>未检测到显著脑区。</p>"
    summary_file_html = ""
    if cluster_summary_path:
        summary_path = Path(cluster_summary_path)
        try:
            rel_summary = summary_path.relative_to(config.results_dir)
        except ValueError:
            rel_summary = summary_path
        summary_file_html = f"<p>显著簇汇总表: <code>{rel_summary.as_posix()}</code></p>"

    significance_section_html = generate_significant_voxel_table(significance_summary_df) if significance_summary_df is not None and not significance_summary_df.empty else "<p>没有体素在分类阈值以上达到统计显著。</p>"
    significance_summary_file_html = ""
    if significance_summary_path:
        sig_summary_path = Path(significance_summary_path)
        try:
            rel_sig_summary = sig_summary_path.relative_to(config.results_dir)
        except ValueError:
            rel_sig_summary = sig_summary_path
        significance_summary_file_html = f"<p>分类阈值显著性汇总表: <code>{rel_sig_summary.as_posix()}</code></p>"

    method_section_html = generate_methodology_section(config)
    cv_description = describe_cross_validation(config)
    chance_level = getattr(config, 'chance_level', 0.5)

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
                    <p><strong>交叉验证:</strong> {cv_description}</p>
                    <p><strong>处理Mask:</strong> {config.process_mask}</p>
                    <p><strong>机会水平:</strong> {chance_level:.2f}</p>
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

        <!-- 方法概述 -->
        <div class="section">
            <h2>🔍 分析方法概述</h2>
            {method_section_html}
        </div>

        <!-- 主要结果可视化 -->
        <div class="section">
            <h2>📈 主要结果可视化</h2>
            <div class="visualization">
                {generate_image_html(main_viz_b64, "主要结果可视化")}
            </div>
        </div>

        <!-- 全脑准确率图与显著脑区 -->
        <div class="section">
            <h2>🧠 全脑准确率图与显著脑区</h2>
            {maps_section_html}
            {summary_file_html}
            <h3>显著脑区定位</h3>
            {cluster_section_html}
        </div>

        <!-- 分类阈值显著性 -->
        <div class="section">
            <h2>🎯 分类阈值显著性汇总 (准确率 &gt; {getattr(config, 'classification_significance_threshold', 0.5):.2f})</h2>
            <div class="visualization">
                {generate_image_html(significance_viz_b64, "分类阈值显著性可视化")}
            </div>
            {significance_section_html}
            {significance_summary_file_html}
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
        # 使用带BOM的UTF-8编码写入，确保在Windows系统（尤其是旧版记事本）中
        # 也能正确识别中文字符，避免出现乱码。
        with open(report_path, 'w', encoding='utf-8-sig') as f:
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
                    <th>t统计量(Δ)</th>
                    <th>t检验p值</th>
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
        significance_symbol = '✓' if is_significant else '✗'
        permutations = row.get('n_group_permutations', np.nan)

        table_html += f"""
                <tr class="{row_class}">
                    <td><strong>{row['contrast']}</strong></td>
                    <td>{row['n_subjects']}</td>
                    <td>{row['t_statistic']:.3f}</td>
                    <td>{row.get('t_pvalue', np.nan):.4f}</td>
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


def generate_map_outputs_section(map_outputs, config):
    """生成全脑准确率图与统计图的文件列表"""

    if not map_outputs:
        return "<p>未生成组水平全脑图。</p>"

    def _format_path(path):
        if not path:
            return "无"
        path_obj = Path(path)
        try:
            rel_path = path_obj.relative_to(config.results_dir)
        except ValueError:
            rel_path = path_obj
        return f"<code>{rel_path.as_posix()}</code>"

    table_html = """
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>对比条件</th>
                    <th>全脑激活图 (Δ准确率)</th>
                    <th>t统计图</th>
                    <th>p值图</th>
                    <th>分类阈值</th>
                    <th>&gt;阈值体素数</th>
                    <th>分类阈值掩模</th>
                    <th>显著性掩模</th>
                    <th>阈值化激活图</th>
                    <th>显著簇表</th>
                </tr>
            </thead>
            <tbody>
    """

    for contrast, outputs in sorted(map_outputs.items()):
        threshold_value = outputs.get('classification_threshold')
        if isinstance(threshold_value, (int, float)):
            threshold_text = f"{threshold_value:.2f}"
        else:
            threshold_text = 'N/A'
        voxel_count = int(outputs.get('n_voxels_above_threshold', 0))
        table_html += f"""
                <tr>
                    <td><strong>{contrast}</strong></td>
                    <td>{_format_path(outputs.get('activation_map'))}</td>
                    <td>{_format_path(outputs.get('t_map'))}</td>
                    <td>{_format_path(outputs.get('p_map'))}</td>
                    <td>{threshold_text}</td>
                    <td>{voxel_count}</td>
                    <td>{_format_path(outputs.get('classification_mask'))}</td>
                    <td>{_format_path(outputs.get('significant_mask'))}</td>
                    <td>{_format_path(outputs.get('thresholded_delta_map'))}</td>
                    <td>{_format_path(outputs.get('cluster_table'))}</td>
                </tr>
        """

    table_html += """
            </tbody>
        </table>
    </div>
    """

    return table_html


def generate_cluster_table(cluster_df):
    """生成显著脑区表格"""

    if cluster_df is None or cluster_df.empty:
        return "<p>未检测到显著脑区。</p>"

    table_html = """
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>对比条件</th>
                    <th>簇ID</th>
                    <th>体素数量</th>
                    <th>体积 (mm³)</th>
                    <th>平均Δ准确率</th>
                    <th>峰值Δ准确率</th>
                    <th>峰值t</th>
                    <th>峰值p</th>
                    <th>峰值坐标 (x, y, z)</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in cluster_df.iterrows():
        table_html += f"""
                <tr>
                    <td>{row['contrast']}</td>
                    <td>{int(row['cluster_id'])}</td>
                    <td>{int(row['n_voxels'])}</td>
                    <td>{row['cluster_volume_mm3']:.1f}</td>
                    <td>{row['mean_delta_accuracy']:.4f}</td>
                    <td>{row['peak_delta_accuracy']:.4f}</td>
                    <td>{row['peak_t_value']:.3f}</td>
                    <td>{row['peak_p_value']:.4f}</td>
                    <td>({row['peak_x']:.1f}, {row['peak_y']:.1f}, {row['peak_z']:.1f})</td>
                </tr>
        """

    table_html += """
            </tbody>
        </table>
    </div>
        """

    return table_html


def generate_significant_voxel_table(summary_df):
    """生成分类阈值显著体素汇总表"""

    if summary_df is None or summary_df.empty:
        return "<p>没有体素在分类阈值以上达到统计显著。</p>"

    display_df = summary_df.copy()
    display_df['proportion_percent'] = display_df['proportion_significant'] * 100.0

    table_html = """
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>对比条件</th>
                    <th>校正方式</th>
                    <th>分类阈值</th>
                    <th>&gt;阈值体素数</th>
                    <th>显著体素数</th>
                    <th>显著体素占比</th>
                    <th>显著体素平均Δ准确</th>
                    <th>显著体素峰值准确率</th>
                    <th>显著簇表</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in display_df.iterrows():
        row_class = 'significant' if row['n_significant_voxels'] > 0 else 'not-significant'
        cluster_table = row.get('cluster_table')
        if isinstance(cluster_table, Path):
            cluster_text = cluster_table.as_posix()
        elif isinstance(cluster_table, str) and cluster_table:
            cluster_text = cluster_table
        else:
            cluster_text = '无'
        mean_delta = row.get('mean_significant_delta')
        mean_delta_text = f"{mean_delta:.4f}" if pd.notna(mean_delta) else 'NaN'
        peak_accuracy = row.get('peak_significant_accuracy')
        peak_accuracy_text = f"{peak_accuracy:.4f}" if pd.notna(peak_accuracy) else 'NaN'
        table_html += f"""
                <tr class="{row_class}">
                    <td>{row['contrast']}</td>
                    <td>{row['correction'].upper()}</td>
                    <td>{row['classification_threshold']:.2f}</td>
                    <td>{int(row['n_voxels_above_threshold'])}</td>
                    <td>{int(row['n_significant_voxels'])}</td>
                    <td>{row['proportion_percent']:.2f}%</td>
                    <td>{mean_delta_text}</td>
                    <td>{peak_accuracy_text}</td>
                    <td>{cluster_text}</td>
                </tr>
        """

    table_html += """
            </tbody>
        </table>
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
                    <th>条件1样本 (用/总)</th>
                    <th>条件2样本 (用/总)</th>
                    <th>启用平衡</th>
                    <th>准确率图</th>
                    <th>机会水平图</th>
                    <th>差值图</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in group_df.iterrows():
        accuracy_map = row.get('accuracy_map_path')
        chance_map = row.get('chance_map_path')
        delta_map = row.get('delta_map_path')

        cond1_used = row.get('n_trials_cond1_used', row.get('n_trials_cond1', 'N/A'))
        cond2_used = row.get('n_trials_cond2_used', row.get('n_trials_cond2', 'N/A'))
        cond1_total = row.get('n_trials_cond1_total', cond1_used)
        cond2_total = row.get('n_trials_cond2_total', cond2_used)

        def format_count(used, total):
            if used == 'N/A' or used is None:
                return 'N/A'
            if total in ('N/A', None):
                return f"{used}"
            return f"{used} / {total}"

        balance_label = '是' if row.get('trial_balanced') else '否'

        def make_link(path):
            if path:
                name = Path(path).name
                return f'<a href="{path}">{name}</a>'
            return '缺失'

        table_html += f"""
                <tr>
                    <td><strong>{row.get('subject', 'N/A')}</strong></td>
                    <td>{row.get('contrast', 'N/A')}</td>
                    <td>{format_count(cond1_used, cond1_total)}</td>
                    <td>{format_count(cond2_used, cond2_total)}</td>
                    <td>{balance_label}</td>
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
                <p><strong>交叉验证:</strong> {cv_desc}</p>
                <p><strong>处理Mask:</strong> {mask}</p>
                <p><strong>机会水平:</strong> {chance:.2f}</p>
                <p><strong>样本平衡:</strong> {balance_label}</p>
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
        cv_desc=describe_cross_validation(config),
        mask=config.process_mask,
        lss_root=config.lss_root.name,
        mask_dir=config.mask_dir.name,
        results_dir=config.results_dir.name,
        chance=getattr(config, 'chance_level', 0.5),
        balance_label='已启用' if getattr(config, 'balance_trials', False) else '未启用',
        n_jobs=config.n_jobs,
        n_perm=config.n_permutations,
        within_perm=getattr(config, 'within_subject_permutations', 'N/A'),
        max_exact=getattr(config, 'max_exact_sign_flips_subjects', 'N/A'),
        alpha=config.alpha_level,
        random_state=config.cv_random_state
    )

    return config_html
