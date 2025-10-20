# report_generator.py
# HTML报告生成模块

import base64
import numpy as np
from pathlib import Path
from utils import log

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
        # 使用变量来避免f-string中的反斜杠问题
        onerror_script = "this.style.display='none'; this.nextElementSibling.style.display='block';"
        return f'<img src="data:image/png;base64,{image_b64}" style="width:100%; max-width:1200px; height:auto; border:1px solid #ddd; border-radius:8px; margin:20px 0;" alt="{alt_text}" onerror="{onerror_script}" /><div style="display:none; color:red; padding:20px; border:1px solid #ddd; border-radius:8px; background:#f8f8f8; text-align:center;">⚠️ {alt_text}加载失败<br><small>图片文件可能不存在或损坏</small></div>'
    else:
        return f'<div style="color:red; padding:20px; border:1px solid #ddd; border-radius:8px; background:#f8f8f8; text-align:center;">⚠️ {alt_text}加载失败<br><small>请检查图片生成是否成功</small></div>'

def generate_html_report(group_df, stats_df, config, main_viz_b64=None, individual_viz_b64=None):
    """生成增强的中文HTML报告"""
    log("生成HTML报告", config)
    
    # 如果没有提供图片数据，尝试从文件加载
    if main_viz_b64 is None or individual_viz_b64 is None:
        main_viz_path = config.results_dir / "enhanced_analysis_visualizations.png"
        individual_viz_path = config.results_dir / "individual_roi_detailed_plots.png"
        
        if main_viz_b64 is None:
            main_viz_b64 = image_to_base64(main_viz_path) if main_viz_path.exists() else None
        if individual_viz_b64 is None:
            individual_viz_b64 = image_to_base64(individual_viz_path) if individual_viz_path.exists() else None
    
    # 按效应排序（优先显示准确率>50%且显著的结果）
    sorted_df = stats_df.copy()
    sorted_df['sort_key'] = (
        sorted_df['significant'].astype(int) * 1000 +  # 显著性权重
        (sorted_df['mean_accuracy'] > 0.5).astype(int) * 100 +  # 超过随机水平权重
        np.abs(sorted_df['cohens_d']) * 10  # 效应量权重
    )
    sorted_df = sorted_df.sort_values('sort_key', ascending=False)
    
    # 计算摘要统计
    total_rois = len(stats_df)
    significant_rois = len(stats_df[stats_df['significant']])
    above_chance_rois = len(stats_df[stats_df['mean_accuracy'] > 0.5])
    max_accuracy = stats_df['mean_accuracy'].max()
    best_roi = stats_df.loc[stats_df['mean_accuracy'].idxmax(), 'roi']
    
    # 生成详细分析函数
    def generate_detailed_analysis():
        if stats_df.empty:
            return "<p>无可用数据进行分析</p>"
        
        # 最佳结果分析
        best_result = stats_df.loc[stats_df['mean_accuracy'].idxmax()]
        best_roi_data = group_df[group_df['roi'] == best_result['roi']]['mean_accuracy']
        best_std = best_roi_data.std()
        
        analysis_html = f"""
        <div class="highlight-box">
            <h4>🏆 最佳结果分析</h4>
            <p><strong>{best_result['roi']}</strong> 表现最佳：</p>
            <ul>
                <li>平均准确率：<strong>{best_result['mean_accuracy']:.3f} ± {best_std:.3f}</strong></li>
                <li>标准误：<strong>{best_result['sem_accuracy']:.3f}</strong></li>
                <li>95%置信区间：<strong>[{best_result['ci_lower']:.3f}, {best_result['ci_upper']:.3f}]</strong></li>
                <li>被试数量：<strong>{best_result['n_subjects']}</strong></li>
                <li>效应量：<strong>Cohen's d = {best_result['cohens_d']:.3f}</strong></li>
                <li>统计显著性：<strong>{'显著' if best_result['significant'] else '不显著'}</strong> (p = {best_result['p_value_permutation']:.3f})</li>
            </ul>
        </div>
        """
        
        # 显著结果列表
        significant_results = stats_df[stats_df['significant']]
        if not significant_results.empty:
            analysis_html += """
            <div class="summary-box">
                <h4>✅ 显著结果列表</h4>
                <ul>
            """
            for _, result in significant_results.iterrows():
                roi_data = group_df[group_df['roi'] == result['roi']]['mean_accuracy']
                roi_std = roi_data.std()
                analysis_html += f"""
                    <li><strong>{result['roi']}</strong>: {result['mean_accuracy']:.3f} ± {roi_std:.3f} 
                    (p = {result['p_value_permutation']:.3f}, d = {result['cohens_d']:.3f})</li>
                """
            analysis_html += "</ul></div>"
        
        # 趋势分析
        above_chance = stats_df[stats_df['mean_accuracy'] > 0.5]
        accuracy_range = stats_df['mean_accuracy'].max() - stats_df['mean_accuracy'].min()
        mean_effect_size = np.abs(stats_df['cohens_d']).mean()
        
        analysis_html += f"""
        <div class="trend-box">
            <h4>📈 整体趋势分析</h4>
            <ul>
                <li>超过随机水平的ROI：<strong>{len(above_chance)}/{total_rois}</strong> ({len(above_chance)/total_rois*100:.1f}%)</li>
                <li>准确率范围：<strong>{stats_df['mean_accuracy'].min():.3f} - {stats_df['mean_accuracy'].max():.3f}</strong> (变异度: {accuracy_range:.3f})</li>
                <li>平均效应量：<strong>{mean_effect_size:.3f}</strong></li>
                <li>统计显著的ROI：<strong>{significant_rois}/{total_rois}</strong> ({significant_rois/total_rois*100:.1f}%)</li>
            </ul>
        </div>
        
        <div class="section">
            <h3>📊 统计方法说明</h3>
            <div class="summary-box">
                <h4>🔬 组水平检验</h4>
                <ul>
                    <li><strong>单样本t检验</strong>：检验组平均准确率是否显著高于50%随机水平</li>
                    <li><strong>置换检验</strong>：非参数方法，通过{config.n_permutations}次置换生成零假设分布</li>
                    <li><strong>效应量</strong>：Cohen's d衡量实际效应大小，0.2/0.5/0.8对应小/中/大效应</li>
                    <li><strong>多重比较</strong>：根据专家建议，使用置换检验的非参数方法，无需额外校正</li>
                </ul>
            </div>
        </div>
        """
        
        return analysis_html
    
    # 生成优化建议函数
    def generate_optimization_suggestions():
        suggestions = [
            {
                "title": "🎯 ROI优化建议",
                "content": "考虑使用功能定位或个体化ROI定义，可能提高分类精度和个体间一致性。"
            },
            {
                "title": "📊 样本量评估",
                "content": f"当前分析包含{len(group_df['subject'].unique())}个被试，建议进行功效分析评估样本量充分性。"
            },
            {
                "title": "🔄 交叉验证策略",
                "content": f"当前使用{config.cv_folds}折交叉验证，可考虑留一法或嵌套交叉验证进一步验证结果稳定性。"
            },
            {
                "title": "⚡ 特征选择",
                "content": "可尝试特征选择方法（如递归特征消除）来识别最具判别性的体素。"
            },
            {
                "title": "🧠 多模态整合",
                "content": "考虑整合结构像或其他功能指标，可能提供更全面的神经机制理解。"
            },
            {
                "title": "🔄 置换检验优化",
                "content": "可调整置换次数以平衡精度和计算效率。"
            }
        ]
        
        suggestions_html = """
        <div class="suggestions-grid">
        """
        
        for suggestion in suggestions:
            suggestions_html += f"""
            <div class="suggestion-box">
                <h4>{suggestion['title']}</h4>
                <p>{suggestion['content']}</p>
            </div>
            """
        
        suggestions_html += "</div>"
        return suggestions_html
    
    # 生成详细的结果表格HTML
    def generate_results_table(df):
        if df.empty:
            return "<p>无符合条件的结果</p>"
        
        table_html = """
        <table class="results-table">
            <thead>
                <tr>
                    <th>排名</th>
                    <th>ROI区域</th>
                    <th>平均准确率±标准差</th>
                    <th>标准误</th>
                    <th>t统计量</th>
                    <th>置换检验p值</th>
                    <th>Cohen's d</th>
                    <th>95%置信区间</th>
                    <th>被试数</th>
                    <th>显著性</th>
                    <th>效应解释</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            # 计算该ROI的标准差
            roi_data = group_df[group_df['roi'] == row['roi']]['mean_accuracy']
            roi_std = roi_data.std()
            
            # 效应量解释
            d_value = abs(row['cohens_d'])
            if d_value < 0.2:
                effect_size = "微小效应"
            elif d_value < 0.5:
                effect_size = "小效应"
            elif d_value < 0.8:
                effect_size = "中等效应"
            else:
                effect_size = "大效应"
            
            # 显著性标记
            significance = "✓ 显著" if row['significant'] else "不显著"
            row_class = "significant-row" if row['significant'] else ""
            
            table_html += f"""
                <tr class="{row_class}">
                    <td>{idx}</td>
                    <td><strong>{row['roi']}</strong></td>
                    <td>{row['mean_accuracy']:.3f} ± {roi_std:.3f}</td>
                    <td>{row['sem_accuracy']:.3f}</td>
                    <td>{row['t_statistic']:.3f}</td>
                    <td>{row['p_value_permutation']:.4f}</td>
                    <td>{row['cohens_d']:.3f}</td>
                    <td>[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]</td>
                    <td>{row['n_subjects']}</td>
                    <td>{significance}</td>
                    <td>{effect_size}</td>
                </tr>
            """
        
        table_html += "</tbody></table>"
        return table_html
    
    # 生成完整的HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>fMRI MVPA ROI分析详细报告</title>
        <style>
            body {{
                font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
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
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: bold;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #007bff;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
                margin-bottom: 5px;
            }}
            .stat-label {{
                color: #666;
                font-size: 0.9em;
            }}
            .results-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.85em;
                overflow-x: auto;
            }}
            .results-table th {{
                background-color: #343a40;
                color: white;
                padding: 10px 6px;
                text-align: center;
                font-weight: bold;
                font-size: 0.8em;
                word-wrap: break-word;
            }}
            .results-table td {{
                padding: 8px 6px;
                text-align: center;
                border-bottom: 1px solid #dee2e6;
                word-wrap: break-word;
                font-size: 0.8em;
            }}
            .results-table tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .significant-row {{
                background-color: #d4edda !important;
                font-weight: bold;
            }}
            .highlight-box {{
                background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #e17055;
            }}
            .summary-box {{
                background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #00b894;
            }}
            .trend-box {{
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #2196f3;
            }}
            .suggestions-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .suggestion-box {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
            }}
            .suggestion-box h4 {{
                color: #28a745;
                margin-top: 0;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .section h2 {{
                color: #343a40;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .section h3 {{
                color: #495057;
                margin-top: 25px;
            }}
            ul {{
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 8px;
            }}
            .config-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}
            .config-item {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 3px solid #007bff;
            }}
            .config-item strong {{
                color: #007bff;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 fMRI MVPA ROI分析详细报告</h1>
                <p>基于置换检验的组水平统计分析结果</p>
            </div>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-number">{total_rois}</div>
                    <div class="stat-label">分析ROI总数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{significant_rois}</div>
                    <div class="stat-label">显著ROI数量</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{above_chance_rois}</div>
                    <div class="stat-label">超过随机水平ROI</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{max_accuracy:.3f}</div>
                    <div class="stat-label">最高准确率 ({best_roi})</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 可视化结果总览</h2>
                <p>以下图表展示了所有ROI的分析结果，包括分类准确率、效应量分布、统计显著性等多个维度的信息：</p>
                
                {generate_image_html(main_viz_b64, '主要可视化图表')}
                
                <div class="summary-box">
                    <h4>📋 图表说明</h4>
                    <ul>
                        <li><strong>左上：分类准确率条形图</strong> - 显示各ROI的平均准确率，绿色表示显著，红色表示不显著，黑点为个体数据</li>
                        <li><strong>右上：显著ROI比较</strong> - 横向比较所有显著ROI的效果，便于识别最佳区域</li>
                        <li><strong>左中：效应量分布</strong> - 展示各ROI的Cohen's d效应量，颜色编码表示效应大小</li>
                        <li><strong>右中：显著性分布饼图</strong> - 显示显著与不显著ROI的比例</li>
                        <li><strong>左下：准确率分布直方图</strong> - 所有数据点的准确率分布情况</li>
                        <li><strong>右下：ROI间相关性热图</strong> - 展示不同ROI间准确率的相关性模式</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 单个ROI详细分析</h2>
                <p>以下为每个ROI的详细分析图表，包含个体数据点、统计信息和置信区间：</p>
                
                {generate_image_html(individual_viz_b64, '单个ROI详细图表')}
                
                <div class="summary-box">
                    <h4>📋 单个ROI图表说明</h4>
                    <ul>
                        <li><strong>条形图</strong>：显示平均分类准确率，绿色表示显著，红色表示不显著</li>
                        <li><strong>误差线</strong>：表示标准误，反映数据的变异性</li>
                        <li><strong>散点</strong>：显示每个被试的个体数据点，展现个体差异</li>
                        <li><strong>红色虚线</strong>：50%随机水平参考线</li>
                        <li><strong>统计信息框</strong>：包含均值±标准误、t统计量、效应量和95%置信区间</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>🏆 主要结果（按效应排序）</h2>
                <p><strong>排序规则：</strong>优先显示准确率>50%且具有显著效应的ROI，按显著性和效应量排序</p>
                {generate_results_table(sorted_df)}
            </div>
            
            {generate_detailed_analysis()}
            
            <div class="section">
                <h2>💡 优化建议</h2>
                {generate_optimization_suggestions()}
            </div>
            
            <div class="section">
                <h2>📊 完整统计结果</h2>
                <p>以下为所有ROI的完整统计结果：</p>
                {generate_results_table(stats_df.sort_values('mean_accuracy', ascending=False))}
            </div>
            
            <div class="section">
                <h2>📝 方法学说明</h2>
                <div class="highlight-box">
                    <h4>🔬 分析流程</h4>
                    <ol>
                        <li><strong>数据预处理</strong>：LSS（Least Squares Separate）模型提取单试次beta值</li>
                        <li><strong>特征提取</strong>：从每个ROI提取BOLD信号特征</li>
                        <li><strong>分类分析</strong>：使用支持向量机（SVM）进行二分类</li>
                        <li><strong>交叉验证</strong>：{config.cv_folds}折分层交叉验证评估分类性能</li>
                        <li><strong>统计检验</strong>：组水平单样本t检验（vs 50%随机水平）</li>
                        <li><strong>置换检验</strong>：组水平置换检验评估统计显著性</li>
                    </ol>
                </div>
                
                <div class="summary-box">
                    <h4>📈 统计指标解释</h4>
                    <ul>
                        <li><strong>分类准确率</strong>：正确分类的试次比例，>50%表示超过随机水平</li>
                        <li><strong>t统计量</strong>：组水平t检验统计量，绝对值越大效应越强</li>
                        <li><strong>Cohen's d</strong>：标准化效应量，0.2/0.5/0.8分别对应小/中/大效应</li>
                        <li><strong>95%置信区间</strong>：真实准确率的可能范围</li>
                        <li><strong>置换检验p值</strong>：基于置换检验的p值，<0.05为显著</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>⚠️ 局限性与注意事项</h2>
                <ul>
                    <li>结果的泛化性需要在独立样本中验证</li>
                    <li>ROI定义可能影响结果，建议尝试不同的ROI划分方案</li>
                    <li>分类准确率受到噪声、个体差异等多种因素影响</li>
                    <li>统计显著性不等同于实际意义，需结合效应量综合判断</li>
                    <li>置换检验提供了非参数的显著性评估方法</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存报告
    report_path = config.results_dir / "enhanced_analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log(f"增强的中文HTML报告已生成：{report_path}", config)
    
    # 同时生成简化版本的CSV报告
    summary_df = sorted_df[[
        'roi', 'mean_accuracy', 'sem_accuracy', 't_statistic', 
        'p_value_permutation', 'cohens_d', 
        'significant', 'ci_lower', 'ci_upper'
    ]].copy()
    summary_df.columns = [
        'ROI区域', '分类准确率', '标准误', 't统计量', 
        '置换检验p值', 'Cohen\'s d', 
        '是否显著', '置信区间下限', '置信区间上限'
    ]
    summary_df.to_csv(config.results_dir / "results_summary_chinese.csv", 
                      index=False, encoding='utf-8-sig')
    
    return report_path