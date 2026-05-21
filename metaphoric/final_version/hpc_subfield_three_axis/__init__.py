"""海马亚区切分 + 三轴叙事 + R hpc MDS 可视化。

本模块与 reviewer_supp/ 和 nc_converge/ 平级共存，输出沙箱：
    paper_outputs/qc/hpc_subfield_three_axis/<module>/

硬约束：
- 不修改任何既有脚本与既有输出；
- 所有写入路径经过 safe_output_path() 检查，已存在即报错；
- result_new_meta_roi.md 只允许末尾追加 §29。
"""
