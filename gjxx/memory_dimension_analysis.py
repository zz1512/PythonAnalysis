from __future__ import annotations


def main() -> None:
    print("Dimension analysis Python entry scripts:")
    print(r"1) python run_remembered_forgotten_dimension.py")
    print(r"   MATLAB mapping: only_consider_remembered_items.m (extended to forgotten)")
    print(r"2) python run_memory_group_dimension.py")
    print(r"   MATLAB mapping: grouped HSC/LSC ROI comparison by memory accuracy")
    print(r"3) python run_story_roi_suite.py")
    print(r"   MATLAB mapping: hl_patternbutnotrdm.m, hl.m, demeantest*.m,")
    print(r"                   get_voxel_mask.m, sametrialnumber_maskvoxel.m,")
    print(r"                   sametrialnumber_first_3_pc.m")
    print(r"4) python run_story_behavior_suite.py")
    print(r"   MATLAB mapping: across_sub_dim.m, Untitled2.m, Untitled3.m")
    print(r"5) python run_story_searchlight_suite.py")
    print(r"   MATLAB mapping: searchlight_RD.m, my_var_measure.m, pairedttest.m")
    print(r"6) python run_story_connectivity_suite.py")
    print(r"   MATLAB mapping: stage1_pca_conn_searchlight.m, my_measure.m")


if __name__ == "__main__":
    main()
