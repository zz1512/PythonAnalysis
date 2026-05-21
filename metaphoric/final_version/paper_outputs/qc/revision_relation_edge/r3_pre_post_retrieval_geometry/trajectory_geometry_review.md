# Trajectory geometry review

Distance metrics are proxy metrics derived from available pair similarities, not voxel-level displacement vectors.
Vector alignment is marked unavailable unless voxel pattern inputs are later added.

## Descriptives

| network | condition | d_pre_post | d_post_retrieval | d_pre_retrieval | post_drop_raw | retrieval_rebound_raw | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpc_spatial | kj | 0.241945 | 0.213966 | 0.229188 | 0.0071559 | 0.0107325 | 910 |
| hpc_spatial | yy | 0.237366 | 0.212899 | 0.222075 | 0.0723572 | 0.058929 | 910 |
| semantic | kj | 0.252549 | 0.220619 | 0.248004 | -0.00633756 | 0.00819675 | 910 |
| semantic | yy | 0.25452 | 0.2318 | 0.233223 | 0.0616233 | 0.0615758 | 910 |

## Key model terms

| network | model | term | estimate | p | q_bh | status | n_obs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpc_spatial | d_pre_post | C(condition)[T.yy] | -0.0272141 | 0.381363 | 0.635605 | ok | 936 |
| hpc_spatial | d_pre_post | memory_ord | -0.0139531 | 0.591647 | 0.84521 | ok | 936 |
| hpc_spatial | d_pre_post | C(condition)[T.yy]:memory_ord | -0.00333673 | 0.922643 | 0.922643 | ok | 936 |
| hpc_spatial | d_post_retrieval | C(condition)[T.yy] | -0.0422928 | 0.121565 | 0.405215 | ok | 936 |
| hpc_spatial | d_post_retrieval | memory_ord | -0.019004 | 0.388781 | 0.544522 | ok | 936 |
| hpc_spatial | d_post_retrieval | C(condition)[T.yy]:memory_ord | 0.0393524 | 0.174709 | 0.436772 | ok | 936 |
| hpc_spatial | d_pre_retrieval | C(condition)[T.yy] | 0.00411033 | 0.887564 | 0.898671 | ok | 936 |
| hpc_spatial | d_pre_retrieval | memory_ord | -0.0230008 | 0.299632 | 0.749081 | ok | 936 |
| hpc_spatial | d_pre_retrieval | C(condition)[T.yy]:memory_ord | -0.00391678 | 0.898671 | 0.898671 | ok | 936 |
| hpc_spatial | return_to_pre_index | C(condition)[T.yy] | -0.0464032 | 0.226624 | 0.561203 | ok | 936 |
| hpc_spatial | return_to_pre_index | memory_ord | 0.00399677 | 0.892822 | 0.951258 | ok | 936 |
| hpc_spatial | return_to_pre_index | C(condition)[T.yy]:memory_ord | 0.0432692 | 0.280602 | 0.561203 | ok | 936 |
| hpc_spatial | retrieval_new_state_index | C(condition)[T.yy] | 0.0464032 | 0.226624 | 0.561203 | ok | 936 |
| hpc_spatial | retrieval_new_state_index | memory_ord | -0.00399677 | 0.892822 | 0.951258 | ok | 936 |
| hpc_spatial | retrieval_new_state_index | C(condition)[T.yy]:memory_ord | -0.0432692 | 0.280602 | 0.561203 | ok | 936 |
| hpc_spatial | post_drop_raw | C(condition)[T.yy] | 0.0314833 | 0.518251 | 0.746621 | ok | 936 |
| hpc_spatial | post_drop_raw | memory_ord | 0.051659 | 0.206812 | 0.517029 | ok | 936 |
| hpc_spatial | post_drop_raw | C(condition)[T.yy]:memory_ord | -0.0272324 | 0.610118 | 0.762647 | ok | 936 |
| hpc_spatial | retrieval_rebound_raw | C(condition)[T.yy] | 0.00553656 | 0.900853 | 0.900853 | ok | 936 |
| hpc_spatial | retrieval_rebound_raw | memory_ord | 0.0355353 | 0.326735 | 0.900853 | ok | 936 |
| hpc_spatial | retrieval_rebound_raw | C(condition)[T.yy]:memory_ord | 0.0202035 | 0.675603 | 0.900853 | ok | 936 |
| semantic | d_pre_post | C(condition)[T.yy] | -0.0177491 | 0.612224 | 0.636266 | ok | 936 |
| semantic | d_pre_post | memory_ord | -0.0446718 | 0.0765668 | 0.255223 | ok | 936 |
| semantic | d_pre_post | C(condition)[T.yy]:memory_ord | 0.0275254 | 0.466542 | 0.636266 | ok | 936 |
| semantic | d_post_retrieval | C(condition)[T.yy] | -0.00420724 | 0.883689 | 0.96046 | ok | 936 |
| semantic | d_post_retrieval | memory_ord | -0.0293002 | 0.157085 | 0.392713 | ok | 936 |
| semantic | d_post_retrieval | C(condition)[T.yy]:memory_ord | 0.05166 | 0.0973395 | 0.392713 | ok | 936 |
| semantic | d_pre_retrieval | C(condition)[T.yy] | -0.0127808 | 0.694252 | 0.792116 | ok | 936 |
| semantic | d_pre_retrieval | memory_ord | -0.00884255 | 0.712904 | 0.792116 | ok | 936 |
| semantic | d_pre_retrieval | C(condition)[T.yy]:memory_ord | -0.0172665 | 0.602307 | 0.792116 | ok | 936 |
| semantic | return_to_pre_index | C(condition)[T.yy] | 0.00857356 | 0.840864 | 0.956801 | ok | 936 |
| semantic | return_to_pre_index | memory_ord | -0.0204577 | 0.508126 | 0.956801 | ok | 936 |
| semantic | return_to_pre_index | C(condition)[T.yy]:memory_ord | 0.0689265 | 0.115098 | 0.575488 | ok | 936 |
| semantic | retrieval_new_state_index | C(condition)[T.yy] | -0.00857356 | 0.840864 | 0.956801 | ok | 936 |
| semantic | retrieval_new_state_index | memory_ord | 0.0204577 | 0.508126 | 0.956801 | ok | 936 |
| semantic | retrieval_new_state_index | C(condition)[T.yy]:memory_ord | -0.0689265 | 0.115098 | 0.575488 | ok | 936 |
| semantic | post_drop_raw | C(condition)[T.yy] | -0.0068761 | 0.897456 | 0.962363 | ok | 936 |
| semantic | post_drop_raw | memory_ord | 0.00194841 | 0.962363 | 0.962363 | ok | 936 |
| semantic | post_drop_raw | C(condition)[T.yy]:memory_ord | 0.0103374 | 0.859616 | 0.962363 | ok | 936 |
| semantic | retrieval_rebound_raw | C(condition)[T.yy] | -0.0180075 | 0.701621 | 0.855552 | ok | 936 |
| semantic | retrieval_rebound_raw | memory_ord | 0.0189406 | 0.581842 | 0.855552 | ok | 936 |
| semantic | retrieval_rebound_raw | C(condition)[T.yy]:memory_ord | 0.0229925 | 0.650016 | 0.855552 | ok | 936 |
