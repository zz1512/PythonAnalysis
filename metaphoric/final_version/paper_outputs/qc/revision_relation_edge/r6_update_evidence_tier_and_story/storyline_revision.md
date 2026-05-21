# Storyline revision

## Core upgraded story

成功的隐喻学习不是简单增强词对相似性，而是在 post 阶段将 trained relation edge 从原有语义邻域中分化出来；这种分化后的 relation-edge 在 retrieval 阶段以任务依赖方式重新组织，并且后续被记住的 YY items 主要表现为 stronger prior post-stage separation。

English:

Successful metaphor learning does not simply increase similarity between associated concepts. Instead, it differentiates trained relation edges from their pre-existing semantic neighborhood, creating relation-edge representations that can be task-dependently reconstructed during retrieval.

## What the new revision analyses add

1. Material moderation: the new r1 models do not support a simple novelty-only or semantic-distance-only account of memory. Post separation remains strongly tied to pre similarity, and YY-specific novelty moderation is at most a support/boundary result rather than a new main mechanism.
2. High/low item profile: high-separation YY items are mainly distinguished by stronger pre-pair similarity profiles; novelty and memory differences are not stable enough to become a main claim.
3. Trajectory geometry: the scalar trajectory proxy does not provide strong evidence that retrieval is simply a return to pre. Because voxel vector alignment is unavailable in this pass, this should be written as boundary evidence, not as proof of new-state reconstruction.
4. Network coupling: semantic retrieval rebound strongly predicts hpc-spatial retrieval rebound. YY-specific coupling and coupling-memory interactions are weaker, so the claim should remain retrieval-stage inter-network coordination rather than YY-specific causal communication.
5. MVPA: learning and retrieval decoding are useful stage-state evidence; MVPA-behavior bridge and cross-role claims remain supplementary/boundary.

## Revised main-text implication

The existing core story remains intact. The revision analyses sharpen the boundaries: YY differentiation is not trivially explained by novelty, material distance, or a simple retrieval-post difference score. The strongest wording is therefore stagewise representational reorganization, with post-stage trained-edge differentiation as the central mechanism and retrieval-stage semantic-hpc coordination as task-driven reconstruction support.

## Evidence tier table

```tsv
tier	analysis	claim	support	boundary
A	Behavior YY memory advantage	existing core result	keep from result_new_meta_roi/result_final	
A	Learning condition-level semantic geometry	existing core result	keep from final_nc_converge	not item-specific causal trace
A	Post Step5C / edge specificity	existing core result	keep from final_nc_converge	do not call generic similarity drop strict pattern separation
A	Memory component	remembered YY = stronger prior post separation	keep from A4	not pure retrieval reinstatement
B	Material moderation post separation hpc_spatial	novelty/semantic distance boundary	novelty interaction estimate=0.2209531212182661, q=0.0593209189433548; pre interaction estimate=-0.018711989014771, q=0.751008174215647	support only if corrected; otherwise boundary
B	Material moderation memory hpc_spatial	material covariates do not explain memory advantage	YY x novelty memory estimate=0.0118710972696848, q=0.8971847762472173	weak/non-significant interactions are boundary evidence
B	Material moderation post separation semantic	novelty/semantic distance boundary	novelty interaction estimate=0.2076731759934664, q=0.0599166838121002; pre interaction estimate=0.049292898569339, q=0.4497536282329398	support only if corrected; otherwise boundary
B	Material moderation memory semantic	material covariates do not explain memory advantage	YY x novelty memory estimate=0.0043763978788599, q=0.9317437419770588	weak/non-significant interactions are boundary evidence
B	Trajectory geometry	proxy analysis distinguishes return-to-pre vs reconstruction	return_to_pre YY estimate=-0.0464031711926525, q=0.5612034677342788	proxy from pair similarities; vector alignment unavailable
A	Network coupling retrieval rebound	retrieval-stage semantic-hpc coordination	main coupling estimate=0.6977087330958573, q=5.569903446075754e-63; YY interaction estimate=-0.109478940832506, q=0.1957861756039661	not YY-specific unless interaction survives correction
B	Network coupling stage model	retrieval stage coupling trend	YY x retrieval estimate=0.1526393759813262, q=0.1279807088232886	stage coupling product proxy
B	Network coupling memory model	coupling-memory bridge	YY x retrieval coupling estimate=0.0119357794143296, q=0.7827897035578363	boundary if non-significant
C	MVPA pre	pre YY/KJ decoding	n_q_lt_05=0; best_q=0.0983135181081718	weak / not stable unless q-corrected rows appear
B	MVPA learning	learning YY/KJ decoding	n_q_lt_05=10; best_q=3.626772905739784e-05	stage-state evidence; not edge mechanism
B	MVPA post	post YY/KJ decoding	n_q_lt_05=2; best_q=0.03537395225361	partial retention, especially temporal pole if q-corrected
B	MVPA retrieval	run7 YY/KJ decoding	n_q_lt_05=4; best_q=3.036527290002737e-05	robust retrieval stage-state evidence
C	MVPA behavior_bridge	MVPA-behavior bridge	n_q_lt_05=0; best_q=nan	boundary unless q-corrected behavior bridge appears
```
