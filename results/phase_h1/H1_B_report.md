# H1-B · 前瞻候选审查清单

**审查清单共 10 条，包含 5 条满足冻结分歧门槛的路径和 5 条单列的分层对照。** 冻结主名单达到 5 条，低于原先预期的 8 条。主门槛原样保留。本阶段新增的对照按较小排序分歧和分层覆盖确定，选择未使用收益标签或 H1-A 的拟合结果。[完整十条及入选理由](structure_output/prospective_review_panel.csv) · [冻结主名单](structure_output/prospective_shortlist.csv) · [对照补充分层协议](panel_controls_protocol.json)

## 24 条候选的复核

- GPU 逐位复算 SOAP、SOAP＋Activation、Activation-LCMD、Activation-PV 的 24 条 Mean 排序，四套完全重现 B5-R。原七套排序文件保留原哈希。SOAP 与 SOAP＋Activation 的前五成员完全一致，原排序顺序保留。匹配报告另有候选间全结构特征距离和迁移 Li 局部曲线距离。[GPU 排序](gpu_output/gpu_verified_rankings.csv) · [原排序](../phase_b5_r/independent_candidate_ranking.csv)
- 与 867 个历史材料及 24 个候选母晶胞查重。匿名化学计量预筛后做 129 组周期晶胞匹配，找出 6 对匿名近结构，涉及 4 条候选，均与历史材料相关，因此这 4 条从审查清单排除。匹配允许元素替换，同构线索不能证明材料化学等价。全历史 1,681 条路径的四最近阴离子径向筛查标记 32 条相似配对。该筛查按图像索引重采样，信息维度有限。按反应坐标复核的迁移 Li SOAP 覆盖候选和 14 条开发路径，标出 2 对候选间相似，没有把同一对纳入面板。[晶胞匹配](structure_output/parent_structure_matches.csv) · [径向筛查](structure_output/radial_historical_flags.csv) · [Li SOAP 复核](structure_output/li_soap_screen_flags.csv)

## 首批前瞻审查面板

| 角色 | 路径 | 入选依据 |
| --- | --- | --- |
| 高分歧 | mp-23497_0_5_0_0_0 | SOAP 第 1，Activation-PV 第 11 |
| 高分歧 | mp-1020013_0_4_0_0_0 | Activation-LCMD 和 PV 均第 2，SOAP 第 12 |
| 高分歧 | mp-1210789_0_3_0_0_0 | Activation-PV 第 7，LCMD 第 14，硫化物对照 |
| 高分歧 | mp-1111659_2_2_0_0_1 | Activation-LCMD 第 1，PV 第 10 |
| 高分歧 | mp-28341_0_1_0_1_0 | Activation-PV 第 5，LCMD 第 13 |
| 分层对照 | mp-1105934_3_0_1_0_0 | 氧化物，四方案名次跨度 3，存在径向相似标记 |
| 分层对照 | mp-1222483_0_0_1_1_0 | 四方案均在前三，保留共识高位对照 |
| 分层对照 | mp-768514_1_3_0_0_0 | 氧化物，四方案均在第 21–24 名 |
| 分层对照 | mp-768361_3_4_0_1_0 | 硫化物，四方案均进前五，存在径向相似标记 |
| 分层对照 | mp-755254_0_0_0_1_0 | 复合体系，四方案名次跨度 4 |

合计卤化物 3 条、硫化物 2 条、氧化物 2 条、复合类 3 条。H1 的家族分层是化学粗分组，匿名晶胞与局部筛查用于控制可发现的重复。mp-1210789 等仍有径向相似标记，进入正式实验前需人工核查。此表记录无标签候选的优先级与对照角色。独立评估面板与真实 DFT 预算仍待审核。[候选全体审计](structure_output/candidate_ranking_and_audit.csv) · [评估面板缺口](../phase_b5_r/evaluation_panel_readiness.md)
