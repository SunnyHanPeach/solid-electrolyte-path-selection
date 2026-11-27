# 公开 NEB 数据复用审计

本轮核验旧索引、公开导向和完整数据包。FP-DeErr 的作者参考包 SHA256 已复核。移动离子由周期最小位移识别，按作者的源端点至 DFT 图像映射校验方向。独立性筛查范围为 LiTraj 的 867 个历史母晶胞、24 个 B5 候选母晶胞和受保护训练包的 278 个 D₀ 几何。未读取原保留测试标签。所有计算在本机完成。

## mp-1191903

目标母体为 Li₂In₂SiS₆，试点为 `mp-1191903_0_2_0_0_0` 的中性 Li 空位迁移。在 LiTraj 的 BVSE 索引中找到该材料的 3 条路径，包含目标路径。1,681 条最终 DFT 轨迹索引中该材料出现 0 次。FP-DeErr 和 621 条文献记录中均无同一元素体系。公开的 Li₂In₂SiS₆ 输运研究采用高温 AIMD 或机器学习分子动力学，可供材料背景核对。当前核查未取得相同空位、同一跳跃方向的完整 DFT-NEB 图像或作业日志。3.8 GB 的 MPLiTrj 全量构型库尚未逐帧排查，发布文档提供按划分的 XYZ 文件。该库包含中间优化构型，命中材料也需核对图像最终收敛和跃迁身份。[LiTraj](https://github.com/AIRI-Institute/LiTraj) · [Li₂In₂SiS₆ AIMD 公开记录](https://patents.justia.com/patent/10923763) · [涂层研究的材料表和方法](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c9e3b17122646f92a4e48e/original/computational-design-of-double-layer-cathode-coatings-in-all-solid-state-batteries.pdf)

## FP-DeErr 可用性与重叠

154 条均通过 7 张有序 DFT 图像及各图能量、全原子力完整性核对。109 条为锂迁移，45 条为钠迁移。锂迁移中 107 条源方向明确，来自 77 个 ICSD ID。移去 Li 后，沿用 H2 的匿名计量预筛及周期晶胞匹配参数。三类项目参考合并去重，排除同元素框架匹配后留下 87 条。再排除允许元素替换的匿名同构后留下 **62 条，45 个 ICSD ID**。与 621 条汇编另发现 4 条同元素骨架候选。所有重叠数按候选路径计，分组间可重复。同一框架标签还需查具体迁移位点和空位身份。62 条是可进一步冻结的候选，尚未做局域跃迁去重、独立家族确认或标签口径核验。[作者数据字段](https://github.com/mogroupumd/FP-DeErr/blob/main/Ion_migration_NEB/data/README.md) · [正式数据存档](https://doi.org/10.6084/m9.figshare.33332610)

| 所需数据 | 可复用范围 |
| --- | --- |
| 固定 DFT 几何上的正向能垒、相对能量和力评分 | 优先以 FP-DeErr 做独立来源的外部评价。核对参考 DFT 协议后再决定能否纳入主面板。 |
| MACE 训练构型 | LiTraj 的 MPLiTrj 含 DFT 优化过程中记录的能量、力和应力，适合做含去重及标签口径审计的训练库。 |
| 候选路径和预筛 | 621 条汇编有端点和文献能垒。20,320 条高通量路径由微调 CHGNet 工作流生成，可供候选发现。 |
| 实际 DFT 成本 | FP-DeErr 公开参考包的 DFT 设置块只保留历史来源说明。未找到逐作业时间、硬件、失败和续算明细。文献汇编主要提供计算参数。真实预算需要用本项目可计费记录标定。 |

[文献汇编](https://www.nature.com/articles/s41597-025-06196-x) · [CHGNet 候选数据库](https://www.cpsjournals.cn/en/article/doi/10.1088/1674-1056/ae88e2) · [代码和重叠结果](../../src/audit_public_neb_reuse.py)

下一步可在剩余候选中先锁定方向、路径近重复和目标家族，单列固定几何外部评价。公开标签可减少评价侧重复 DFT，并支撑回顾性选样比较。完整的同预算主张仍依赖一致口径下的实际作业成本，单条试点只能标定一个案例。
