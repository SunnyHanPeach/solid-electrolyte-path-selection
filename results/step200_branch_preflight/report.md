# 八路径追加的文献核对与第 200 步分支预检

2026-11-27。核对的核心假设是，首段八路径训练结束后，继续暴露于所追加的路径是否会进一步抬高正向迁移能垒 MAE。新一轮实施前，已有结果足以预检训练批次和分解已归档的力预测。本次未启动训练或 DFT。

## 文献证据与适用范围

| 研究 | 对本项目的意义 |
| --- | --- |
| Dembitskiy 等的 LiTraj | 表 5 中，无过渡金属子集 MACE 的 0.07→0.10 eV 来自不按划分限制的 nebDFT2k 轨迹汇总，独立测试见补表 S7。只用空位端点构型的软化是 SevenNet 的单独消融。两项证据说明训练数据构成和微调会影响迁移能垒。 [论文表 5 及图 3](https://www.nature.com/articles/s41524-025-01571-z) |
| Deng 等的通用势软化 | 第一作者为 Deng。研究涵盖 Mg 跃迁的 ApproxNEB 及以两端参照的 KRA 能垒。为高能区下移提供先例。我们的固定图像数据尚不足以判断局部曲率或证明同一成因。 [作者公开论文](https://perssongroup.lbl.gov/papers/deng-2025_systematic_softening.pdf) |
| Mohandas 等、Kim 等、Tompa 等 | 钢体系的跨材料保持、固态电解质的 Replay/EWC 与微调策略比较均说明微调可能影响已有能力。Kim 的 Li₆PS₅I 例子来自 600 K、扩散活性较低的采样，以扩散率评估。目前的本项目终点是迁移能垒，且已固定的是本地 D₀ 的曝光。 [Mohandas](https://link.springer.com/article/10.1007/s11669-025-01225-z) · [Kim](https://www.nature.com/articles/s41524-025-01895-w) · [Tompa 预印本](https://arxiv.org/abs/2606.12704) |
| Wong 和 Yang、Ge 等 | Wong 与 Yang 对照的是液相材料特定轨迹的并行与迭代采样。Ge 等展示的是甘氨酸势函数的任务加权。两者支持检验训练样本构成和任务指标。MACE 指南的固定权重讨论针对训练阶段切换。 [Wong 和 Yang](https://pmc.ncbi.nlm.nih.gov/articles/PMC13379080/) · [Ge 等](https://doi.org/10.1021/acs.jpclett.4c00746) · [MACE 指南](https://mace-docs.readthedocs.io/en/latest/guide/finetuning_guidance.html) |
| Goryaeva 等、Wu 等 | 2020 年已有用全原子力积分重构缺陷迁移剖面的先例。2026 年铝 MACE 预印本已经有同更新数的历史回放对照、监督消融和相对能量约束。项目可能的贡献在于完整锂迁移路径、严格匹配的 D₀ 暴露、同检查点后续训练干预，以及独立材料上的路径收益。 [Goryaeva 等](https://www.nature.com/articles/s41467-020-18282-2) · [Wu 等预印本 §2.1、§2.3](https://arxiv.org/html/2610.35653v1) |

仅按审阅到的近邻文献界定项目差异，保留查新范围。软化、遗忘、匹配回放和力积分已有先例。原稿引用的“7.28 倍梯度范数”待补训练日志及统计定义。已定位到的 7.2808 出现在第二批权重实验的高能区轮廓 MAE 结果中，单位是 meV。 [第二批第 400 步原表](../force_weighting/batch2/summary_400.json)

## 已有积分与新增分段

全原子力积分已有结果。此次从已验证的 12 份力数组按同一初始几何与 DFT 最高内部图像拆分，通过周期最小位移把配对力差给出的有符号能量变化分解为迁移 Li 和其余原子两部分。第 400 步上坡段，主面板（91 个材料）均值为 −17.84 和 −3.96 meV。辅助面板（87 个迁移 Li 索引通过复核的材料）为 −25.24 和 −22.44 meV。两段及三个检查点均在表中。记录是沿固定坐标的做功分解。它能说明有符号力变化的位置和原子贡献，训练中的因果链仍需使用分支干预。 [原全原子审计](../followup_cpu_audit_20261127/force_audit/report.md) · [分段汇总](force_work_components_summary.csv)

对同一组固定图像，令旧预测峰值图像为 \(p\)，新预测峰值为 \(q\)。正向能垒变化为 \(\delta E_p-\delta E_0+(E_q^{new}-E_p^{new})\)，最后一项是换峰贡献。固定 DFT 峰位的积分和各模型取最大值的能垒保留为两项。

## 推荐实验

每个种子从原追加组第 200 步的同一模型参数、Adam 和共同随机状态分叉。保持后续更新数量、D₀ 主排程和原子数。A 走既有 Replay，B 继续原八路径追加，C 对追加内部图像按相对能量降权（α=0.75），D 使用逐路径匹配剂量的均匀降权。B/C/D 共享追加批次。两种子各有 24 个不同的插入批次，覆盖八条路径。按第 201–400 步重新匹配后，两种子的八条路径均匀系数均与 807 步全程系数不同。 [排程与输入哈希](continuation_protocol_inputs.json) · [逐路径剂量](continuation_dose.csv)

第 400 步的 B−A 衡量继续追加相对于回放的效应，C−B 与 D−B 衡量减少力监督的效果，C−D 衡量等剂量下按能量分配的增量。每组报告正向能垒 MAE、逐材料配对区间和分种子结果。A 在 Adam 交叉中有同一初始状态的回放结果，只有对齐随机状态、硬件、运行时、步数及评分口径后才复用。符合复用时须补 6 组续训。当前历史面板用于机制验证。采用新权重为正式协议或评价真实 DFT 预算下的选样优势，仍依赖冻结的独立材料评估。 [现有 Adam 对照](../adam_cross_factorial/report.md) · [已复验的等剂量结果](../force_weighting/batch2/README.md)
