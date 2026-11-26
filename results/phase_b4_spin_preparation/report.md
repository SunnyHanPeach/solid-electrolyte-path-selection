# B4 自旋小试准备 · mp-1191903

为 BVSE 预处理的图像 0（端点）和图像 1（最高内部能量图像）各生成两套固定几何 SCF 输入。四份输入经 ASE 回读，元素顺序、晶胞和坐标与原始结构一致。图像 1 是由 BVSE 确定的试算位置，其 DFT 能量和路径峰值仍待计算。[预处理记录](../phase_b4_source_pilot/bvse_pilot.json) · [准备清单](manifest.json)

| 项目 | 设置 |
| --- | --- |
| 晶胞 | 87 原子，Li₁₅In₁₆S₄₈Si₈，固定晶胞和原子坐标 |
| 电荷与价电子数 | `tot_charge=0`，573 个价电子 |
| 自旋对照 | `nspin=1` 和 `nspin=2, tot_magnetization=1` μB |
| 赝势 | SSSP 1.3.0 PBE Efficiency 的 In、Li、S、Si USPP，逐文件验证发布 MD5 并记录 SHA256 |
| 数值设置 | PBE、Γ 点、50/400 Ry、0.10 eV 高斯展宽、电子阈值 10⁻⁸ Ry |

50/400 Ry 取四种元素的 SSSP 推荐值和 UPF 头部建议值的逐项最大值，由 In 决定。Li、Si 文件的 SHA256 与已完成的桥接相同。两个自旋分支采用同一电荷态和展宽，用于检验固定几何下的自旋敏感性。正式空位电荷态、最低能磁态以及本体系的 k 点与截断收敛仍待确认。[SSSP 发布数据](https://archive.materialscloud.org/records/rcyfm-68h65) · [QE 自旋参数说明](https://www.quantum-espresso.org/Doc/INPUT_PW.html#tot_magnetization) · [历史桥接清单](../phase_b4_bridge01/run_manifest.json)

## 执行与判定

1. 在单张 GPU 上依次计算端点的两个自旋分支。检查 SCF 收敛、电子数、磁矩和力输出，并记录用时和显存。两个端点作业通过检查后，再进入两项峰区作业。运行脚本使用桥接时的 QE 7.3.1 二进制和独立工作目录。
2. 比较两种自旋设置下的端点到峰区相对能量差，同时比较各图像全原子力分量 RMSE 和迁移 Li 矢量差范数。用 10 meV 和 0.02 eV/Å 作为小试判据。若触及门槛，先补充自旋分支和数值收敛验证，再决定端点弛豫与 NEB 设置。
3. 汇总实际 GPU 时间、主机配置、显存、SCF 迭代与失败消耗。四个保留测试材料继续封存。

准备过程只生成输入、校验清单和可上传的本地归档。QE 计算待启动。仓库保存生成代码和校验清单。
