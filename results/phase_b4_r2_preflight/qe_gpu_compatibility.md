# QE 7.3.1 云端 GPU 兼容性复核

2026-11-26

在新开 RTX 4090 24 GB 上从 NVIDIA 发布清单重建历史 QE 环境。清单 SHA256 为 `e77e134422f3b242f87056dd2dc0789c931021afe862178c660dc3867f437f03`。恢复的 `pw.x` SHA256 为 `9d4510c4aff6c7340725bdb9244aef5b6011fdcd91f6a7197f526bdb1f0cc702`，与历史 RTX 3090 一致。使用已校验的 2×2×2 端点输入和原起始电荷密度，驱动为 595.71.05。 [历史运行记录](../phase_b4_bridge01/run_manifest.json)

| 短测 | 启动时间，北京时间 | 求解器耗时 | 退出码 | 日志结果 |
| --- | --- | ---: | ---: | --- |
| 默认方式 | 20:18:45 | 5 秒 | 1 | OpenACC 在 `interp_vloc` 初始化时报需为 `cc89` 重编译。 |
| `CUDA_FORCE_PTX_JIT=1` | 20:20:16 | 不足 1 秒 | 127 | 设备函数查找失败。`cuobjdump -lptx` 未在该可执行文件中找到 PTX。 |

两次均未进入 SCF。写盘对照和网格验收仍待完成。末次查询确认无 QE 进程。表中耗时仅为求解器耗时，另行核算租用实例的计费时间。当前最短复现路线是复用已跑通的 RTX 3090 与原二进制。另行适配 4090 需要包含 `cc89` 的构建和数值对照。 [已有 3090 结果](../phase_b4_numeric_grid_recovered/report.md) · [NVIDIA Ada 兼容性测试](https://docs.nvidia.com/cuda/archive/12.6.3/ada-compatibility-guide/index.html)

本地日志归档 SHA256

- 默认短测 `ef1256c10cdbbd7cb647bdec52af81464cb209ca187f44bc14bf4c2e78d552f5`
- PTX 短测 `a63ab7515121347189367571915f1cc163cc6d44e71fb5b4f542651e37058c8a`
