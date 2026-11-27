# Adam 交叉续训短测

两个训练因素为第 200 步的模型参数来源和 Adam 状态来源。四组统一读取相同种子冻结的 201–400 步 Replay 批次，以 Replay 检查点的随机状态为共同起点。第 200、201、210、250、300、400 步保留盲推理，续训权重及优化器状态可从最近保存点恢复。

本机 CPU 对种子 17 的四组各执行一次更新和五张图像评分。同参数的两组在更新前得到完全相同的推理结果、损失和梯度范数，四组的 Adam 步数均升到 201。交换 Adam 后，更新后预测字节有差异。短测核对的是接线和可执行性。完整实验与 GPU 短测尚未运行。

开机后先用相同运行环境短测 CUDA 显存和速度，短测目录与正式目录分开。两种子可在匹配的设备上按种子分配，也可在一张卡上依次执行。所有论文评分使用此前冻结的两个历史面板，保留逐种子结果。

    python run_adam_cross.py frozen_bundle.zip step200_states.zip cuda_smoke --seed 17 --device cuda --smoke
    python run_adam_cross.py frozen_bundle.zip step200_states.zip full --seed 17 --device cuda
    python run_adam_cross.py frozen_bundle.zip step200_states.zip full --seed 29 --device cuda

启动包和权重保留于本地。版本库记录四组短测结果及启动包哈希。
