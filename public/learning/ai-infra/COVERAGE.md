# AIInfraGuide 前置基础覆盖矩阵

这份矩阵用于核对“原教程讲了什么、本课程在哪里讲、如何验收”。课程不是逐段改写原文，而是把知识补成“概念 → 推导 → 代码 → 实验 → 过关任务”的闭环。

| 原教程范围 | 本课程单元 | 可执行练习 / 验收 |
|---|---|---|
| 第 1 章：编程语言基础总览 | 01 跨层地图 | 画调用链；按证据顺序定位故障 |
| Python 基础、工程语义与运行模型 | 02 Python 工程语义与并发 | `python_semantics_lab.py`；解释引用、可变性、上下文管理器和并发选择 |
| C/C++、内存与跨语言扩展 | 03 C++ 边界 | `cpp_raii`；完成 CMake 构建并说明 RAII、ABI 和生命周期 |
| Linux、环境管理与工具链 | 04 Linux 与环境 | `environment_audit.py`；交付可复现环境快照和动态库诊断顺序 |
| 第 2 章：数学基础总览 | 05–06 数学与数值 | 手算 shape、FLOPs、访存量、梯度和数值稳定性 |
| 线性代数、张量、矩阵乘 | 05 张量、GEMM 与性能账本 | `tensor_math_lab.py`；完成 GEMM shape/FLOPs/算术强度账本 |
| 概率、微积分、反向传播、混合精度 | 06 概率、自动微分与数值稳定 | `numerics_lab.py`；证明稳定 softmax、非结合性与精度风险 |
| 第 3 章：Transformer 总览和快速入门 | 07–09 Transformer 数据流 | 从 token 到 logits 画出完整 shape 流 |
| Embedding、Token、Q/K/V、Self-Attention | 07 Self-Attention | `attention_lab.py`；仅用基础数组实现因果注意力并做 shape 检查 |
| Multi-Head、MQA/GQA、Mask 与缩放点积 | 07 Self-Attention | 比较 MHA/MQA/GQA 的 KV 规模与带宽代价 |
| FFN、激活、残差、归一化、位置编码 | 08 Decoder Block | `decoder_block_ledger.py`；核对 RMSNorm、RoPE、SwiGLU 和残差路径 |
| 参数量、激活量与并行切分入口 | 08 Decoder Block | 交付单层参数账本并指出张量并行切分位置 |
| 自回归生成、采样、Prefill/Decode、KV Cache | 09 LLM 推理 | `kv_cache_calculator.py`；计算 KV 显存并解释延迟/吞吐权衡 |
| 第 4 章：PyTorch 框架与快速入门 | 10 PyTorch 工程 | `pytorch_training_lab.py`；跑通训练、autograd、AMP 和 profiler |
| Tensor 元数据、Module、训练循环、显存 | 10 PyTorch 工程 | 检查 dtype/device/stride/grad；提交 profiler 证据 |
| 第 5 章：GPU 硬件概论、基础与演进 | 11 GPU 架构 | `roofline_lab.py`；区分算力瓶颈、带宽瓶颈和 launch 开销 |
| SM、Warp、Tensor Core、存储层次、Occupancy | 11 GPU 架构 | 用 Roofline 给算子分类，并提出能被测量验证的优化假设 |
| 第 6 章：集合通信基础与通信原语 | 12 集合通信与 NCCL | `collectives_lab.py`；验证 collective 语义并计算 ring 通信量 |
| AllReduce、AllGather、ReduceScatter、Ring/Tree | 12 集合通信与 NCCL | 根据消息大小和拓扑选择算法；识别 overlap 条件和死锁风险 |
| 六章综合能力 | 13 毕业项目 | 分析 `capstone_case`，提交瓶颈排序、证据链、修复方案和复测计划 |

## 完成标准

“覆盖”不等于“掌握”。只有满足以下条件才算完成：

1. 12 个知识单元的即时测验均达到 80%。
2. 所有与本机条件兼容的实验均能运行，并能解释关键输出。
3. 不能运行的 GPU / 多卡实验，要完成纸面账本、预期结果和验证命令。
4. 毕业项目至少达到 80/100，且结论必须由日志、profile、公式或复现实验支撑。

