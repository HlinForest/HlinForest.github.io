# 05 资源核算、GPU、Triton 与 FlashAttention

对应官方 Lecture 2、5、6 与 Assignment 2 前半。

## 5.1 GPU 快不是因为“很多 CPU 核”

现代 GPU 由多个 SM 组成；SM 调度 warp（NVIDIA 通常 32 threads）执行相同指令。内存层级大致：register → shared memory/L1 → L2 → HBM。越近越快、容量越小。Tensor Core 对特定 dtype/矩阵 tile 提供极高吞吐。

性能常受五类限制：

1. **compute-bound**：Tensor Core/ALU 已满。
2. **memory-bandwidth-bound**：在等 HBM 数据，算术强度低。
3. **latency/launch-bound**：算子太小、Python/kernel launch 太多。
4. **occupancy/resource-bound**：register/shared memory 太多，活跃 warp 不足。
5. **shape-bound**：维度不对齐硬件 tile、batch 太小，峰值算力用不上。

优化前要识别类别。把 memory-bound 算子的数学 FLOPs 再减 10% 可能毫无意义；少一次 HBM 往返更重要。

## 5.2 正确 benchmark

GPU 调用异步，CPU 计时会只量到 launch。最小规则：

- 固定硬件、功耗/频率环境、软件版本、dtype 与 shape；
- 预热，排除 CUDA context、allocator、JIT/`torch.compile`/Triton 编译；
- 使用 CUDA events 或测量边界同步；
- 重复多次，报告 median/分位数而非只报一次；
- forward、backward、optimizer、end-to-end 分开又合起来测；
- 同时量 peak memory、有效 TFLOP/s、tokens/s；
- 验证输出/梯度，错误 kernel 可以最快。

公开 Assignment 2 write-up 展示了“无预热”计时均值和方差大幅异常，而少量预热后稳定。这是可迁移的方法论：预热不是美化数字，而是定义 steady-state workload；若研究首请求延迟，则应另设 cold-start benchmark。

## 5.3 Profiler 的层级

1. Python end-to-end：发现数据加载、同步、checkpoint 等宏观问题。
2. `torch.profiler`：算子时间、shape、memory、CPU-GPU overlap。
3. Nsight Systems：线程、CUDA launch、kernel、通信的时间线。
4. Nsight Compute：单 kernel 的 occupancy、memory transactions、Tensor Core、stall reason。

先找 top contributors，再下钻。不要花一天把占 0.3% 的 GELU 优化 2 倍。

## 5.4 Kernel fusion 与 `torch.compile`

朴素序列 `read x -> op A -> write y -> read y -> op B -> write z`。融合后把中间 y 留在 register/shared memory，减少 launch 和 HBM bytes。elementwise、norm、activation、bias/dropout/residual 常适合融合。

`torch.compile` 捕获图、做算子融合和 codegen；它可能因动态 shape、graph break、首次编译、内存布局而表现不同。benchmark 应把编译时间单列，并检查生成图/guard。自写 Triton 适合需要明确 tile/online reduction 或编译器未覆盖的核心 kernel。

## 5.5 Triton 编程模型

Triton 用 program instance 处理一个 tile，而非手写每个 CUDA thread。典型结构：

```text
pid = program_id
offsets = pid * BLOCK + arange(0, BLOCK)
mask = offsets < n
x = load(ptr + offsets, mask)
... vector operations / reductions ...
store(out + offsets, y, mask)
```

关键选择：

- tile shape `BLOCK_M/N/K`；
- `num_warps`、stages；
- contiguous/coalesced load；
- mask 边界与 padding；
- accumulation dtype；
- register/shared memory 压力；
- autotune key 是否覆盖真实 shape。

先写 PyTorch reference，再写 forward kernel，再做随机 shape/dtype/causal 梯度测试，最后 profile。仅通过课程固定 shape 会产生脆弱 kernel。

## 5.6 为什么 naive attention 爆显存

标准实现显式 materialize：

```text
S = Q K^T       [B,H,T,T]
P = softmax(S)  [B,H,T,T]
O = P V         [B,H,T,Dh]
```

$T^2$ 矩阵要反复写/读 HBM。attention 数学 FLOPs 很多，但对许多形状，softmax 和中间 I/O 使 Tensor Core 等数据，尤其长序列激活容量也爆炸。

## 5.7 Online softmax

FlashAttention 的核心数学工具是分块、可合并的稳定 softmax。对一行 logits 的已处理块维护最大值 $m$、指数和 $\ell$、加权 value 累积 $o$。

新块最大值 $m_b$，新全局最大：

$$m'=\max(m,m_b).$$

旧贡献重缩放：

$$\ell'=e^{m-m'}\ell+\sum_{j\in b}e^{s_j-m'},$$

$$o'=e^{m-m'}o+\sum_{j\in b}e^{s_j-m'}v_j.$$

最终输出 $o/\ell$。因为最大值变化时旧指数按精确比例缩放，所以结果与整行稳定 softmax 数学等价（只存在浮点求和顺序差异）。

## 5.8 FlashAttention 的 IO-aware 分块

对 Q tile：

1. 从 HBM 载入一块 Q 到片上存储。
2. 顺序遍历 K/V tiles。
3. 计算 score tile，加 causal mask。
4. 用 online softmax 更新行最大、归一化因子、输出累积。
5. 只把最终 O 和少量统计写回 HBM。

不保存完整 $T^2$ P；反向根据 Q/K/V/O 与 log-sum-exp 重算局部 score/probability。这是以额外重算 FLOPs 换大幅 I/O/激活显存下降。复杂度仍是 exact dense attention 的 $O(T^2D)$ FLOPs，显存中间量从 $O(T^2)$ 降到近 $O(TD)$。

### FlashAttention-2 的改进

FA2 主要改善工作划分，而非新数学 attention：

- 减少非矩阵乘 FLOPs（GPU 上指数/归约相对昂贵）；
- 在 sequence 维增加并行，尤其 batch/head 小时；
- 更好地在 warp 间分配 Q/K/V 与输出，减少 shared-memory 通信和同步；
- causal 情况跳过完全遮蔽 tiles，并处理三角负载不均。

论文在 A100 上报告接近 GEMM 的效率与显著端到端提升，但你的结果依赖 head dim、causal、dtype、GPU、序列和 backward；“FlashAttention 更快”需在真实 workload 验证。

## 5.9 FlashAttention backward 的结构

设 $O=PV$，上游 $dO$：

$$dV=P^T dO,qquad dP=dO V^T.$$

softmax 行的 Jacobian 可化为：

$$dS=P\odot\left(dP-\sum_j P_jdP_j\right).$$

再有：

$$dQ=dS K/\sqrt{D_h},\qquad dK=dS^TQ/\sqrt{D_h}.$$

分块 kernel 需安全累积跨 tile 的 dQ/dK/dV，避免 race；可选择一个维度独占、atomic add 或多阶段归约。保存 $\delta_i=\sum_d O_{id}dO_{id}$ 可替代显式 $\sum_j P_{ij}dP_{ij}$。梯度检查必须覆盖 causal、非 2 的幂长度和不同 head dim。

## 5.10 Activation checkpointing 与 min-cut

反向通常需前向中间激活。checkpoint 只保存边界，反向重算内部，降低 activation memory，增加约一部分前向计算。选择 checkpoint 粒度：

- 每层：实现简单，重算多；
- attention/FFN 分开：更细；
- 编译器 min-cut：把图视作保存 bytes 与重算 FLOPs 的权衡，自动选边界。

它可能让更大 microbatch 提高吞吐，从而抵消重算；也可能在本已 compute-bound 时变慢。报告 end-to-end tokens/s，而非只说“省显存”。

## 5.11 Mixed precision 与缩放

bf16 因指数范围大，通常不需 loss scaling；fp16 小梯度可能下溢，做动态 loss scaling：前向 loss 乘 $S$，反向梯度放大；检查 finite 后除 $S$、clip、step。溢出则跳 step 并降低 $S$。

FP8/FP4 需要为 activation/weight/gradient 选择缩放粒度（tensor/channel/block）、格式与 amax 历史。量化误差不是只由 bit 数决定，outlier 和缩放更新同样关键。先用 bf16 正确基线，再引入低精度。

## 5.12 MFU 不等于整体效率

Model FLOPs Utilization：

$$\mathrm{MFU}=\frac{\text{模型理论 FLOPs/token}\times\text{tokens/s}}
{\text{设备峰值 FLOPs/s}\times\text{设备数}}.$$

它便于比较训练吞吐，但“模型 FLOPs”通常不计重算、通信、某些 elementwise 和 padding；峰值取决于 dtype/稀疏模式。高 MFU 可能伴随低硬件 FLOPs 利用之外的浪费，低 MFU 也可能因小模型/通信不可避免。应同时报 tokens/s、有效训练 loss/FLOP、功耗/成本和利用率。

## 5.13 论文思路

### Dao et al., *FlashAttention* (2022)

问题：attention 的墙钟/显存瓶颈来自 HBM I/O，而不只是 FLOPs。方法：用 tiling + online softmax 在 SRAM 中计算 exact attention，避免 materialize score/probability；给出 I/O complexity 分析。结果是长上下文明显加速和线性激活显存；局限是 dense attention 的数学平方复杂度仍在。

### Dao, *FlashAttention-2* (2023)

问题：FA1 仍未充分利用 GPU，warp 间归约/同步和并行划分有损失。方法：重排算法、减少非 matmul 操作、sequence 并行、更好的 warp 分工。论文报告约 2 倍于 FA1、A100 上达峰值的 50%-73%；结论绑定当时硬件/实现。

### PyTorch/编译器 activation checkpointing 工作

把保存激活看成图切割：在显存预算下选择保存哪些节点、重算哪些子图。核心价值是从“每层 checkpoint”升级为基于 bytes/FLOPs 的系统优化；动态 shape 和真实 kernel 代价仍让自动决策不完美。

## 5.14 系统作业验收

对每个优化给三张表：

1. **正确性**：前向、输入/参数梯度误差，shape/dtype/causal 覆盖。
2. **资源**：峰值 allocated/reserved memory，中间量理论 bytes。
3. **性能**：预热后 median/p10/p90，forward/backward/end-to-end，真实 shape sweep。

并记录 profiler 截图/trace 中瓶颈怎样从旧组件转移到新组件。没有 end-to-end 提升的微 kernel 胜利，应如实说明 Amdahl's law 限制。
