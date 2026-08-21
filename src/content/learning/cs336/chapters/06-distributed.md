# 06 分布式训练：切数据、切状态、切模型与隐藏通信

对应官方 Lecture 7-8 与 Assignment 2 后半。

## 6.1 先分清两个目标

- **容量扩展**：单卡放不下模型/激活，必须分片。
- **吞吐扩展**：单卡放得下，但要更多 GPU 缩短墙钟。

不同并行法切分不同维度：

| 方法 | 切什么 | 主要通信 | 解决 |
|---|---|---|---|
| Data Parallel / DDP | batch | gradient all-reduce | 吞吐 |
| ZeRO-1 | optimizer state | reduce-scatter/all-gather 更新 | 状态显存 |
| ZeRO-2 | optimizer + gradient | reduce-scatter/all-gather | 更多显存 |
| ZeRO-3 / FSDP | 参数 + gradient + optimizer | 每层 all-gather、reduce-scatter | 模型容量 |
| Tensor Parallel | 单层矩阵维度 | all-reduce/all-gather/reduce-scatter | 单层放不下/算力 |
| Pipeline Parallel | 层 | stage 间 activation/gradient send | 深度容量 |
| Sequence/Context Parallel | 序列 | activation/attention 通信 | 长上下文 |
| Expert Parallel | experts | token all-to-all | MoE 容量 |

真实大训练把多种方法组成多维 mesh；每个维度优先放在最合适的拓扑上，例如高频 tensor parallel 放节点内 NVLink，data parallel 跨节点。

## 6.2 Collective communication

常见 collective：

- broadcast：一份发给所有 rank；
- reduce：多份按 sum/max 等聚到一处；
- all-reduce：reduce 后每 rank 都有结果；
- all-gather：每 rank 的 shard 拼成完整张量并分发；
- reduce-scatter：先 reduce，再把结果分片给各 rank；
- all-to-all：每 rank 给每个 rank 发送不同片段，MoE 常用。

ring all-reduce 可视为 reduce-scatter + all-gather。每 rank 总传输量约 $2\frac{p-1}{p}M$ bytes，随着 rank 数趋近 $2M$，但环有 $O(p)$ 阶段延迟。tree 算法阶段少，适合小消息；库会按拓扑/大小选择。

通信时间粗模：

$$T\approx \alpha\cdot n_{messages}+\beta\cdot n_{bytes},$$

$\alpha$ 是延迟，$\beta$ 是每 byte 时间。把许多小梯度逐个 all-reduce 会被延迟支配；bucket 能摊薄 $\alpha$，但 bucket 太大又推迟重叠起点。

## 6.3 DDP 的正确语义

每 rank 持完整模型和 optimizer，读不同 microbatch。反向得到本地梯度 $g_r$，all-reduce 后通常除 world size：

$$g=\frac1P\sum_{r=1}^{P}g_r.$$

若每 rank loss 是本地平均，这等价于 global batch 平均，前提是本地 batch 相同；不等长/mask token 数不同要按有效 token 加权，不能盲平均 rank loss。

初始化时所有 rank 参数必须一致；optimizer step 只在梯度同步完成后发生。随机 dropout 可不同，但数据 sampler 必须在 epoch/step 上正确 reseed。

### naive、flat、overlap、bucketed

1. **naive**：反向全结束后逐参数 all-reduce；大量小通信且无重叠。
2. **flat**：把梯度拼平后一次 all-reduce；减少 latency，但通信完全在反向后。
3. **per-parameter overlap**：autograd hook 在某参数梯度 ready 时异步 all-reduce；能重叠，但消息太碎。
4. **bucketed overlap**：按反向 ready 顺序分 bucket，bucket 满/ready 立即通信；在带宽利用与早启动间折中。

参数注册顺序不一定等于梯度 ready 顺序，bucket 顺序错会推迟通信。异步 handle 必须在 optimizer 前 wait；否则产生极难察觉的错误更新。

## 6.4 Scaling efficiency

强扩展：global workload 固定，加 GPU 是否缩短时间：

$$E_{strong}(P)=\frac{T_1}{P T_P}.$$

弱扩展：每 GPU workload 固定，GPU 增加时每 step 时间是否不变。训练论文还应报告 tokens/s/GPU，因为 global batch 扩大可能改变优化，纯系统速度不能等同到达某 loss 的时间。

DDP 在模型反向计算足够大、网络快、bucket 合理时可很好扩展；小模型/短序列/慢网络时 communication exposed time 占比上升。

## 6.5 ZeRO 与 FSDP

设每参数 Adam 类状态近似包括参数、梯度、两阶矩等。纯 DDP 在每卡复制所有状态。

- **ZeRO-1**：每 rank 只保留约 $1/P$ optimizer states；梯度同步后对应 owner 更新 shard，再让参数一致。
- **ZeRO-2**：梯度也分片，通常 reduce-scatter 直接给 owner。
- **ZeRO-3/FSDP**：参数也静态分片；某层计算前 all-gather 完整参数，计算后可 reshard；梯度 reduce-scatter。

FSDP 把显存从 $O(N)$ 推近 $O(N/P)$，但每层增加通信和临时完整参数。wrap 粒度太细消息多，太粗峰值临时内存大；prefetch/reshard 策略影响 overlap。checkpoint 可保存 full 或 sharded state dict，恢复时 world size 变化需专门转换。

### ZeRO-1 通信并非“免费”

可用 reduce-scatter 梯度到 optimizer owner，再 all-gather 更新后的参数，通信量与 all-reduce 同量级，但 optimizer state 显存下降。若每 rank 先做完整 all-reduce 再只更新 shard，正确但浪费。作业价值在于理解 collective 组合，而不是调用现成 FSDP。

## 6.6 Tensor Parallel

对 $Y=XW$：

### Column parallel

按输出列切 $W=[W_1,\dots,W_P]$：每 rank 算 $Y_r=XW_r$，输出天然分片；若下一层能消费分片则不立刻 gather。

### Row parallel

按输入行切 $W=[W_1;\dots;W_P]$ 且 $X=[X_1,\dots,X_P]$：每 rank 算部分 $X_rW_r$，需要 sum/all-reduce 得完整输出。

MLP 常把第一层 column-parallel、第二层 row-parallel 配对，只在块末一次通信。attention 可按 heads 切。TP 每层都通信，频率高，通常放高带宽低延迟互连内。

Sequence Parallel 把在 TP 中原本复制的 norm/dropout 激活沿 sequence 分片，通过 reduce-scatter/all-gather 与 TP 配合，减少激活复制。

## 6.7 Pipeline Parallel

把连续层分为 $P$ stages。最朴素整 batch 先全 forward 再 backward，会让大部分 stage 空闲。把 batch 切成 $m$ microbatches，用 1F1B 等 schedule 交错，气泡比例粗略随 $(P-1)/m$ 降低。

代价：

- stage 负载不均，最慢 stage 决定吞吐；
- activation 在 stage 间传输；
- microbatch 增多影响 kernel 效率和 batch 语义；
- schedule、checkpoint、失败恢复复杂；
- embedding/head 或层 FLOPs 不均需虚拟 pipeline/interleaving。

## 6.8 Context Parallel 与长序列

sequence length 太大时，激活和 KV 不仅放不下，attention 还需跨设备。常见 ring attention：每 rank 持一段 Q/K/V，让 K/V block 沿 ring 传递，逐块用 online softmax 累积本地 Q 的输出。数学仍 exact，可把显存和计算分摊；但通信与 causal 负载平衡复杂。

仅把序列沿 rank 切开而不通信 K/V 会改变 receptive field，不是等价并行。

## 6.9 Expert Parallel

MoE router 决定 token 去哪个专家：先按目标 rank 打包，all-to-all 发送，专家计算，再 all-to-all 返回。性能取决于 token 数、capacity、top-k、专家分布和网络。负载均衡不仅是统计质量问题，也是同步 step 的系统瓶颈；99th-percentile rank 决定全局速度。

## 6.10 3D/4D 并行怎么选

实用顺序：

1. 若单卡能放，先 DDP，最简单且高效。
2. optimizer/参数放不下，用 FSDP/ZeRO。
3. 单层或每层计算太大，节点内 TP。
4. 层数深且还有容量压力，加 PP。
5. 长上下文加 CP/SP；MoE 加 EP。

选择 mesh 要核算：每 rank 参数/状态/激活/临时 buffer；每 step 各 collective bytes 和次数；可与哪段 compute overlap；节点内/间带宽；failure domain。不要只拿参数量除 GPU 数。

## 6.11 常见死锁与静默错误

- 不同 rank 以不同顺序调用 collective → 死锁。
- 某 rank 因数据异常提前 return/exception，其他 rank 永远等待。
- 异步 collective 未 wait 就读 buffer。
- gradient accumulation 中每个 microbatch 都同步，白白通信；应用 `no_sync` 类策略，仅最后一次同步。
- rank loss 直接平均但有效 token 不同。
- sampler 每 rank 读相同数据，表面 batch 变大实际重复。
- 参数有条件分支导致某 rank unused，不同计算图。
- checkpoint 只在 rank 0 保存未 gather 的 shard。
- 日志/随机 seed 完全相同造成增强/dropout 意外相关，或完全不可复现。

调试时先 `world_size=2`、单机、极小确定输入；给每个 collective 加递增序号/shape 日志；设置 timeout；比较单卡 global batch reference 的一步参数更新。

## 6.12 论文思路

### Shoeybi et al., *Megatron-LM* (2019)

把 Transformer 内的大矩阵按行/列切成 tensor parallel 配对，避免频繁 gather，在节点内高带宽互连上训练数十亿参数模型。关键贡献是匹配 MLP/attention 结构的切法，而非泛泛“模型分片”。

### Rajbhandari et al., *ZeRO* (2019/2020)

观察 data parallel 的模型状态在每卡完全复制，按 optimizer→gradient→parameter 逐级分片，在不改变模型数学的情况下大幅降冗余。代价是通信调度和临时 materialization；它奠定 FSDP/DeepSpeed 的状态分片思想。

### Huang et al., *GPipe* (2018/2019)

用 microbatch pipeline 把层分 stage，并配重计算降低激活显存。展示大模型跨加速器的通用方式；同步 pipeline 的 bubble 和 stage balance 是主要局限。

### Narayanan et al., *Efficient Large-Scale Language Model Training* (2021)

系统组合 tensor、pipeline、data parallel，建立通信/计算模型并按拓扑选择 3D 并行配置。核心思路是没有单一并行法覆盖所有规模，必须共同优化 mapping、microbatch 和硬件层级。

## 6.13 分布式验收

1. 单卡和多卡用同一 global batch 做一步，参数更新对齐。
2. 2/4/8 GPU 强扩展与弱扩展曲线。
3. profiler 显示通信与 backward overlap，并量 exposed communication。
4. 每卡 peak memory 与理论分片比例对照。
5. 不等长/masked token 的 loss/gradient 加权测试。
6. 中途 checkpoint 后在同/不同 world size 恢复（若设计支持）。
7. 注入一个 rank 异常，程序能 fail fast 而非永久挂起。
