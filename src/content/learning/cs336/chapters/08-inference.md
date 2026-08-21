# 08 推理与服务：prefill、decode、KV cache 与动态批处理

对应官方 Lecture 10，并连接 Lecture 3-6 的架构/系统知识。

## 8.1 训练与生成不是同一个 workload

给长度 $T_p$ 的 prompt，再生成 $T_g$ 个 token：

- **Prefill**：prompt 全部已知，可并行算所有位置；矩阵大，类似 training forward，常更 compute-bound。
- **Decode**：每步只新增一个 query，必须等上一个采样后继续；读取权重和历史 KV，batch 小时常 memory-bandwidth/latency-bound。

服务指标至少分开：time to first token (TTFT)、time per output token (TPOT)/inter-token latency、end-to-end latency、tokens/s、并发、尾延迟、成本。只给平均 tokens/s 会掩盖交互体验。

## 8.2 KV cache 为什么必要

不缓存时，生成第 $t$ 个 token 要重算前 $t-1$ 个位置每层 K/V，整个生成近立方/大量重复。缓存每层历史 K/V 后，每步只算新 token 的 Q/K/V，并让 Q 对 cache 做 attention。

标准 MHA 每请求 cache bytes：

$$M_{KV}=2\times L\times T\times H_{kv}\times D_h\times \text{bytes/dtype}.$$

例：$L=32,T=8192,H_{kv}=8,D_h=128,$ bf16，则约 `2*32*8192*8*128*2 ≈ 1 GiB/request`。并发几十个请求时 cache 往往比权重更吃显存。

GQA/MQA 降 $H_{kv}$；MLA 压低 latent 维；量化 KV 降 bytes；滑窗 attention 只保留最近窗口；跨层共享 K/V（CLA）减少层维。这些都可能损质量，需按长上下文/检索/多轮任务验证。

## 8.3 Decode 的 roofline

一次小 batch decode 的线性层大致每 token 读一次模型权重。若模型 $N$ 参数、权重 2 bytes，理论最低仅读权重就约 $2N$ bytes/token；7B 是 14GB。真实系统靠 batching 让同一权重读取服务多个 tokens，从而增算术强度。

这解释：

- batch 1 降权重 bit 数通常直接提速；
- 增 batch 能提高吞吐但增加排队和 token latency；
- prefill/decode 混在同一 batch 会互相干扰；
- MoE 每 token 只激活部分参数，但路由/专家权重 locality 决定实际带宽。

## 8.4 Continuous batching

静态 batch 等最慢请求结束，短请求浪费槽位。continuous batching 在每个 decode iteration：

1. 移除已完成序列；
2. 加入新请求或 prefill chunk；
3. 为所有活动序列各推进一个/若干 token；
4. 更新 cache page 和调度优先级。

需要 ragged attention、动态 cache 管理和公平调度。吞吐最大策略可能饿死长请求；生产要设优先级、最大等待、配额和取消传播。

## 8.5 Paged KV cache

若每请求预留最大长度的连续显存，内部碎片巨大，增长时还需搬迁。PagedAttention 把 KV cache 切成固定 blocks/pages，逻辑 token 位置通过 page table 映射到非连续物理块，类似虚拟内存：

- 按需分配，减少预留浪费；
- 不要求请求 cache 物理连续；
- beam/prefix 可 copy-on-write 或共享 blocks；
- kernel 需高效追踪间接地址。

page 太小元数据/随机访问开销大，太大内部碎片高。

## 8.6 Prefix caching

多个请求共享 system prompt、工具 schema 或文档前缀时，可按 token block hash 复用 prefill KV。必须把模型版本、adapter、RoPE/position、dtype、cache 配置纳入 key；不同租户还涉及隐私隔离。cache hit 提高 TTFT，不改变 decode 本身。

## 8.7 Quantization

### 权重量化

把 fp/bf16 权重 $w$ 映射到低 bit 整数 $q$：

$$q=\mathrm{clip}(\mathrm{round}(w/s)+z),\qquad \hat w=s(q-z).$$

粒度可 per-tensor/channel/group；group 小精度好、scale 开销和 kernel 复杂。weight-only INT8/INT4 对 memory-bound decode 很有效，prefill 若反量化开销/低 bit Tensor Core 不理想，提升不同。

AWQ 的核心是观察少数 activation 通道对输出误差特别敏感，用激活统计选择缩放/保护重要权重，再做低 bit weight-only，避免昂贵的完整重构。GPTQ 类方法用近似二阶信息逐块补偿量化误差。

### 激活/KV 量化

权重+激活可进一步提 compute，但 activation outlier、动态 scale、校准数据更难。KV 的 K/V 分布和 head/位置不同，量化误差会累积影响 attention；要专门测长上下文和生成质量。

量化评估不能只看 PPL 平均差：还要 reasoning、代码、长上下文、rare token、校准域偏移与速度/能耗。

## 8.8 Speculative decoding

小 draft 模型一次提出 $k$ 个 token，大 target 模型用一次并行 forward 验证。按接受/拒绝规则可保证最终样本分布与直接从 target 采样完全相同，因此是 lossless acceleration。

收益近似由 draft 成本、target 验证效率和 acceptance length 决定。draft 太弱接受率低，太强本身贵；batch 大/target 已吃满时额外并行收益可能下降。也可用同模型的早退头、多 token prediction head 或 n-gram 作为 draft。

“target 给每个位置打分”不等于简单接受 argmax；采样情形要用概率比修正，拒绝后从 residual distribution 采样，才能保持精确分布。

## 8.9 架构层面的推理优化

- **GQA/MQA**：减少 KV cache 与 decode bandwidth。
- **MLA**：低维 latent cache，复杂投影可吸收进权重。
- **Local/sliding attention**：cache 只随窗口增长；丢远程精确访问。
- **CLA**：相邻层共享 K/V；减 cache，容量可能降。
- **MoE**：active FLOPs 小于总参数，但专家权重调度/并发复杂。
- **Linear/SSM**：固定状态、线性/常数内存 decode；质量和状态跟踪权衡。
- **Multi-token prediction**：一次预测多个未来 token，可作训练辅助或 speculative head。

架构论文若只报 pretraining FLOPs，不足以判断 serving；应画 quality-latency-memory frontier。

## 8.10 并行推理

单请求模型放不下用 tensor parallel；每层 collective 会增加 latency，batch 大时可用计算覆盖。pipeline parallel 会给单请求增加 stage 延迟，但可多 microbatch/请求提高吞吐。data parallel 复制完整模型，路由请求，最利于独立扩容但权重显存重复。

Prefill 可 context parallel，decode cache 可按 heads/tensor parallel shard。实际部署常把 prefill 与 decode disaggregate 到不同 GPU 池：前者优化 compute，后者优化带宽/显存；代价是传 KV、调度和负载预测。

## 8.11 Serving capacity 估算

先算三张预算表：

1. 权重 + runtime buffer + graph capture 占用。
2. 每 token KV bytes，推并发/平均上下文能容多少。
3. prefill FLOPs 与 decode 权重/KV bytes，推理论瓶颈。

再用流量分布做离散事件/压测：prompt/output 长度、到达率、SLO、取消、cache hit。Little's Law：

$$L=\lambda W$$

系统平均并发 $L$ 等于到达率 $\lambda$ 乘平均停留时间 $W$；接近饱和后 queueing latency 非线性爆炸，所以容量不能按平均吞吐 100% 配置。

## 8.12 论文思路

### Ainslie et al., *GQA* (2023)

减少 KV heads 以降低 cache/带宽，并通过少量 uptraining 把 MHA 模型转为 GQA；质量接近 MHA、速度接近 MQA。核心是 serving 约束反向塑造预训练架构。

### DeepSeek-V2 / MLA (2024)

把 K/V 表示压缩为 latent，缓存 latent 而非完整多头 K/V；通过矩阵吸收避免每步显式恢复全部权重，同时处理 RoPE 需要解耦的分量。显著缩 cache，但 kernel 和并行实现复杂。

### Longformer / Mistral sliding-window

用固定窗口让每 token attention 成本与 cache 有界；堆叠层扩大有效感受野。适合局部依赖强的任务；远距离精确检索与超长多轮状态是主要风险。

## 8.13 推理验收

报告冷/热 TTFT、TPOT、p50/p95/p99、吞吐、峰值显存、每请求 KV、不同 prompt/output 分布、batch/concurrency sweep、取消与 overload 行为。量化/投机/稀疏优化必须同时给质量等价性或退化曲线。
