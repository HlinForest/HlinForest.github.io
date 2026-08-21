# 00 学习路线：把“会调用模型”升级为“能从底层判断模型”

## 0.1 先定义“高手”

LLM 工作大致有五层：

1. **使用层**：prompt、API、RAG、工具调用。
2. **后训练层**：SFT、偏好数据、DPO、RL、评测。
3. **预训练层**：数据、tokenizer、模型、优化与 scaling law。
4. **系统层**：kernel、并行、显存、通信、推理服务。
5. **研究层**：设计可证伪实验，从噪声中判断改进是否真实、能否跨规模迁移。

CS336 的独特价值在于把 2-4 层连起来。高手不一定亲自维护每个 CUDA kernel，但必须知道抽象何时泄漏：OOM 是参数、优化器、激活还是临时张量造成？吞吐低是 Python launch、HBM、矩阵形状、通信还是负载不均？能力差是架构、token 预算、数据质量、训练不稳定还是评测污染？

## 0.2 你已有的知识怎样接上课程

只会 Python 和一点深度学习，最容易缺四块：

- **张量形状思维**：不是“attention 有 QKV”，而是能写出 `Q: [B,H,T,Dh]`，知道哪个维度做点积/softmax。
- **概率目标思维**：loss 是负对数似然；perplexity 是 `exp(loss)`，但跨 tokenizer 不可直接比。
- **资源思维**：每个操作同时消耗计算、显存容量、显存带宽、网络带宽和延迟。
- **实验思维**：一次漂亮曲线不是结论；要固定预算、记录配置、重复种子、报告不确定性和失败运行。

第 01 章补齐最低数学与 PyTorch。不要先用三个月“学完线代概率”；在具体机制处补知识更有效。

## 0.3 三条并行学习线

### 概念线

按正文顺序，目标是能脱离原文复述机制和取舍。每章后用空白纸画数据流，尤其是 Transformer、FlashAttention、DDP 和 GRPO。

### 实现线

按五个实验推进：

1. 小型 LM：tokenizer → Transformer → AdamW → 生成。
2. 系统：benchmark → profiler → fused kernel → DDP/FSDP。
3. Scaling：小实验 → IsoFLOP → 外推与误差分析。
4. 数据：HTML → 文本 → 过滤 → PII → 去重 → 数据消融。
5. 后训练：baseline → SFT/DPO 或 GRPO → 多指标评测。

所有实现都遵循同一节奏：先写接口和 shape test，再写最慢但清楚的正确版本，最后优化；每次优化都必须和 reference 对齐数值，并做 end-to-end 验证。

### 论文线

第一次只读摘要、图 1、方法总图和结论，回答“它改变了哪一个瓶颈”。第二次读公式/算法和实验设置。第三次才读附录与复现细节。每篇论文留下六行卡片：

```text
问题：现有方法在什么条件下失败？
洞见：作者把问题重新表述成什么？
方法：改了数据、架构、目标还是系统？
证据：与什么基线，在什么预算上比较？
局限：规模、数据、硬件或评测有哪些边界？
迁移：我会在哪个项目里使用/拒绝它？
```

## 0.4 24 周建议计划

| 周 | 主题 | 可验收产物 |
|---:|---|---|
| 1-2 | 张量、概率、PyTorch、数值稳定 | 手算并验证 linear/softmax/cross-entropy |
| 3-4 | Unicode、byte BPE | tokenizer round-trip 与特殊 token 测试 |
| 5-7 | Transformer | 小模型过拟合一个 batch，生成可读 TinyStories |
| 8 | 优化与训练循环 | 可恢复 checkpoint、可重复曲线、消融表 |
| 9-11 | GPU、profiling、Triton/FlashAttention | 正确性 + 显存 + 吞吐三份报告 |
| 12-13 | DDP/FSDP 与并行策略 | 单卡/多卡等价性与强/弱扩展曲线 |
| 14-15 | Scaling law | IsoFLOP 拟合、置信区间、外推审计 |
| 16 | 推理 | KV cache 内存表、prefill/decode roofline 判断 |
| 17 | 评测 | 自建 evaluation card 与污染检查 |
| 18-20 | 数据工程 | 带 lineage 的小型语料和过滤消融 |
| 21-23 | SFT/DPO/GRPO | baseline、训练曲线、reward hacking 检查 |
| 24 | 综合复盘 | 从预算出发写一份 1B 模型训练方案 |

若只有 CPU/消费级 GPU，把目标缩成“机制正确 + 小规模趋势成立”。不要用极小实验的绝对性能推断前沿模型，但可以学习 mechanics、mindset 和测量方法。

## 0.5 每次实验必须记录什么

最小运行清单：

```yaml
code: git commit / dirty diff
data: source, snapshot, hash, filter version
tokenizer: vocab, merges hash, special tokens
model: all dimensions and parameter count
optimization: optimizer, lr, schedule, batch tokens, clipping
runtime: hardware, software, dtype, seed, world size
budget: steps, tokens, wall time, estimated FLOPs
metrics: train/val loss, throughput, peak memory, task metrics
artifacts: checkpoint, logs, samples, failure reason
```

“大概用了同样设置”在 LLM 实验中等于不可复现。global batch 应写成 token 数：

$$B_{tokens}=B_{micro}\times T\times\text{grad\_accum}\times\text{world\_size}.$$

## 0.6 调试优先级

训练失败时按以下顺序，能避免把 bug 当成“需要调参”：

1. **数据与标签**：next-token 是否错位？padding 是否被 mask？文档边界是否正确？
2. **shape 与 mask**：causal mask 方向、broadcast 维度、RoPE 位置是否正确？
3. **数值**：softmax 是否减最大值？norm/累加是否升到 fp32？是否出现 NaN/Inf？
4. **梯度**：关键参数有没有梯度？范数多大？更新前后权重是否真的变化？
5. **过拟合小样本**：不能把一个 batch 的 loss 压到很低，就不应开大训练。
6. **与 reference 对齐**：固定随机张量，逐层比较输出和梯度。
7. **最后才调超参数与规模**。

## 0.7 如何知道自己真的掌握了

三个层次：

- **解释**：不用术语循环定义，能给出反例和极端情况。
- **推导**：能从形状推公式，从公式推复杂度，从复杂度推瓶颈。
- **改变**：换 tokenizer、序列长度、GPU 数或 reward 后，能预判哪些指标会变，再用实验检验。

读完一章后，尝试回答：“如果删掉这个组件会怎样？”“它的收益在哪个规模/硬件/数据分布下可能消失？”这两个问题比背定义更接近研究能力。
