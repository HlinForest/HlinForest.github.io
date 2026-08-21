# 01 概率、张量与 PyTorch 生存包

对应官方 Lecture 1-2。本章只补后文必需的基础。

## 1.1 语言模型究竟学什么

给定 token 序列 $x_{1:T}$，自回归语言模型用概率链式法则写成：

$$p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).$$

模型每个位置输出词表上 $V$ 个未归一化分数 `logits`。softmax 把它们变成条件概率：

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.$$

训练最小化正确 token 的负对数概率：

$$\mathcal L=-\frac{1}{M}\sum_{m=1}^{M}\log p_\theta(y_m\mid x_m).$$

这等价于最大似然。它不是直接教“事实”或“推理规则”，而是让参数分布对训练序列赋更高概率；知识、格式、模仿、推理启发式都通过同一个 next-token 目标混在参数里。

### Cross-entropy、NLL、perplexity

one-hot 标签下，交叉熵就是负对数似然。若 loss 用自然对数且按 token 平均：

$$\mathrm{PPL}=e^{\mathcal L}.$$

直觉上 PPL 是模型在每一步面对的“有效分支数”。但它依赖 tokenizer：同一文本被切成不同 token 后，token 数和难度都变了，因此不同 tokenizer 的 token-level PPL 不能裸比。更严谨可报告 bits-per-byte/character 或在完全相同 tokenizer、语料和处理上比较。

### 稳定的 log-softmax

直接算 `exp(1000)` 会溢出。softmax 对所有 logits 加同一常数不变，因此：

$$\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m},\quad m=\max_j z_j.$$

稳定交叉熵可写为 `logsumexp(logits) - target_logit`。关键测试：给 logits 整体加 10,000，loss 应基本不变。

## 1.2 张量形状是第一语言

本书统一符号：

| 符号 | 含义 |
|---|---|
| $B$ | batch 中序列数 |
| $T$ | 序列长度/上下文长度 |
| $V$ | 词表大小 |
| $D$ | 模型宽度 `d_model` |
| $H$ | attention 头数 |
| $D_h=D/H$ | 每头维度 |
| $F$ | FFN 中间宽度 `d_ff` |
| $L$ | Transformer 层数 |

最常见形状链：

```text
token_ids             [B, T]
embedding              [B, T, D]
q/k/v before split     [B, T, D]
q/k/v after split      [B, H, T, Dh]
attention scores       [B, H, T, T]
attention output       [B, H, T, Dh] -> [B, T, D]
FFN hidden             [B, T, F]
vocabulary logits      [B, T, V]
targets                [B, T]
```

矩阵乘法只约掉相邻的共享维。例如 `X:[B,T,D] @ W:[D,F] -> [B,T,F]`。attention 中 `Q @ K.transpose(-1,-2)` 是 `[B,H,T,Dh] @ [B,H,Dh,T] -> [B,H,T,T]`。

### stride、view 与 contiguous

张量不仅有 shape，还有 stride：沿每个维度移动一个元素，底层存储要跨多少格。`transpose` 常只改 stride，不搬数据；随后 `view` 可能失败或误解布局。`reshape` 必要时复制，`contiguous()` 明确生成连续布局。kernel 性能常由访问是否合并决定，所以“数学形状一样”不等于“运行代价一样”。

### broadcasting

PyTorch 从末维对齐；维度相同或其中一个为 1 才能广播。causal mask 常是 `[T,T]`，会广播到 `[B,H,T,T]`。危险在于错误形状也可能“成功广播”。每个广播变量都应在名字或注释中标出期望形状，并用非对称的小维度测试。

## 1.3 einsum 与 einops：把下标写出来

`einsum` 例子：

```python
# x: [batch, time, model], w: [model, hidden]
y = torch.einsum("btd,df->btf", x, w)

# q,k: [batch, head, time, head_dim]
scores = torch.einsum("bhtd,bhsd->bhts", q, k)
```

字母只是一致性约束，不自带语义。重复且未出现在输出的下标被求和。`einops.rearrange` 适合显式拆头/合头：

```python
q = rearrange(q, "b t (h d) -> b h t d", h=num_heads)
out = rearrange(out, "b h t d -> b t (h d)")
```

高手习惯先写带名字的下标，再翻译成库调用，而不是靠反复试 `transpose`。

## 1.4 自动微分要理解到什么程度

若标量 loss $L$ 依赖中间量 $y=f(x)$，反向传播用链式法则把上游梯度乘局部 Jacobian：

$$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}.$$

无需显式构造巨大 Jacobian，框架做 vector-Jacobian product。需要牢记：

- 只有参与图且 `requires_grad=True` 的叶子参数积累 `.grad`。
- `.backward()` 默认累加梯度；每个 optimizer step 前要 `zero_grad` 或设为 `None`。
- `.detach()`、`torch.no_grad()`、把 tensor 变成 Python 数会切断图。
- in-place 操作可能破坏反向所需的保存值。
- gradient checkpointing 用反向时重算换激活显存，不减少参数/优化器状态。

用有限差分检查自定义算子：

$$\frac{\partial f}{\partial x_i}\approx\frac{f(x+\epsilon e_i)-f(x-\epsilon e_i)}{2\epsilon}.$$

梯度检查用 float64、小张量、非饱和输入；比较相对误差，不要求 bitwise 相等。

## 1.5 浮点数：范围、精度与累加

浮点数可抽象为符号、指数和尾数。指数位决定动态范围，尾数位决定相邻可表示数的间隔。

| dtype | 典型特点 | 训练用途 |
|---|---|---|
| fp32 | 范围和精度均较好，4 bytes | master weights、敏感归约、调试基线 |
| fp16 | 尾数较多但指数范围窄，2 bytes | 需 loss scaling，易上/下溢 |
| bf16 | 与 fp32 类似指数范围，尾数较少，2 bytes | 现代预训练常用，稳定性通常优于 fp16 |
| fp8/fp4 | 更省带宽/算力，格式与缩放复杂 | 需要分块/张量缩放和硬件支持 |

混合精度不是“把所有东西 cast 成低精度”。常见策略：矩阵乘用低精度输入、内部/输出按硬件策略处理；softmax、norm、loss、梯度归约或 optimizer state 在更高精度累加。数值是否安全取决于值域和归约长度。

## 1.6 参数量、显存与 FLOPs 的第一遍核算

Linear `D -> F` 无 bias 的参数是 $DF$；一次前向对一个 token 约 $2DF$ FLOPs（乘和加各算一次）。矩阵乘训练中，输入梯度和权重梯度各再约一个前向 matmul，因此线性层完整训练约是前向的 3 倍。

粗略的 dense decoder-only Transformer 每 token 前向主项：

$$\text{FLOPs}\approx 2N + 4LTD,$$

$2N$ 来自每个参数大体参与一次乘加，第二项是 attention 的 $QK^\top$ 与 $PV$，会随上下文 $T$ 增长。训练常用近似：

$$C\approx 6ND,$$

这里最后的 $D$ 在 scaling 文献中常表示训练 token 总数，容易与 `d_model` 冲突；后文改记为 $D_{tok}$。这个 `6N D_tok` 忽略 embedding、attention 平方项、稀疏激活、重算和硬件低效，只适合量级规划。

AdamW 训练显存不能只算权重。例如混合精度下可能有：低精度参数 2B、梯度 2B、fp32 master 参数 4B、一阶/二阶矩各 4B，总计约 16 bytes/parameter，另加激活、临时 buffer、allocator 碎片和通信 bucket。具体实现会不同，必须用 profiler 验证。

## 1.7 算术强度与 roofline 直觉

算术强度：

$$I=\frac{\text{FLOPs}}{\text{从主存搬运的 bytes}}.$$

硬件有峰值计算 $P_{peak}$ 和内存带宽 $BW$，可达性能近似：

$$P\le \min(P_{peak}, I\cdot BW).$$

若 $I$ 小，算子 memory-bound；优化应减少 HBM 往返、融合算子或增大复用。若 $I$ 大且矩阵形状适合 tensor core，可能 compute-bound；优化才聚焦更好的 tiling/并行。小矩阵还可能是 latency/launch-bound，roofline 两端都没吃满。

这解释了为什么：

- prefill 有大矩阵乘，较可能 compute-bound；
- 单 token decode 不断读取整套权重，batch 小时常 memory-bound；
- naive attention 把 $T^2$ score 写到 HBM，I/O 成为核心瓶颈；
- kernel fusion 的收益不是减少数学 FLOPs，而是少读写中间结果。

## 1.8 最小 PyTorch 测试清单

每个自写模块至少测：

1. 输出 shape/dtype/device。
2. 固定随机输入与 PyTorch reference 的前向误差。
3. 输入和参数梯度误差。
4. batch、长度为 1、非整齐维度等边界。
5. 极大/极小输入无 NaN/Inf。
6. CPU float64 的严格检查和目标 dtype 的合理容差。
7. 确定性：固定 seed 后结果可复现到预期程度。

性能测试要另外做；单元测试快不代表 GPU 快。GPU 是异步的，计时前后要同步或使用 CUDA event，并先预热以排除初始化、编译和缓存影响。

## 1.9 本章自测

1. 为什么把 logits 全加 100 不改变 softmax？
2. `Q:[2,8,512,64]` 与 `K` 点积得到什么形状，包含多少元素？
3. 为什么 fp16 比 bf16 更需要 loss scaling？
4. 一个算子 FLOPs 下降却变慢，可能有哪些原因？
5. AdamW 的 7B 参数模型为什么不可能只用 14GB 显存训练？

若不能从形状和 bytes 回答，不要急着进入 Transformer；先用小张量在 REPL 中验证。
