# 03 Transformer：从张量形状到现代 Decoder-only LM

对应官方 Lecture 3-4 与 Assignment 1 模型部分。

## 3.1 完整数据流

一个现代 dense decoder-only Transformer：

```text
ids [B,T]
  -> token embedding [B,T,D]
  -> repeat L times:
       x = x + Attention(RMSNorm(x), causal=True, RoPE=True)
       x = x + SwiGLU(RMSNorm(x))
  -> RMSNorm
  -> LM head [B,T,V]
```

常见现代配置是 pre-norm、RMSNorm、RoPE、SwiGLU、无 bias；可能 weight tying，可能 GQA/MQA、局部 attention 或 MoE。原始 2017 Transformer 是 encoder-decoder、正弦绝对位置、post-LayerNorm、ReLU；不要把“Transformer”当成一个固定实现。

## 3.2 Linear、Embedding 与初始化

Linear：

$$y=xW^\top,\quad x:[B,T,D_{in}],\ W:[D_{out},D_{in}].$$

Embedding 是从矩阵 $E:[V,D]$ 按 ID 取行。它不是 one-hot 矩阵乘的实际实现，但数学等价。LM head 把 hidden 投到词表；若与 embedding 共享权重：`logits = x @ E.T`，节省 $VD$ 参数并给输入/输出表示一个共同空间，未必在所有规模和 tokenizer 上最优。

初始化要控制前向激活和反向梯度的尺度。Xavier 思想根据 fan-in/fan-out 设方差；深 residual 网络还常缩小残差分支输出投影，使 $L$ 层累加不过度放大。关键不是背某个常数，而是监控每层 activation RMS、gradient RMS、update-to-weight ratio 随深度和规模是否稳定。

## 3.3 Norm：LayerNorm、RMSNorm 与 pre-norm

LayerNorm 对最后一维：

$$\mathrm{LN}(x)=g\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+b.$$

RMSNorm 不减均值：

$$\mathrm{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\frac1D\sum_i x_i^2+\epsilon}}.$$

RMSNorm 少算均值，结构简单，现代 LLM 常用。计算平方和时通常升到 fp32，再 cast 回输入 dtype。

post-norm：`LN(x + F(x))`；pre-norm：`x + F(LN(x))`。pre-norm 给梯度提供更直接的 identity path，深模型更易优化；post-norm 的表示尺度被每层约束但训练可能更敏感。公开作业消融常见现象是删 norm 或改 post-norm 后 loss 更不稳定，但结论受模型深度、LR 和初始化影响，不能把一次小模型实验当普遍定律。

## 3.4 Scaled dot-product attention

对单头：

$$S=\frac{QK^\top}{\sqrt{D_h}}+M,\qquad P=\mathrm{softmax}(S),\qquad O=PV.$$

形状：`Q,K,V:[B,H,T,Dh]`，`S,P:[B,H,T,T]`，`O:[B,H,T,Dh]`。

若 $q_i,k_i$ 独立、均值 0、方差 1，则点积方差约 $D_h$；除以 $\sqrt{D_h}$ 把尺度拉回 O(1)，避免 softmax 过饱和、梯度变小。

causal mask 要让位置 $t$ 只能看到 $s\le t$。被屏蔽处加负无穷；在低精度中可用 dtype 最小有限值，但要确保整行不会全 mask，否则 softmax 可能 NaN。padding mask 和 causal mask 是不同约束，组合时需测试。

### attention 真正在做什么

`QK^T` 是内容相关的寻址；softmax 把每个 query 对历史 key 的相似度变成权重；`PV` 从历史 value 聚合信息。多头让模型在不同投影子空间并行建立不同关系，但头不保证自动对应人类可命名语法规则。

## 3.5 Multi-head attention

从 `x:[B,T,D]` 线性投影出 Q/K/V，再拆为 $H$ 头。合头后做输出投影。标准 MHA 参数主项约 $4D^2$（QKV 三个 + 输出一个），每 token 线性投影前向约 $8D^2$ FLOPs；attention score/value 约 $4T D$ FLOPs/token/layer（causal 是否按三角实际计算取决于 kernel）。

常见错误：

- `D % H != 0`；
- transpose 后把 head/time 维混了；
- softmax 做在 query 维而不是 key 维；
- mask 上三角/下三角反了；
- RoPE 用在 V 或在拆头前旋错维度；
- 合头前没有正确 transpose/contiguous；
- scale 用 `sqrt(D)` 而非 `sqrt(Dh)`。

最强测试是手造 $T=3$ 的 Q/K/V 和 mask，逐元素验证；再与 reference 比前向和梯度。

## 3.6 RoPE：旋转怎样表达相对位置

RoPE 把每对 hidden 维视为二维平面，对位置 $m$ 旋转角 $m\theta_i$：

$$R_m=
\begin{bmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{bmatrix}.$$

对 Q/K 应用旋转后：

$$q_m^\top k_n=(R_m q)^\top(R_n k)=q^\top R_{n-m}k,$$

点积显式依赖相对位移 $n-m$。频率通常 $\theta_i=\Theta^{-2i/D_h}$。实现缓存 sin/cos，支持任意 position IDs；增量 decode 时位置不能每步从 0 重启。

RoPE 不是自动无限外推。超出训练长度时，相位模式和 attention 分布会变；频率缩放、NTK/YaRN 类方法是在特定假设下折中近距离分辨率与远距离范围，必须用长上下文任务验证。

## 3.7 FFN 与 SwiGLU

原始 FFN：

$$\mathrm{FFN}(x)=W_2\,\sigma(W_1x).$$

SwiGLU：

$$\mathrm{SwiGLU}(x)=W_2\left(\mathrm{SiLU}(W_gx)\odot W_vx\right),$$

$$\mathrm{SiLU}(z)=z\,\sigma(z).$$

门控分支让每个位置按内容调制 value 分支。因为有三矩阵而非两矩阵，为保持参数/FLOPs 接近，$F$ 常取约 $\frac{8}{3}D$ 并向硬件友好倍数取整，而不是传统 $4D$。

Attention 负责跨位置通信，FFN 对每个位置独立变换并提供大量参数容量。不能把 FFN 简化成“增加非线性”就结束：在许多尺度下它占参数和矩阵 FLOPs 主体，也是 MoE 替换的主要位置。

## 3.8 Residual stream 的视角

把 $x$ 看作贯穿全网的 residual stream，每个 attention/FFN 读取规范化后的流并写入一个增量。这样容易理解：

- identity path 为什么帮助深层梯度；
- residual 分支初始化/缩放为何关键；
- attention 和 FFN 可并行计算的某些架构变体；
- activation patching 为什么可以在 residual stream 上干预。

但“可加和”不代表每层功能线性独立，后续层会非线性读取所有累计结果。

## 3.9 MQA、GQA 与 MLA：主要为 KV cache 服务

自回归 decode 要缓存每层历史 K/V。标准 MHA 每 token 每层缓存约：

$$2H D_h \times \text{bytes}=2D\times\text{bytes}.$$

- **MQA**：所有 query 头共享一组 K/V，cache 约缩 $H$ 倍，速度好但容量可能下降。
- **GQA**：$H_q$ 个 query 头分成 $H_{kv}$ 组共享 K/V，在质量与 cache/带宽间折中。
- **MLA**：先把 K/V 压到低维 latent，并在计算时吸收/展开投影，进一步压 cache；实现、RoPE 解耦和 kernel 更复杂。

GQA 论文的核心方法是把已有 MHA checkpoint 通过复制/平均等方式 uptrain 为中间 KV 头数，证明它可接近 MHA 质量而接近 MQA 速度。选择时要看真实 serving 的 batch、上下文和内存，不只看预训练 loss。

## 3.10 长上下文 attention 的路线

### 局部/滑窗 + 少量全局

每个 token 只看最近 $w$ 个位置，复杂度从 $O(T^2D)$ 降为 $O(TwD)$。Longformer/Sparse Transformer 通过固定稀疏模式增加跨区块路径。优点是保留 softmax attention 语义、容易高效；缺点是远距离信息需多层传播或显式全局 token。

### Linear attention

若相似度可写为 $\phi(q)^T\phi(k)$：

$$\mathrm{Attn}(Q,K,V)=\frac{\phi(Q)(\phi(K)^TV)}{\phi(Q)(\phi(K)^T\mathbf 1)}.$$

先算 $K^TV$ 可把序列复杂度线性化，并有递归状态 $S_t=S_{t-1}+\phi(k_t)v_t^T$。难点是 softmax kernel 的精确表示、状态容量、数值稳定和实际硬件效率；线性时间不保证墙钟更快。

### State Space / Mamba 类

把历史压入固定状态并递归更新。Mamba 的关键是让状态更新参数依赖输入，从而选择性保留/遗忘内容，再设计硬件友好的并行 scan。Mamba-2 用 state-space duality 与矩阵结构统一 SSM/attention 视角；Mamba-3 又强调更有表达力的离散化、复数状态和 MIMO，在 constant-memory decode 与状态跟踪间改善 Pareto 前沿。局限是固定状态对精确任意检索可能困难，训练和 kernel 生态也不如 attention 成熟。

## 3.11 Mixture of Experts

MoE 把 dense FFN 换成 $E$ 个专家，router 为每 token 选 top-$k$：

$$g(x)=\mathrm{softmax}(W_rx),\qquad y=\sum_{e\in\mathrm{TopK}(g)}g_e(x)\,\mathrm{FFN}_e(x).$$

总参数可增大 $E$ 倍附近，但每 token 只激活 $k$ 个专家，因此 active FLOPs 较小。代价：

- all-to-all 通信和跨设备路由；
- 专家负载不均，最慢专家决定 step 时间；
- capacity overflow/drop token；
- router collapse、训练抖动和辅助 loss；
- 推理仍需存/加载大量专家权重，batch 小时不一定划算。

Shazeer et al. 2017 证明稀疏门控可把模型容量扩到巨大规模；Switch Transformer 把 top-1 routing 简化并研究稳定训练。后续方法用 load-balancing loss、capacity factor、noise/jitter、无辅助 loss 的 bias 调整等改善负载。核心判断：MoE 优化的是“每单位 active compute 的参数容量”，不是免费计算。

## 3.12 资源核算

忽略 bias/norm，单层 dense SwiGLU Transformer 参数近似：

$$N_{layer}\approx 4D^2+3DF.$$

embedding/head 另加 $VD$（若不共享则两份）。attention 激活的 naive score 是 $BHT^2$，长上下文极大；FlashAttention 不保存完整 score，但反向仍需重算/保存少量统计。

模型变宽会让矩阵参数/FLOPs 约二次增长；变深线性增长；上下文增加会让 dense attention 平方项增长，同时降低同 token budget 下可采样序列数。任何架构比较必须固定至少一种预算：参数、active FLOPs、训练 tokens、墙钟或推理成本，不能混着赢。

## 3.13 核心论文思路

### Vaswani et al., *Attention Is All You Need* (2017)

问题：RNN 顺序依赖限制训练并行和长距离路径。方法：完全用 multi-head self-attention + position-wise FFN，配位置编码、残差和规范化；整序列并行训练，任意两位置路径长度为 1。证据来自机器翻译；局限是 attention 随长度平方增长，且原论文是 encoder-decoder，不等同现代 GPT。

### Su et al., *RoFormer / RoPE* (2021)

问题：把位置信息加入 token 表示后，attention 点积不自然表达相对位移。方法：对 Q/K 施加随位置变化的分块旋转，使点积只通过旋转差依赖相对位置。优势是无额外 attention bias 表、适配增量推理；局限是长度外推仍需处理频率分布。

### Shazeer, *GLU Variants Improve Transformer* (2020)

系统比较 ReLU/GELU/Swish 与 GLU 门控变体，发现 SwiGLU/GEGLU 在大致匹配参数/计算预算下改进。它更多是经验发现而非完备理论；今天的结论应理解为强默认值，而非任何数据/规模都必胜。

### Ainslie et al., *GQA* (2023)

介于 MHA 和 MQA：减少 K/V 头以缩小 cache 和带宽，同时保留多组 K/V 的表达能力；提出从 MHA checkpoint uptrain。适合 decode 成本重要的 LLM。

## 3.14 模型实现验收

1. 每个组件与 reference 比前向/梯度。
2. causal 测试：修改未来 token 不得影响过去 logits。
3. permutation/position 测试：无位置编码时序列置换应表现出相应等变性；加 RoPE 后不再如此。
4. 一个 batch 可被过拟合到极低 loss。
5. 每层 activation/gradient RMS 无系统爆炸或消失。
6. 参数量与手算误差小；profile 中主要算子与理论一致。
7. 消融只改一个因素并固定 token/FLOP/墙钟预算，报告多个 seed 或至少说明方差限制。
