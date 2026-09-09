---
title: "Attention 的计算与优化：从 Softmax、FlashAttention 到 KDA"
description: "面向只学过 Attention 的读者，用矩阵图、手算例子和详细 NumPy 注释，推导 GQA、MLA、DSA、Linear Attention，以及从 3-pass safe softmax 到 FlashAttention 的优化过程，并衔接 DeltaNet 与 KDA。"
publishedAt: "2026-09-07"
updatedAt: "2026-09-09"
tags: ["Transformer", "Attention", "LLM Systems", "Optimization"]
series: "Machine Learning"
featured: false
draft: false
lang: "zh"
---


理解 attention 的优化，最好先问清楚：我们究竟在为哪一种成本付费？

有时是每个 query 都要与历史 key 比较，计算量随序列长度平方增长；有时是中间矩阵在显存里搬来搬去，算力在等待数据；到了推理阶段，还会遇到不断增长的 KV cache、内存碎片，以及多条请求反复计算同一段前缀。这几类成本同时存在，却不能用同一种办法解决。

沿着这个问题往下走，FlashAttention、PagedAttention 和 RadixAttention 的位置就清楚了：它们尽量保留原 attention 的语义，分别改变计算组织、缓存存储和计算复用。Linear attention 走的是另一条路：把历史压进固定大小的矩阵状态。接下来的 DeltaNet、Gated DeltaNet 和 KDA，则依次回答怎样改写记忆、怎样遗忘、怎样更细致地遗忘。

本文按这条思路展开，先把一个 attention 层算清楚，再讨论如何优化。正文展示关键运算；完整实现、反向传播和自动测试集中在文末附录，避免在推导中反复切换到大量辅助代码。

> **实现与验证范围**：参考代码依赖 NumPy 和 Matplotlib，可在 CPU 运行。配套参考程序 129 项与新增程序 18 项 CPU 检查均通过，共 147 项；另通过 PyTorch SDPA CPU 对照。FA1/2/3 的 Python 代码是算法与调度教学模型，真实 CUDA 异步指令与硬件性能仍需官方内核验证；本次验证环境已通过 PyTorch SDPA CPU 对照，但没有可用 CUDA，对应项目明确跳过。KDA 在这里指 **Kimi Delta Attention**。

这次阅读只要求你知道“query 用相似度给 value 加权”。暂时不懂 KV cache、低秩、外积、SRAM 或 `einsum` 都没有关系，第一次使用时会拆开说明。前向部分可以顺序阅读；第 10 节块内三角系统和第 12 节反向传播属于进阶内容，第一次可跳过。

**分步图解：** 本文 38 组矩阵图使用 3b1b/ManimGL 1.7.2 重新渲染。先看局部向量运算，再扩展到矩阵与时间链；每步说明放在对应图片下方。手机使用独立纵向布局，点击图片可打开高清图。Q 蓝、K 橙、V 紫、概率绿、状态及分母赭色、输出粉红，head 用编号区分。教学顺序参考用户提供的截图与 [Jia-Bin Huang 视频](https://youtu.be/Y-o545eYjXM)；视频主题为 GQA/MLA/DSA，Linear 和 Flash 的推导另依据文中原论文。各组使用局部教学数值，不代表同一套端到端模型参数；示意概率和 latent 的假设见图注。 [场景源码与复现说明](https://github.com/HlinForest/HlinForest.github.io/tree/main/scripts/attention-manim) · [生成清单](/images/notes/attention-from-softmax-to-kda/manim/manifest-desktop.json)

| 你想解决的问题 | 阅读位置 |
|---|---|
| 加权和到底算出了什么，多个 head 怎样合并 | §2、§2.1a–b |
| 为什么全遮蔽会 NaN，dropout 又是什么 | §2.1c–d |
| 矩阵乘法为什么是 $2mkn$ | §2.2a |
| KV、GQA、MLA、解耦 RoPE、DSA 怎样串起来 | §2.3–2.8 |
| 3-pass 怎么一步步变成 Flash | §3 |
| Linear 的固定状态里究竟存了什么 | §7 |
| 不熟悉 NumPy，怎样逐行运行 | 附录 A 开头的语法桥梁和已注释代码 |

## 1. 先分清三种问题：计算、搬运和记忆

| 路线 | 技术 | 是否改变 softmax attention 数学结果 | 优化对象 |
|---|---|---|---|
| 算子/模型 | Linear、DeltaNet、Gated DeltaNet、KDA | 通常改变，需按新模型训练 | 序列复杂度、有限状态记忆 |
| 内核 | FlashAttention 1/2/3 | 实数算术下相同；低精度有舍入/量化误差 | HBM IO、并行度、硬件流水线 |
| 推理系统 | PagedAttention、RadixAttention | 在 cache 正确时保持原 attention 语义 | 内存碎片、共享、重复 prefill |

这些不是一串互相淘汰的算法：分页存储、radix 前缀匹配和 Flash 类计算可以协同使用。

采用 **key-major 状态**：$S\in\mathbb R^{d_k\times d_v}$。有些论文写转置状态，更新左右乘法也会一起改变。

| 符号 | 含义 | 本文 ndarray 布局 |
|---|---|---|
| $B,H,N,M$ | batch、query heads、query 长度、KV 长度 | 不把 batch/head 轴参与矩阵收缩 |
| $d_k,d_v,r$ | key 维、value 维、特征映射维 | 可互不相等 |
| $Q,K,V$ | query/key/value | `[B,H,N,dk]`, `[B,H,M,dk]`, `[B,H,M,dv]` |
| $A,P,O$ | logits、概率、输出 | `[B,H,N,M]`, 同前，`[B,H,N,dv]` |
| $S,z$ | recurrent 状态与归一化和 | `[B,H,dk,dv]` 或 `[B,H,r,dv]`, `[B,H,r]` |
| $C,R,T$ | chunk 长度、query tile、key tile | 均与模型维度分开 |
| $\beta,\alpha$ | 写入率、遗忘门 | `[B,H,N]`, `[B,H,N,dk]` |

`@` 仅收缩最后两个维度；`einsum('...i,...j->...ij')` 是外积，不能写成内积。
`reshape(B,N,H,d).transpose(0,2,1,3)` 是分头；只 reshape 而不交换 N/H 会混乱 token 与 head。

为了让后面的优化有可比较的对象，我们先固定两件事：张量采用哪种布局，以及“计算完成”究竟指什么。输出形状相同不意味着模型相同；只有运算定义相同，才有资格讨论不同实现之间的数值等价。

## 2. 标准 attention：每一维如何消失，又如何留下

给定 $X\in\mathbb R^{B\times N\times D}$，先作线性投影 $XW_Q,XW_K,XW_V$，再分头。单头：

1. **点积**：$A_{ij}=\sum_{a=1}^{d_k}Q_{ia}K_{ja}/\sqrt{d_k}$。Q 的第 i 行与 K 的第 j 行收缩 key 维，留下 token×token。独立零均值单位方差分量下点积方差约为 $d_k$，缩放防止 logits 随维度增大而过大。
2. **mask**：$A_{ij}\leftarrow-\infty$ 若 key 不可见。因果可见条件为 `key_position <= query_position`；不能把 decode 的 query 局部下标 0 当成全局位置 0。
3. **逐行稳定 softmax**：$m_i=\max_j A_{ij}$，$E_{ij}=\exp(A_{ij}-m_i)$，$P_{ij}=E_{ij}/\sum_jE_{ij}$。减去相同常数不改变比值。不能把归一化放在 key 维之外。
4. **加权求和**：$O_{ib}=\sum_jP_{ij}V_{jb}$，最后拼接各 head 并乘 $W_O$。

先把这些操作各自算清楚。下面专门解释全遮蔽行、padding 和 dropout；它们不是同一个操作。

来源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。以下维度展开与实现为教学推导。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-01.png" alt="图 1：Q 与 K 的 key 维收缩，产生 token×token 权重，再与 V 收缩得到输出。图中展示一个 batch/head 的前四个 token。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 1：Q 与 K 的 key 维收缩，产生 token×token 权重，再与 V 收缩得到输出。图中展示一个 batch/head 的前四个 token。</figcaption>
</figure>



<!-- manim-group:projections:start -->
<section class="matrix-steps" data-matrix-group="projections" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 288px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/projections-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：一个 token 先做一次投影，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/projections-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/projections-01-desktop.png" alt="步骤 1：一个 token 先做一次投影" width="912" height="590" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：一个 token 先做一次投影</strong> 每个输出坐标都是输入向量与投影矩阵一列的点积；W_Q 对所有 token 共享。 <a href="/images/notes/attention-from-softmax-to-kda/manim/projections-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 266px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/projections-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：对每一行重复，得到 Q，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/projections-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/projections-02-desktop.png" alt="步骤 2：对每一行重复，得到 Q" width="912" height="663" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：对每一行重复，得到 Q</strong> 第一行与上一步完全相同。乘法只收缩输入特征轴，保留三个 token。 <a href="/images/notes/attention-from-softmax-to-kda/manim/projections-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 261px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/projections-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：使用另一组权重得到 K，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/projections-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/projections-03-desktop.png" alt="步骤 3：使用另一组权重得到 K" width="912" height="646" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：使用另一组权重得到 K</strong> Q、K、V 各自有训练得到的投影权重；它们不是从 Q 复制出来的。 <a href="/images/notes/attention-from-softmax-to-kda/manim/projections-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 279px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/projections-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：value 投影保留输出信息，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/projections-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/projections-04-desktop.png" alt="步骤 4：value 投影保留输出信息" width="1080" height="648" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：value 投影保留输出信息</strong> V 可以有三个通道，与本例两维 Q、K 不同。 <a href="/images/notes/attention-from-softmax-to-kda/manim/projections-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:projections:end -->



<!-- manim-group:scores:start -->
<section class="matrix-steps" data-matrix-group="scores" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 370px; --diagram-mobile-width: 370.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/scores-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：六个 token：先定位 query 与 key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/scores-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/scores-01-desktop.png" alt="步骤 1：六个 token：先定位 query 与 key" width="740" height="959" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：六个 token：先定位 query 与 key</strong> 总览图的格子表示因果可见性，不是概率大小。q₃ 只读取 k₁、k₂、k₃；后面的分步图再放大点积、归一化与 value 聚合。 <a href="/images/notes/attention-from-softmax-to-kda/manim/scores-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 488px; --diagram-mobile-width: 306px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/scores-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：把 key 行转成参与点积的列，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/scores-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/scores-02-desktop.png" alt="步骤 2：把 key 行转成参与点积的列" width="976" height="972" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：把 key 行转成参与点积的列</strong> 转置真实交换高和宽；原第 j 行变成第 j 列，token 编号保持不变。 <a href="/images/notes/attention-from-softmax-to-kda/manim/scores-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 372px; --diagram-mobile-width: 285px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/scores-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：选 query 3 与 key 2，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/scores-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/scores-03-desktop.png" alt="步骤 3：选 query 3 与 key 2" width="744" height="586" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：选 query 3 与 key 2</strong> 两个对应分量相乘后相加，只产生一个分数。 <a href="/images/notes/attention-from-softmax-to-kda/manim/scores-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 792px; --diagram-mobile-width: 348.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/scores-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：同一个 query 依次比较全部 key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/scores-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/scores-04-desktop.png" alt="步骤 4：同一个 query 依次比较全部 key" width="1584" height="589" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：同一个 query 依次比较全部 key</strong> 输出的第 j 格对应历史位置 j；下一步才屏蔽未来位置。 <a href="/images/notes/attention-from-softmax-to-kda/manim/scores-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 792px; --diagram-mobile-width: 306px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/scores-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：把 query 行排在一起，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/scores-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/scores-05-desktop.png" alt="步骤 5：把 query 行排在一起" width="1584" height="670" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：把 query 行排在一起</strong> 每条 query 独立生成一行。标准 logits 为 A=B/√d_k，这里 d_k=2。 <a href="/images/notes/attention-from-softmax-to-kda/manim/scores-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 656px; --diagram-mobile-width: 306px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/scores-06-desktop.png" target="_blank" rel="noopener" aria-label="步骤 6：每个点积分数使用相同缩放，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/scores-06-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/scores-06-desktop.png" alt="步骤 6：每个点积分数使用相同缩放" width="1312" height="716" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 6：每个点积分数使用相同缩放</strong> 缩放分母来自 key 通道数，而不是 token 数；所有图内除法使用上下分式。 <a href="/images/notes/attention-from-softmax-to-kda/manim/scores-06-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:scores:end -->

### 2.1a 加权求和：一行概率如何变成一个输出向量

把每个 value 看成一条有多个坐标的信息。假设一个 query 给两个 key 的权重为 $p=[1/4,3/4]$，对应 $v_1=[1,2],v_2=[5,6]$。先分别缩放两条信息，再按坐标相加：

$$
o=\tfrac14[1,2]+\tfrac34[5,6]=[4,5].
$$

同一组权重同时作用于所有 value 通道。它没有把两个 token 拼在一起，也没有取其中一个 token；它得到一个混合后的向量。把所有 query 的权重行叠起来，就是 $O=PV$。$P$ 的列是 key 位置，$V$ 的行也是 key 位置，两者必须一一对应。


<!-- manim-group:weighted-example:start -->
<section class="matrix-steps" data-matrix-group="weighted-example" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 320px; --diagram-mobile-width: 200px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：选中一条概率行，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-01-desktop.png" alt="步骤 1：选中一条概率行" width="640" height="636" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：选中一条概率行</strong> p 的第 j 项对应 V 的第 j 行；两个数是概率，value 行包含两个输出通道。 <a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 414px; --diagram-mobile-width: 278px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：一个概率缩放整条 value，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-02-desktop.png" alt="步骤 2：一个概率缩放整条 value" width="828" height="554" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：一个概率缩放整条 value</strong> 同一个标量乘到这一行的每个通道；不改变向量的长度。 <a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 414px; --diagram-mobile-width: 278px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：一个概率缩放整条 value，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-03-desktop.png" alt="步骤 3：一个概率缩放整条 value" width="828" height="554" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：一个概率缩放整条 value</strong> 同一个标量乘到这一行的每个通道；不改变向量的长度。 <a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 280.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：对应通道相加，得到输出，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-04-desktop.png" alt="步骤 4：对应通道相加，得到输出" width="922" height="512" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：对应通道相加，得到输出</strong> 第一个通道：0.25+3.75=4；第二个通道：0.5+4.5=5。每个输出通道都收到两条 value 的贡献。 <a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-example-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:weighted-example:end -->



<!-- manim-group:weighted-values:start -->
<section class="matrix-steps" data-matrix-group="weighted-values" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 298px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-values-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：一条概率行聚合全部 value，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/weighted-values-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/weighted-values-01-desktop.png" alt="步骤 1：一条概率行聚合全部 value" width="996" height="663" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：一条概率行聚合全部 value</strong> 第三个输出同时接收三个历史位置的贡献。 <a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-values-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 290px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-values-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：所有 query 使用同一个 V，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/weighted-values-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/weighted-values-02-desktop.png" alt="步骤 2：所有 query 使用同一个 V" width="996" height="646" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：所有 query 使用同一个 V</strong> P 的行对应 query，列对应 value 的行。P 的每一行独立得到 O 的同一行；不存在 v_i 单独决定 o_i 的关系。 <a href="/images/notes/attention-from-softmax-to-kda/manim/weighted-values-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:weighted-values:end -->

### 2.1b 多头的拼接：先横向排好，再学会怎样混合

每个 head 用自己的 Q/K/V 投影得到一个输出 $O^{(h)}$。同一个 token 在不同 head 下可以聚合不同的信息。若两头各输出 3 个数，则拼接后每个 token 有 6 个数；token 的数量不变。

$$
O_{merge}=[O^{(1)}\mid O^{(2)}\mid\cdots\mid O^{(H)}],\qquad
\Delta X=O_{merge}W_O.
$$

$W_O$ 的每一列告诉模型如何混合所有 head 的通道，得到一个新的模型通道。若按 head 将 $W_O$ **沿行切开**，同一运算也可理解为“每头输出乘自己对应的权重行块，再把结果相加”。切分方向由内维 $Hd_v$ 决定。注意这里的 $\Delta X$ 是 Attention 分支输出；是否再与 X 做残差相加，由外部 Transformer 层决定。


<!-- manim-group:heads:start -->
<section class="matrix-steps" data-matrix-group="heads" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 325px; --diagram-mobile-width: 325px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/heads-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：同一 token 的两份 head 输出，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/heads-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/heads-01-desktop.png" alt="步骤 1：同一 token 的两份 head 输出" width="650" height="566" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：同一 token 的两份 head 输出</strong> 两头对应同一个 token。head 用上标区分，颜色仍表示输出。 <a href="/images/notes/attention-from-softmax-to-kda/manim/heads-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 302.5px; --diagram-mobile-width: 303px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/heads-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：沿通道拼接，token 数不变，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/heads-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/heads-02-desktop.png" alt="步骤 2：沿通道拼接，token 数不变" width="605" height="499" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：沿通道拼接，token 数不变</strong> 左两列属于 head 1，右两列属于 head 2；1×2 与 1×2 拼成 1×4。 <a href="/images/notes/attention-from-softmax-to-kda/manim/heads-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 302px; --diagram-mobile-width: 302.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/heads-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：输出权重按输入通道分行块，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/heads-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/heads-03-desktop.png" alt="步骤 3：输出权重按输入通道分行块" width="604" height="839" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：输出权重按输入通道分行块</strong> 上两行接收 head 1，下两行接收 head 2。每块都输出相同的两个模型通道。 <a href="/images/notes/attention-from-softmax-to-kda/manim/heads-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 291.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/heads-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：head 1 乘自己的权重行块，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/heads-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/heads-04-desktop.png" alt="步骤 4：head 1 乘自己的权重行块" width="912" height="616" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：head 1 乘自己的权重行块</strong> 每个输出格子由左侧向量与右侧对应列点积得到；两个 contracted 轴长度均为 2。 <a href="/images/notes/attention-from-softmax-to-kda/manim/heads-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 291.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/heads-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：head 2 乘自己的权重行块，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/heads-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/heads-05-desktop.png" alt="步骤 5：head 2 乘自己的权重行块" width="912" height="616" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：head 2 乘自己的权重行块</strong> 每个输出格子由左侧向量与右侧对应列点积得到；两个 contracted 轴长度均为 2。 <a href="/images/notes/attention-from-softmax-to-kda/manim/heads-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 323px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/heads-06-desktop.png" target="_blank" rel="noopener" aria-label="步骤 6：两头贡献相加，完成输出投影，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/heads-06-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/heads-06-desktop.png" alt="步骤 6：两头贡献相加，完成输出投影" width="922" height="498" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 6：两头贡献相加，完成输出投影</strong> 拼接后的大矩阵乘法等于两个行块乘法之和。本例输出为 [6,5]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/heads-06-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:heads:end -->

### 2.1c 全遮蔽行：不是“大家概率都很低”，而是没有合法对象

`keep[i,j]=True` 表示 query i 可以使用 key j。因果 mask 不允许看未来；padding key mask 不允许读取为凑长度填进去的假 token。正常因果 Attention 包含自己，所以第一行至少能看到第一个 token，通常不会全遮蔽。全遮蔽来自额外约束，例如把某个 padding query 整行标为无效。

考虑三个分数 `[2,1,0]`。只保留前两个时，mask 后为 `[2,1,−∞]`，减最大值 2 后指数为 `[1,e⁻¹,0]`，再除以 $1+e^{-1}$ 即可。被遮蔽位置严格贡献零。

若三个位置全被遮蔽，分数为 `[−∞,−∞,−∞]`，最大值也是 $-\infty$。直接相减会出现 $-\infty-(-\infty)$，得到 NaN。即使把指数安全地设成全 0，归一化仍是 $0/0$。这不是一个有定义的概率分布，不能解释成均匀分布。

本文为固定无效 query 定义：**概率行全 0，Attention 输出全 0，该 query 的 Attention 分支梯度贡献也为 0**。共享的 K/V 仍可从其他有效 query 获得梯度；外层残差或投影偏置也不因此自动清零。代码先把非有限的行最大值替换为 0，让 `exp(-inf)=0`，再仅在分母大于 0 的位置做除法，其他位置保留预先写好的零。

为什么只 mask padding key 不够？它只禁止其他 query 读取这些列；padding query 本身仍能读取真实 key。若要求它的输出为 0，还须把该 query 行禁用。

### 2.1d Dropout：在训练中随机删掉部分贡献

dropout 用来给训练过程加入随机性。设丢弃概率 $p_d=0.5$，先按普通规则得到概率 P，再独立采样保留指示 $R_{ij}\in\{0,1\}$。训练使用 $\widetilde P=R\odot P/(1-p_d)$，随后计算 $\widetilde PV$；`⊙` 表示同位置相乘，`@` 才是矩阵乘法。

例如 `[0.25,0.75]` 本次采到保留指示 `[0,1]`，结果为 `[0,1.5]`，这一行和为 1.5，不再是概率分布，也不重新做 softmax。为什么乘 2？每一项有一半机会被保留，乘 2 后它的**期望贡献**与原值一致。单次输出仍会改变，两项也可能都被丢掉。

mask 决定哪些信息合法；dropout 决定本次训练随机保留哪些合法贡献。推理一般关闭 dropout，直接使用 P。本文比较各 Attention 实现时统一令 dropout=0；若实现 Flash 的训练 dropout，反向还须重现同一个随机掩码，不能重新随意采样。[PyTorch SDPA 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 也明确要求推理调用显式传 `dropout_p=0.0`。


<!-- manim-group:mask-dropout:start -->
<section class="matrix-steps" data-matrix-group="mask-dropout" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 355px; --diagram-mobile-width: 355.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：第 3 个 query 只能看前三个位置，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-01-desktop.png" alt="步骤 1：第 3 个 query 只能看前三个位置" width="710" height="569" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：第 3 个 query 只能看前三个位置</strong> 空格表示不参与计算的未来位置。mask 应在 softmax 前把对应 logits 设为负无穷。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 323px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：全遮蔽行没有可归一化的概率，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-02-desktop.png" alt="步骤 2：全遮蔽行没有可归一化的概率" width="808" height="632" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：全遮蔽行没有可归一化的概率</strong> 先判断是否存在可见 key；全遮蔽行直接返回零。不要先执行 −∞−(−∞)，那会产生 NaN。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 387px; --diagram-mobile-width: 387.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：dropout 在概率算完后丢弃部分连接，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-03-desktop.png" alt="步骤 3：dropout 在概率算完后丢弃部分连接" width="774" height="682" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：dropout 在概率算完后丢弃部分连接</strong> 示例保留掩码 R=[0,1]。保留项除以 0.5；这一行和为 1.5，不再强制归一化，期望保持原值。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 343.5px; --diagram-mobile-width: 343.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：全遮蔽与随机全丢弃是两种情况，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-04-desktop.png" alt="步骤 4：全遮蔽与随机全丢弃是两种情况" width="687" height="509" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：全遮蔽与随机全丢弃是两种情况</strong> 这里原概率有效，只是本次训练抽样恰好全部丢弃；推理时关闭 dropout。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mask-dropout-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:mask-dropout:end -->


### 2.1 一段可以作为基准的实现

这里将 causal mask 和外部 mask 合并，再执行数值稳定的 softmax。完整代码中的 `softmax_masked` 负责全遮蔽行的零输出约定。所有优化实现都先与这一基准比较。

```python
# 输入 a 的最后一轴为 key；固定全遮蔽行返回全零。合法分数应为有限值，mask 用 -inf。
def softmax_masked(a, keep=None):
    if keep is not None:
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        a = np.where(keep, a, -np.inf)
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    m = np.max(a, axis=-1, keepdims=True)
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
    safe_m = np.where(np.isfinite(m), m, 0.)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    e = np.exp(a - safe_m)
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    den = e.sum(-1, keepdims=True)
    # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    return np.divide(e, den, out=np.zeros_like(e), where=den > 0)

# q/k/v 前导轴是 batch/head；输出 o[...,N,dv] 与概率 p[...,N,M]。
def softmax_attention(q, k, v, causal=False, keep=None, q_start=0):
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
    assert q.shape[-1] == k.shape[-1] and k.shape[-2] == v.shape[-2]
    if causal:
        cm = position_mask(q.shape[-2], k.shape[-2], q_start)
        keep = cm if keep is None else (keep & cm)
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    p = softmax_masked((q @ k.swapaxes(-1,-2)) / math.sqrt(q.shape[-1]), keep)
    return p @ v, p
```

### 2.2 训练与生成：同一公式，两种负载

训练或 prefill 时，一次输入整段序列。单头需要两个主要矩阵乘法：$QK^T$ 和 $PV$，总计算量为 $\Theta(N^2(d_k+d_v))$。推理生成时，每次只新增一个 query，但它仍需读取全部可见历史，因此单步成本是 $\Theta(N(d_k+d_v))$。

缓存历史 K/V 可以避免反复计算旧 token 的投影，却没有取消新 query 对历史的读取。也正因为如此，prefill 常有较大的矩阵乘法并行空间，decode 则更容易受 KV 读取带宽和请求 batch 大小影响。

### 2.2a 矩阵乘法的运算量：从一个格子开始数

计算 $C=AB$，A 的形状为 $m\times k$，B 为 $k\times n$。输出格子 $C_{ij}$ 要做 k 次乘法，把 k 个结果加起来需 k−1 次加法。输出共有 mn 个格子，因此精确标量计数为 $mn(2k-1)$，通常记作约 $2mkn$ FLOPs。硬件的一条 FMA 可以同时做乘加，但按常见 FLOPs 口径仍计 2 次浮点运算。


<!-- manim-group:matmul-count:start -->
<section class="matrix-steps" data-matrix-group="matmul-count" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 414px; --diagram-mobile-width: 342px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：先数一个输出格子的运算，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-01-desktop.png" alt="步骤 1：先数一个输出格子的运算" width="828" height="660" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：先数一个输出格子的运算</strong> 内积长度 k=3：需要 3 次乘法、2 次加法，共 5 FLOPs。 <a href="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 239px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：输出有 m×n 个格子，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-02-desktop.png" alt="步骤 2：输出有 m×n 个格子" width="996" height="642" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：输出有 m×n 个格子</strong> 本例 m=2、n=2，共 4 个内积：4×5=20 FLOPs。 <a href="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 366px; --diagram-mobile-width: 366.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：推广到任意矩阵大小，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-03-desktop.png" alt="步骤 3：推广到任意矩阵大小" width="732" height="642" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：推广到任意矩阵大小</strong> 一次乘加通常按 2 FLOPs 计。QKᵀ 约 2NMd_k，PV 约 2NMd_v；batch 和 head 数再乘到外面。 <a href="/images/notes/attention-from-softmax-to-kda/manim/matmul-count-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:matmul-count:end -->


| 运算 | 左形状 × 右形状 → 输出 | 单 batch、单头主乘法 FLOPs |
|---|---|---|
| 匹配 Q/K | $(N,d_k)(d_k,M)\to(N,M)$ | $2NMd_k$ |
| 聚合 value | $(N,M)(M,d_v)\to(N,d_v)$ | $2NMd_v$ |
| 自注意力 $M=N$ | 两次乘法相加 | $2N^2(d_k+d_v)$ |
| decode 只有一个新 query | $N_q=1,M=t$ | $2t(d_k+d_v)$ |
| 三个 QKV 投影，设 $Hd_k=Hd_v=D$ | 每次 $(N,D)(D,D)$ | 三次合计 $6ND^2$（全部头） |
| 输出投影 | $(N,D)(D,D)$ | $2ND^2$（全部头） |

多 batch、多头的混合部分再乘 BH；投影表中已经包含全部 head，不能再乘 H。缩放、mask、exp 和行归约还有 $O(BHNM)$ 工作，表中只列主矩阵乘法；这些标量操作也可能影响实际速度。

例：$N=M=4096,d_k=d_v=64$，单头 QK 与 PV 各约 21.47 亿 FLOPs，合计约 42.95 亿。长度翻倍，两项都变成四倍。若有 32 个头，混合部分约 1374.39 亿 FLOPs。单头一张 FP32 score 矩阵则占 $4096^2\times4=64$ MiB：**FLOPs、存储字节、实际读写量是三个不同的数**。

因果有效位置为 $N(N+1)/2$。真正跳过上三角的内核可以接近减半；先做完整 `q @ k.T` 再 mask 的 NumPy 程序仍计算了全部格子。


### 2.3 从 KV cache 走向 MQA/GQA：历史里到底存了几份

沿着提问所附视频章节表给出的 **07:07 KV cache → 09:42 MQA → 11:03 GQA** 顺序，先看逐 token 生成。第 t 步只有一个新 query；第 1 到 t−1 步的 K/V 已经计算过，因果层中的旧表示也不会因新增未来 token 而改变，因此可以留在缓存中。新一步只计算新 token 的投影，把新 K/V 追加进去，再让新 query 读取历史。query 用完即可，不需要作为 KV 缓存保存。

MHA 有 $H_q$ 个 query head，每个 head 都有自己的一套历史 K/V。如果很多 head 读到的历史表示相似，能否让它们共享历史，而保留不同的查询方式？MQA 把 K/V 缩成一组，所有 Q heads 共享；GQA 则折中，把 Q heads 分组，每组共享一套 K/V。

设 $H_q=8,H_{kv}=2$，每组 4 个 query head。Q0–Q3 使用 K0/V0，Q4–Q7 使用 K1/V1。**共享相同的 K/V，不代表得到相同的概率**：query 不同，点积分数、softmax 权重和每头输出仍可以不同。

令 $g=H_q/H_{kv}$，零基 head 编号 h 所属组为 $u(h)=\lfloor h/g\rfloor$：

$$
P_h=\operatorname{softmax}\!\left(\frac{Q_hK_{u(h)}^T}{\sqrt{d_k}}+mask\right),\qquad O_h=P_hV_{u(h)}.
$$


<!-- manim-group:gqa-groups:start -->
<section class="matrix-steps" data-matrix-group="gqa-groups" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 488px; --diagram-mobile-width: 361px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：八个 query heads，共享两组 KV，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-01-desktop.png" alt="步骤 1：八个 query heads，共享两组 KV" width="976" height="657" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：八个 query heads，共享两组 KV</strong> 每列槽位仍代表一个 query head；组号是离散 ID。组内共享历史 K/V，不共享 Q 或概率。 <a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 274px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：head 0 读取组 0 的 key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-02-desktop.png" alt="步骤 2：head 0 读取组 0 的 key" width="912" height="600" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：head 0 读取组 0 的 key</strong> 得到自己的分数后，head 0 独立做缩放、mask 和 softmax。 <a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 290.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：head 1 再读取同一份 key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-03-desktop.png" alt="步骤 3：head 1 再读取同一份 key" width="912" height="599" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：head 1 再读取同一份 key</strong> K₀ 与上一帧是同一缓存对象。query 不同，所以概率可以不同。 <a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 298px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：head 0 的概率读取共享 V₀，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-04-desktop.png" alt="步骤 4：head 0 的概率读取共享 V₀" width="912" height="584" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：head 0 的概率读取共享 V₀</strong> 这里 a=exp(1/√2)，与前面点积按 d_k=2 缩放后的 softmax 一致。两个 head 读同一个 V₀，得到不同输出。 <a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 298px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：head 1 的概率读取共享 V₀，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-05-desktop.png" alt="步骤 5：head 1 的概率读取共享 V₀" width="912" height="583" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：head 1 的概率读取共享 V₀</strong> 这里 a=exp(1/√2)，与前面点积按 d_k=2 缩放后的 softmax 一致。两个 head 读同一个 V₀，得到不同输出。 <a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 307px; --diagram-mobile-width: 307px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-06-desktop.png" target="_blank" rel="noopener" aria-label="步骤 6：缓存按 KV heads 计数，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-06-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-06-desktop.png" alt="步骤 6：缓存按 KV heads 计数" width="614" height="626" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 6：缓存按 KV heads 计数</strong> 本例相对八头独立 KV 缓存减少为四分之一；QKᵀ、PV 的主要运算仍按八个 query heads 计算。 <a href="/images/notes/attention-from-softmax-to-kda/manim/gqa-groups-06-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:gqa-groups:end -->


| 方案 | Q heads | KV heads | 每层每 token 的 KV 元素数 |
|---|---:|---:|---:|
| MHA | $H_q$ | $H_q$ | $H_q(d_k+d_v)$ |
| GQA | $H_q$ | $1<H_{kv}<H_q$ | $H_{kv}(d_k+d_v)$ |
| MQA | $H_q$ | 1 | $d_k+d_v$ |

例如 $H_q=8,H_{kv}=2,d_k=d_v=64$，使用每元素 2 bytes 的格式，单层 4096 token 的 MHA KV 为 8 MiB，GQA 为 2 MiB，MQA 为 1 MiB。乘上 batch 与层数，才是模型的总 KV 量。缓存减少是形状直接推出的；实际带宽收益还取决于 kernel 是否复用共享数据。

混合部分仍要为每个 Q head 计算自己的 QK/PV，主乘法量按 $H_q$ 计。`np.repeat` 可以复制 K/V 来写一个便于验证的基准，但它会把缓存复制回多份，不能作为生产内存节约的实现。GQA 是架构选择；把任意已训练 MHA 的 KV 头直接平均并不保持原输出，原论文还使用继续预训练适配。来源：[MQA](https://arxiv.org/abs/1911.02150)、[GQA](https://arxiv.org/abs/2305.13245)。

### 2.4 MLA 第一步：把多头历史压进共同的 latent

按所附章节表定位 **13:32 MLA**。GQA 让若干头直接共享同一份 K/V；MLA 让所有头共享一份**压缩表示**，再用不同的升维矩阵构造各头所需的内容 key 和 value。latent 在这里只是“一条较短的中间向量”。

以下省略 batch，采用行向量，令 C 为经过压缩投影及 RMSNorm 后的历史表示。为了先看线性关系，暂将归一化略写在 C 的定义中：

$$
C=\operatorname{RMSNorm}(XW_{DKV}),\quad K_h^C=CU_h^K,\quad V_h=CU_h^V.
$$

X 为 $M\times D$，C 为 $M\times r$，$U_h^K$ 为 $r\times d_c$，$U_h^V$ 为 $r\times d_v$。$d_c$ 是每头内容 key 宽度，r 是 KV latent 宽度，二者不同。所谓压缩针对的是全部头展开后的总宽度，不要求 r 小于每个单头的宽度。

一个 token 只存一行 C，但不同头有不同 $U_h^K,U_h^V$，仍可从同一行中提取不同信息。这里 r 是每条历史的压缩维度；第 7 节 Linear Attention 用 r 表示核特征维度，含义不同。MLA 历史仍有 M 行，因此缓存随历史长度增长。


<!-- manim-group:mla-expand:start -->
<section class="matrix-steps" data-matrix-group="mla-expand" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 329px; --diagram-mobile-width: 330px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：先保留每个 token 的压缩表示，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-01-desktop.png" alt="步骤 1：先保留每个 token 的压缩表示" width="658" height="656" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：先保留每个 token 的压缩表示</strong> 下列数值从示意 latent C 开始，不声称是特定 RMSNorm 参数的输出。每行仍是独立历史 token，所有 heads 共享这些行。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 280px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：某个 head 展开内容 key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-02-desktop.png" alt="步骤 2：某个 head 展开内容 key" width="1080" height="670" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：某个 head 展开内容 key</strong> 3×2 latent 经 2×3 投影展开为 3×3 内容 key；每个 head 有自己的升维权重。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 243px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：从同一 C 展开 value，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-03-desktop.png" alt="步骤 3：从同一 C 展开 value" width="912" height="668" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：从同一 C 展开 value</strong> 这一步展示概念展开。推理时可用下一组结合律避免缓存展开后的每头 K/V。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-expand-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:mla-expand:end -->


### 2.5 MLA 第二步：推理时为什么可以不展开全部历史

按所附章节表定位 **15:37 推理时的 MLA**。如果每次都把历史 C 展开成所有头的 K/V，省下的缓存可能又变成大量临时读写。注意这次只新增一条 query $q_h^C$，可以把 key 升维搬到它这一侧：

$$
q_h^C(CU_h^K)^T
=q_h^C(U_h^K)^TC^T
=\bar q_hC^T,\qquad \bar q_h=q_h^C(U_h^K)^T.
$$

左边先展开 M 条 key，右边只变换当前 query，再读取 M 条压缩历史。括号重排没有跨过 softmax，因此保持同一个 MLA 的分数。概率 p 得到后，value 路径也可改为先聚合 latent：

$$
p_h(CU_h^V)=(p_hC)U_h^V=z_hU_h^V.
$$

这次避免的是 M 条 value 的逐条升维，只对汇总后的一个向量升维。每个 head 有自己的概率 $p_h$，所以仍有自己的 $z_h$。不能先把不同 head 的 z 合并成一份，再假定结果相同。


<!-- manim-group:mla-absorb:start -->
<section class="matrix-steps" data-matrix-group="mla-absorb" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 582px; --diagram-mobile-width: 262px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：先看展开后的内容点积，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-01-desktop.png" alt="步骤 1：先看展开后的内容点积" width="1164" height="682" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：先看展开后的内容点积</strong> 这一写法需要展开全部历史内容 key。这里只比较未缩放内容分数。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 325px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：把升维权重吸收到当前 query，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-02-desktop.png" alt="步骤 2：把升维权重吸收到当前 query" width="996" height="685" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：把升维权重吸收到当前 query</strong> 一次 query 变换得到 latent 宽度 2，避免对每条历史重复展开。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 321px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：直接读取压缩缓存，分数相同，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-03-desktop.png" alt="步骤 3：直接读取压缩缓存，分数相同" width="1080" height="584" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：直接读取压缩缓存，分数相同</strong> 输出与第一帧同为 [1,1,2]。吸收只改变括号，不跨过 softmax 或 RMSNorm。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 262.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：先用主概率聚合 latent，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-04-desktop.png" alt="步骤 4：先用主概率聚合 latent" width="996" height="652" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：先用主概率聚合 latent</strong> 为便于验算，此处另取一条示意主概率 [1/4,1/4,1/2]；它不是前一帧分数的 softmax。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-absorb-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:mla-absorb:end -->


再把总 $W_O$ 沿 head 对应的行分块为 $W_h^O$，可得到：

$$
\Delta x=\sum_h z_h(U_h^VW_h^O).
$$

这叫权重吸收：两个相邻线性变换可以组合。组合后的权重可能更大，工程中要权衡存储、形状和吞吐；代数等价不自动保证每种实现更快。RMSNorm 和 softmax 是非线性操作，不能跨过它们把所有矩阵任意合并。类似地，若内容 query 由当前 query latent 乘 $U_h^Q$ 得到，也可在其后组合 $U_h^Q(U_h^K)^T$；query latent 本身的归一化仍需执行。


<!-- manim-group:mla-output:start -->
<section class="matrix-steps" data-matrix-group="mla-output" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 326px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-output-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：组合相邻的 value 与输出投影，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-output-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-output-01-desktop.png" alt="步骤 1：组合相邻的 value 与输出投影" width="912" height="616" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：组合相邻的 value 与输出投影</strong> W_O 的第 h 个行块接收该 head 的 value 输出。这里没有越过非线性运算。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-output-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 342.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-output-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：latent 汇总直接投影到模型宽度，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-output-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-output-02-desktop.png" alt="步骤 2：latent 汇总直接投影到模型宽度" width="912" height="571" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：latent 汇总直接投影到模型宽度</strong> 每个 head 算出相同模型宽度的一份贡献；所有 head 的贡献最后相加。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-output-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 302.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-output-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：对应模型通道累加各头贡献，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-output-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-output-03-desktop.png" alt="步骤 3：对应模型通道累加各头贡献" width="922" height="549" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：对应模型通道累加各头贡献</strong> 本帧使用独立的两头求和小例子，强调这里相加的是投影后的模型通道，不是 latent 通道。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-output-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:mla-output:end -->


MLA 的因果混合仍遍历历史，每 token 每头 latent 比较与聚合主项约 $4Mr$ FLOPs，另有当前 query 变换、位置支路和输出投影。它首先改变 KV 存储和执行组织，并不把主 Attention 变成固定状态的 Linear Attention。依据：[DeepSeek-V2 §2.1](https://arxiv.org/abs/2405.04434)、[DeepSeek-V3 §2.1.1](https://arxiv.org/abs/2412.19437)。

### 2.6 RoPE 为什么卡住吸收，解耦后又怎样恢复

按所附章节表定位 **18:15 解耦 RoPE**。先把 RoPE 理解为：每两个特征坐标构成一个二维小平面，位置 t 决定在这个平面里旋转多少角度。不同坐标对使用不同频率；query 和 key 分别按自己的位置旋转，二者点积就能反映相对位置。[RoFormer 原论文](https://arxiv.org/abs/2104.09864)。


<!-- manim-group:rope-pairs:start -->
<section class="matrix-steps" data-matrix-group="rope-pairs" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 261.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/rope-pairs-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：先旋转一个二维坐标对，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/rope-pairs-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/rope-pairs-01-desktop.png" alt="步骤 1：先旋转一个二维坐标对" width="912" height="596" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：先旋转一个二维坐标对</strong> 本组采用行向量右乘约定。90 度旋转把 [1,0] 变成 [0,1]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/rope-pairs-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 323px; --diagram-mobile-width: 323px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/rope-pairs-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：对每个坐标对独立旋转，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/rope-pairs-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/rope-pairs-02-desktop.png" alt="步骤 2：对每个坐标对独立旋转" width="646" height="745" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：对每个坐标对独立旋转</strong> 示例第一对旋转 90 度，第二对旋转 0 度；对外的零值留白。实际角度由位置乘该对频率确定。 <a href="/images/notes/attention-from-softmax-to-kda/manim/rope-pairs-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:rope-pairs:end -->


采用行向量右乘旋转矩阵的约定。若内容 query 与内容 key 都直接旋转，单个历史位置 s 的分数会出现：

$$
(q_t^CR_t)(c_sU^KR_s)^T
=q_t^CR_tR_s^T(U^K)^Tc_s^T.
$$

中间的 $R_s^T$ 随历史位置 s 变化，且通常不能与 $U^K$ 交换。原先“给当前 query 做一次固定变换就能匹配所有历史”的路径不再成立。这不是括号不够灵活，而是矩阵顺序和位置依赖不允许它那样移动。



<!-- manim-group:rope-obstruction:start -->
<section class="matrix-steps" data-matrix-group="rope-obstruction" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 384px; --diagram-mobile-width: 385px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：普通 RoPE 在投影中间插入位置因子，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-01-desktop.png" alt="步骤 1：普通 RoPE 在投影中间插入位置因子" width="768" height="662" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：普通 RoPE 在投影中间插入位置因子</strong> 当前 query 的 R_t 固定，但每个历史 s 的 R_s 不同。不能把所有历史位置共用一次 query 变换。 <a href="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 294.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：历史位置 0 的 query 变换，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-02-desktop.png" alt="步骤 2：历史位置 0 的 query 变换" width="912" height="601" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：历史位置 0 的 query 变换</strong> 暂取 R_t=I；下一帧只换历史位置旋转。 <a href="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 305px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：历史位置 1 的变换已经不同，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-03-desktop.png" alt="步骤 3：历史位置 1 的变换已经不同" width="912" height="597" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：历史位置 1 的变换已经不同</strong> 一般矩阵不可交换。MLA 用独立内容与位置两路解决这个障碍。 <a href="/images/notes/attention-from-softmax-to-kda/manim/rope-obstruction-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:rope-obstruction:end -->

解耦方案把 query/key 分成内容和位置两条支路：内容支路按上一节吸收；位置支路单独投影并应用 RoPE。query 的位置向量 $q_{t,h}^R$ 每头不同；历史位置 key $k_s^R$ 在各头间共享。拼接两路再点积，相当于两路分数相加：

$$
A_{t,h,s}=\frac{\bar q_{t,h}c_s^T+q_{t,h}^R(k_s^R)^T}{\sqrt{d_c+d_R}}.
$$

mask、softmax 在两路相加并缩放之后进行，不能分别 softmax 后相加。吸收后内容 query 宽度变成 r，但**缩放仍取原来的 $\sqrt{d_c+d_R}$**；不能因实现形状变化就改成 $\sqrt{r+d_R}$，否则不再是原来同一个运算。


<!-- manim-group:mla-rope:start -->
<section class="matrix-steps" data-matrix-group="mla-rope" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 219px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：内容路读取 latent，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-01-desktop.png" alt="步骤 1：内容路读取 latent" width="1080" height="585" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：内容路读取 latent</strong> 此路可以吸收内容 key 的升维权重。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 295.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：位置路读取共享 RoPE key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-02-desktop.png" alt="步骤 2：位置路读取共享 RoPE key" width="1080" height="600" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：位置路读取共享 RoPE key</strong> 这里输入已经按各自位置旋转。位置 query 每头不同，位置 key 跨主 heads 共享。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 587px; --diagram-mobile-width: 240px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：两路同位置分数相加，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-03-desktop.png" alt="步骤 3：两路同位置分数相加" width="1174" height="568" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：两路同位置分数相加</strong> 对每个历史位置对齐相加后缩放、mask、softmax。分母仍使用原内容宽度 d_c 加位置宽度 d_R，不改为 latent 宽度 r。 <a href="/images/notes/attention-from-softmax-to-kda/manim/mla-rope-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:mla-rope:end -->


推理每层每个历史 token 只需缓存 C 的 r 个数和位置 key 的 $d_R$ 个数，合计 $r+d_R$；主 query latent 不属于历史 KV。比如取 $r=512,d_R=64$，每元素 2 bytes，一条历史为 1152 bytes。这里只演示 KV 元素量，不含权重、量化 scale、索引器或分配器。内容压缩与解耦定义见 [DeepSeek-V2 §2.1.2–2.1.3](https://arxiv.org/html/2405.04434v5)。

### 2.7 DSA：先挑要读的位置，再做主 Attention

按所附章节表定位 **22:18 DSA**。MLA 让每条历史更小，但每个新 query 仍要扫描很多条历史。DeepSeek Sparse Attention 加入一个轻量 **Lightning Indexer**：先为可见历史位置打分，选出少量位置，再让主 MLA 在这些位置上计算自己的 Attention。

索引器有自己的 query、key 和少量索引头，不等于主 Attention heads。它可以先于主 Attention 执行：索引 query 来自当前 token 的 query latent，索引 key 来自各历史输入的投影与归一化；不需要先得到主 Attention 输出或完整概率矩阵。

$$
I_{t,s}=\sum_{h=1}^{H_I}w_{t,h}^I\operatorname{ReLU}\!\left(q_{t,h}^I(k_s^I)^T\right).
$$

ReLU 把每个索引头的负点积变为 0；之后乘当前 query 对应的 head 权重并求和。官方实现的 $w^I$ 没有正值约束，因此总分 I 仍可能为负。**I 是选取位置用的分数，不是主 Attention 概率**；TopK 不需要预先把它归一化。


<!-- manim-group:dsa-indexer:start -->
<section class="matrix-steps" data-matrix-group="dsa-indexer" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 262px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：索引器先算自己的点积，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-01-desktop.png" alt="步骤 1：索引器先算自己的点积" width="1080" height="610" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：索引器先算自己的点积</strong> 轻量索引器先扫描候选 key；它不是从主 Attention 输出反推索引。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 237px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：每头先截去负点积，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-02-desktop.png" alt="步骤 2：每头先截去负点积" width="808" height="507" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：每头先截去负点积</strong> ReLU 把负点积变为零；接下来仍有可为负的 head 权重。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 587px; --diagram-mobile-width: 262px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：加权汇总得到索引分数，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-03-desktop.png" alt="步骤 3：加权汇总得到索引分数" width="1174" height="564" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：加权汇总得到索引分数</strong> 本例权重为 2 和 −1。I 可为负，是排序分数而非概率；主 Attention 将重新计算自己的 logits。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-indexer-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:dsa-indexer:end -->


用零基位置定义有序索引列表 $J_t=(j_{t,0},\ldots,j_{t,k_t-1})$，其中 $k_t=\min(k,t+1)$，每个位置不同且位于 0…t。TopK 得到位置集合后，本文为便于阅读按历史位置排序；同时重排所有被选 K/V 不改变 Attention 的加权和。前缀不足 k 个 token 时使用全部可见位置，不能混入未来或无效填充槽位。

例如 t=4 时索引分数为 `[8,2,9,1,7]`，k=3，选中的历史位置为 `[0,2,4]`。先按这些**整数地址**取出 C 与位置 key：$C_{sel}[a,:]=C[J_t[a],:]$；主 Attention 然后在所选三条历史上重新计算内容分数、位置分数、softmax 和 latent 加权和。索引器的 `[8,9,7]` 不会直接成为主概率。


<!-- manim-group:dsa-gather:start -->
<section class="matrix-steps" data-matrix-group="dsa-gather" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 530px; --diagram-mobile-width: 312px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：屏蔽未来，再取 Top-K 位置，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-01-desktop.png" alt="步骤 1：屏蔽未来，再取 Top-K 位置" width="1060" height="562" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：屏蔽未来，再取 Top-K 位置</strong> 位置从 0 开始。分数前三名是位置 2、0、4；图中再按源位置排序成 [0,2,4]。未来位置 5 不可选。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 357px; --diagram-mobile-width: 282.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：整数位置变成实际缓存行，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-02-desktop.png" alt="步骤 2：整数位置变成实际缓存行" width="714" height="908" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：整数位置变成实际缓存行</strong> 选中第 0、2、4 行，顺序对应三个 selected slots。Kᴿ 必须使用同一组位置 gather；所有主 heads 共享这组整数位置。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 331.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：主 query 对选中内容重新打分，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-03-desktop.png" alt="步骤 3：主 query 对选中内容重新打分" width="1080" height="600" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：主 query 对选中内容重新打分</strong> 再加同位置的 RoPE 分数，并由主 softmax 归一化；索引器分数不直接作为主权重。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 326px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：主概率读取选中的 value 信息，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-04-desktop.png" alt="步骤 4：主概率读取选中的 value 信息" width="996" height="662" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：主概率读取选中的 value 信息</strong> 此处用示意主概率验算 gather 后的聚合。latent 汇总随后进入 value/output 投影。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-gather-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:dsa-gather:end -->


同一 query 的全部主 MLA heads 共享 J，但每头仍产生自己的概率。对应当前 query 的主混合部分从读 M 条变成读 $k_t$ 条：内容和 value latent 约 $4H_qk_tr$ FLOPs，位置打分约 $2H_qk_td_R$。不过索引器本身仍需扫描候选历史：整段序列的索引工作仍是二次量级。不能把整个 DSA 直接说成严格 O(Nk)，也不能把它与固定大小状态的 Linear Attention 混同。

要实际节约主算术，必须在主计算前 gather 或使用支持索引读取的稀疏 kernel。先计算全长主 logits 再 mask 只适合作为正确性基准，不会省掉那些乘法。本文新增代码同时给出两条路径，比较的是同一选取集合下的数值一致性。

### 2.8 DSA 的旋转、量化与训练：为什么能选得又快又准

按所附章节表定位 **23:57 量化与旋转 → 27:44 DSA 训练**。索引器要扫描历史，所以自身必须便宜。官方路径将索引 Q/K 的部分通道先做 RoPE，再对完整向量做相同的归一化 Hadamard 变换，随后用 FP8 计算索引分数。Hadamard 是由正负号构成的正交变换，可把少数通道上的大数分散到更多通道；量化则用较少位数表示数字，降低存储和计算成本。历史索引 key 及其量化 scale 也需要缓存。

量化前的正交点积保持可复用图 4-A：$(qH)(kH)^T=qk^T$。量化后存在误差，排名也可能变化，不能保证所有输入的误差都下降。RoPE 与 Hadamard 有执行顺序，不能无理由交换。本文的均匀 INT8 风格演示只解释量化直觉，不冒充 FP8 编码或官方性能。

TopK 是离散选取：在排名不变的一小段扰动范围里，输出位置不变；排名交换时又发生跳变。单靠主语言模型损失，难以直接用通常的链式求导教会索引器挑位置。因此训练给它一个“老师”：主 Attention 已经算出的跨头平均概率。

| 阶段 | 主 Attention 怎么算 | 谁更新 | 索引器学什么 |
|---|---|---|---|
| dense warmup | 对全部可见历史做稠密 Attention | 冻结主模型，只更新索引器 | 在完整可见集合拟合主 Attention 的概率分布 |
| sparse training | 只在 TopK 选中集合做主 Attention | 主模型用语言模型损失；索引器用独立 KL | 在同一个选中集合上拟合当前稀疏主 Attention 的分布 |

令 $p_{t,s}=\frac1{H_q}\sum_hP_{t,h,s}$，即把主 head 的概率相加后再沿 key 归一化；让索引器分数在当前监督集合上做 softmax 得 $\pi_t$。索引损失为 $\mathrm{KL}(\operatorname{stopgrad}(p_t)\|\pi_t)$。方向是 teacher p 到 indexer π。`stopgrad` 表示把老师答案当作常数，不让该辅助损失反过来修改老师；索引器输入也 detach，使主模型由语言模型损失训练。稀疏阶段的老师来自当前**稀疏**主 Attention，不需要另算一遍全长 dense teacher。



<!-- manim-group:dsa-teacher:start -->
<section class="matrix-steps" data-matrix-group="dsa-teacher" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 332px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：跨主 heads 平均得到教师概率，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-01-desktop.png" alt="步骤 1：跨主 heads 平均得到教师概率" width="808" height="569" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：跨主 heads 平均得到教师概率</strong> 先固定同一 query、同一监督位置集合。教师概率 stop-gradient，不通过该 KL 更新主模型。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 323px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：索引分数在同一集合上归一化，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-02-desktop.png" alt="步骤 2：索引分数在同一集合上归一化" width="808" height="486" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：索引分数在同一集合上归一化</strong> 训练索引器时用 softmax 得到学生概率；推理选位置时直接用索引分数排名。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 261.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：用教师分布监督索引器，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-03-desktop.png" alt="步骤 3：用教师分布监督索引器" width="808" height="544" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：用教师分布监督索引器</strong> warmup 的监督集合是全部可见位置；稀疏阶段是所选位置。主模型用语言模型损失，索引器用 KL，索引器输入也从主模型梯度分离。 <a href="/images/notes/attention-from-softmax-to-kda/manim/dsa-teacher-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:dsa-teacher:end -->

这些是为什么“先选再算”可以训练的机制；不是把索引器 I 与主 QK 设成相同公式。主要来源：[DeepSeek-V3.2-Exp 官方发布](https://api-docs.deepseek.com/news/news250929/)、[官方 Indexer 源码](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)、[V3.2 报告 §2.1](https://arxiv.org/html/2512.02556v1)。后者用于交叉核验 Exp 架构与训练说明，并非视频逐字稿。

我们现在有了两个直接问题：能不能保留同一个公式，却不把 N×N 中间矩阵完整写入显存？能不能保留全部 KV，却更合理地存放和复用它们？先看第一个问题。

## 3. FlashAttention：不保存整张概率矩阵，怎样保证结果正确

先只看一行分数 $x=[x_1,\ldots,x_M]$。softmax 输出每项为 $\exp(x_j)/\sum_u\exp(x_u)$。若 $x_j=1000$，直接指数化会溢出；先减去最大值 m 后，每项指数都不超过 1，而分子分母同乘 $e^{-m}$，比值不变。

### 3.0a 起点：3-pass safe softmax

设一整行无法一直放在快速存储中，且不缓存中间指数数组。最直观的稳定算法是：

1. 第一遍读 x：找到 $m=\max_jx_j$。
2. 第二遍读 x：累计 $l=\sum_j\exp(x_j-m)$。
3. 第三遍读 x：输出 $P_j=\exp(x_j-m)/l$。

这里是三遍**算法读取**，共读 3M 个输入标量、写 M 个输出标量，不等于固定三个 GPU kernel。若保存中间指数数组，读取和写入的对象会改变；实际 HBM 流量也取决于缓存。以 `[0,1,2]` 为例，m=2，l=$e^{-2}+e^{-1}+1\approx1.5032$，输出约 `[0.0900,0.2447,0.6652]`。


<!-- manim-group:softmax-passes:start -->
<section class="matrix-steps" data-matrix-group="softmax-passes" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 320px; --diagram-mobile-width: 254px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：第 1 遍：只找最大值，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-01-desktop.png" alt="步骤 1：第 1 遍：只找最大值" width="640" height="568" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：第 1 遍：只找最大值</strong> 最大值需要看完所有元素才能确定；这一遍不输出概率。 <a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 327px; --diagram-mobile-width: 328px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：第 2 遍：重新读取并累加指数，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-02-desktop.png" alt="步骤 2：第 2 遍：重新读取并累加指数" width="654" height="636" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：第 2 遍：重新读取并累加指数</strong> 减最大值让指数不超过 1。若不保存 E，最终输出时还需重新读取 x 并计算指数。 <a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 266px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：第 3 遍：逐项输出概率，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-03-desktop.png" alt="步骤 3：第 3 遍：逐项输出概率" width="808" height="535" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：第 3 遍：逐项输出概率</strong> 三次读取分数行。也可存 E 把计算换成额外内存搬运；遍数必须说明中间量是否保存。 <a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 261px; --diagram-mobile-width: 261.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：在线更新最大值与分母，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-04-desktop.png" alt="步骤 4：在线更新最大值与分母" width="522" height="568" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：在线更新最大值与分母</strong> 读完 x₁=0 后 m=0,l=1；读到 x₂=log2 时旧分母乘 1/2，再加新项 1，得到 3/2。 <a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 384px; --diagram-mobile-width: 384.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：在线分母仍不等于一次输出全部概率，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-05-desktop.png" alt="步骤 5：在线分母仍不等于一次输出全部概率" width="768" height="542" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：在线分母仍不等于一次输出全部概率</strong> 在线第一遍得到最终 m,l；若要输出所有概率 P，通常仍需第二遍。Flash 的关键是直接累计加权 value，避免物化整行 P。 <a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-passes-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:softmax-passes:end -->


### 3.0b 优化一：把求最大值和分母合成一遍

读到新值 x 时，可能发现更大的最大值。旧分母使用旧基准 m，新分母要使用新基准 $m'=\max(m,x)$。旧每项权重满足 $e^{x_j-m'}=e^{x_j-m}e^{m-m'}$，所以无需回看旧元素，只要把旧总和统一乘上 $c=e^{m-m'}$：

$$
m'=\max(m,x),\qquad l'=e^{m-m'}l+e^{x-m'}.
$$

手算先读 0 再读 2：读完 0 时 `(m,l)=(0,1)`；读入 2 后，旧项要从 $e^{0-0}=1$ 改成 $e^{0-2}$，所以得到 `(2,e⁻²+1)`。把旧 l 原封不动加到新项上，会错误地把两个不同的指数基准混在一起。

这一遍结束得到了最终 m,l。若你的任务是输出**每个概率**，还要再读一遍 x 计算 P：因此 online normalizer 将 3-pass 改为 2-pass，并没有凭空消除输出 P 的需要。[Online normalizer 原论文](https://arxiv.org/abs/1805.02867)。

### 3.0c 优化二：既然只要 PV，能否连概率矩阵也省掉

Attention 最终需要的是 value 加权和，不一定需要保存每个概率。对一行 query 保留一个 value 长度的向量 $a=\sum_j e^{x_j-m}v_j$，它是**尚未归一化的分子**。m 改变时，a 与 l 受同一个缩放因子影响：

$$
a'=ca+e^{x-m'}v,\qquad O=a/l\quad(l>0).
$$

手算两个分数 `[0,2]`，两个 value `[1,0]`、`[0,2]`。读完第一个得到 `(m,l,a)=(0,1,[1,0])`；读完第二个得到 `(2,1+e⁻²,[e⁻²,2])`，最终输出约 `[0.1192,1.7616]`。如果分别对两个单元素块做 softmax，每块都产生概率 1，直接相加会得到 `[1,2]`，显然错误。

每读到一个 score/value 就可以更新统计量，最后输出 a/l；无需第二遍回看分数来输出完整 P。但这仍只是逐元素算法，还没有解决 GPU 需要高吞吐矩阵乘法的问题。

### 3.0d 优化三：把单元素更新推广成矩阵 tile

GPU 有容量大但读写相对昂贵的显存 HBM，也有容量小、靠近计算单元的片上存储，例如 shared memory 和寄存器。tile 是能放进片上工作区的一小块矩阵：一次取 R 条 query 和 T 条 key/value，用矩阵乘法计算 $x=Q_iK_j^T/\sqrt{d_k}$，立即消费它，不把全长 $N\times M$ 的 score 或 P 写回 HBM。

每条 query 行各自维护 m,l,a，新块 X 到来时：

$$
m'=\max(m,\operatorname{rowmax}(X)),\quad
c=\exp(m-m'),\quad W=\exp(X-m'),
$$
$$
l'=c\odot l+\operatorname{rowsum}(W),\qquad
a'=c\odot a+WV_j.
$$

m,l,c 是每行一个数，参与 X 或 a 的计算时沿列广播。W 不是归一化概率。所有 KV tile 完成后才输出 $O=a/l$。query 行数 R 和 key 块长 T 互相独立，最后不足一个块时使用真实长度。


<!-- manim-group:flash-tile:start -->
<section class="matrix-steps" data-matrix-group="flash-tile" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 324px; --diagram-mobile-width: 324px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：先消费第一个 score/value，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-01-desktop.png" alt="步骤 1：先消费第一个 score/value" width="648" height="506" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：先消费第一个 score/value</strong> 初始空状态由首个有效元素建立。a 是尚未除以 l 的加权和；全空 tile 必须跳过。 <a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 363px; --diagram-mobile-width: 364px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：最大值变大：同时重标定分子分母，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-02-desktop.png" alt="步骤 2：最大值变大：同时重标定分子分母" width="726" height="604" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：最大值变大：同时重标定分子分母</strong> 旧 a 从 [2,0] 变成 [1,0]，旧 l 从 1 变成 1/2。两者乘同一个因子，所以旧输出比值不变。 <a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 315px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：同一个缩放系数作用于 l 与 a，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-03-desktop.png" alt="步骤 3：同一个缩放系数作用于 l 与 a" width="808" height="556" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：同一个缩放系数作用于 l 与 a</strong> 竖线左侧是标量分母，右侧是两通道分子。每格同时乘 1/2；旧比值 [2,0]/1 与 [1,0]/(1/2) 相同。 <a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 319px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：在新尺度下加入第二条 value，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-04-desktop.png" alt="步骤 4：在新尺度下加入第二条 value" width="922" height="636" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：在新尺度下加入第二条 value</strong> 新 score 等于新最大值，因此指数系数为 1；这一步仍不需要保存概率向量。 <a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 320px; --diagram-mobile-width: 241px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：最后才做分式归一化，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-05-desktop.png" alt="步骤 5：最后才做分式归一化" width="640" height="561" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：最后才做分式归一化</strong> 本例与对 [0,log2] 做 softmax 再乘 V 完全一致。Flash 减少中间矩阵的显存读写，主要点积计算仍随序列长度平方增长。 <a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 329.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-06-desktop.png" target="_blank" rel="noopener" aria-label="步骤 6：推广成一块 query 与一块 key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-06-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-06-desktop.png" alt="步骤 6：推广成一块 query 与一块 key" width="912" height="729" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 6：推广成一块 query 与一块 key</strong> 每条 query 各自维护 m,l,a。W 仅在当前 tile 存在；W V_j 累计到 a，再处理下一块。 <a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-06-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 587px; --diagram-mobile-width: 324.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-07-desktop.png" target="_blank" rel="noopener" aria-label="步骤 7：每条 query 累积自己的 l 与 a，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-07-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-07-desktop.png" alt="步骤 7：每条 query 累积自己的 l 与 a" width="1174" height="666" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 7：每条 query 累积自己的 l 与 a</strong> 竖线左边各行的分母独立累加，右边各行的分子独立累加。这里旧 m=0，新 m′=log2，旧状态已同时重标定一半。 <a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-07-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 362px; --diagram-mobile-width: 304px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-08-desktop.png" target="_blank" rel="noopener" aria-label="步骤 8：所有 tile 完成后逐行归一化，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-08-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-08-desktop.png" alt="步骤 8：所有 tile 完成后逐行归一化" width="724" height="660" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 8：所有 tile 完成后逐行归一化</strong> 两条 query 本例分母都是 2，各自的 [2,2] 除以自己的分母得到 [1,1]。一般情况下不同行的分母不同，不能混用。 <a href="/images/notes/attention-from-softmax-to-kda/manim/flash-tile-08-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:flash-tile:end -->


从头到尾全遮蔽的行保留 `l=0,a=0`，最终按约定返回零。首次遇到全遮蔽 tile 时，代码用安全基准绕开 $-\infty-(-\infty)$；若已有合法历史、当前块全遮蔽，则历史统计量保持不变。

| 阶段 | 保存什么 | 改进了什么 | 仍需付出什么 |
|---|---|---|---|
| 3-pass safe softmax | 最终 P 和行统计量 | 指数不溢出 | 三次读入分数 |
| online normalizer | 最终 P 和行统计量 | m/l 合并计算 | 仍需第二遍生成 P |
| 融合 value 加权和 | m,l,a | 不必输出全 P | 所有 query-key 点积仍存在 |
| 分块、融合、片上计算 | 每块 score 与每行统计量 | 避免 N×M 中间量在 HBM 往返 | tile 选择、并行度、同步成本 |
| FA2/FA3 | 数学仍相同 | 更好的工作分配和硬件重叠 | 受具体硬件和形状制约 |

因此 Flash 的关键不是取消 softmax，也不是把全部 Attention 算术改成线性复杂度。它保留同一 Attention 定义，利用统计量可合并、分块和融合来减少 IO。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)。

### 3.1 用三个统计量完成全局归一化

online softmax 最值得记住的不是循环顺序，而是一个不变量：当 `l > 0` 时，`a / l` 是截至当前已处理键集合的正确输出。新块带来更大的最大值时，只要旧分子和分母一起重标定，就可以继续累积。

```python
# m/l 每条 query 一个数，a 每条 query 一个 value 向量；新最大值改变时一起重标定。
def online_update(m,l,a,x,v):
    # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    new_m=np.maximum(m,np.max(x,axis=-1))
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
    safe=np.where(np.isfinite(new_m),new_m,0.)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    rescale=np.exp(m-safe)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    weights=np.exp(x-safe[...,None])
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    new_l=rescale*l+weights.sum(-1)
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    new_a=rescale[...,None]*a+weights@v
    return new_m,new_l,new_a
```

### 3.2 从矩阵分块看到统计量更新

图中 query 行按 R=3 分块、key 列按 T=4 分块。格子中的 i,j 是 query/key 的全局 token 位置；黑色是因果不可见区域。
右图是后 3 行处理每个 KV tile 后保存的 m,l,a。a 为未归一化输出的一个 value 通道，仅在所有 tile 完成后除以 l。
这同时说明为什么 tile-local softmax 不能独立拼接：不同 tile 必须共享并更新指数基准 m。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-02.png" alt="图 2：左侧为因果 score 分块；右侧为 query 6–8 在依次消费 KV tile 后的 m、l 与一个输出通道的未归一化分子。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 2：左侧为因果 score 分块；右侧为 query 6–8 在依次消费 KV tile 后的 m、l 与一个输出通道的未归一化分子。</figcaption>
</figure>

### 3.3 从 FA1 到 FA2：省下 IO 之后，瓶颈转向工作划分

FA1 让中间概率不必在 HBM 中完整物化。此后，即使读写量已经降低，GPU 仍可能没有足够多的独立工作块，也可能在 warp 之间花费过多时间交换部分结果。

FA2 继续沿着相同数学定义调整实现：让 query tile 的输出归属更清楚，增加长序列上的并行工作，减少每轮归一化和跨 warp 通信。这里优化的是执行开销与资源利用率，attention pair 的数量仍然没有改变。


#### FA1：IO-aware tiling + recomputation

附录中的 FA1 教学实现采用 K/V tile 外层、Q tile 内层，按论文风格保存归一化 O 并更新 m,l。工程上的 SRAM 容量约束决定块形状。
传统 HBM 中间读写量含 $\Theta(N^2)$，FA 在理想两级存储、$d\le M_s\le Nd$ 条件下有 $O(Nd+N^2d^2/M_s)$ 元素传输上界（设 dk=dv=d，$M_s$ 为快速存储元素容量）。这是模型化 IO 分析，不是任意 NumPy 程序的实际 cache miss 数。

#### FA2：更少标量工作，更好的并行与 warp 分工

Q tile 外层令一个 block/CTA 拥有输出行，增加沿 query 长度的并行度；保留未归一化 a，最终只除一次 l，减少非 matmul FLOPs。
典型 split-Q 分工让各 warp 写不同 query 行，避免 split-K 中为同一输出做跨 warp 归约。实际调度随前向/反向、形状及实现改变。
来源：[FA1](https://arxiv.org/abs/2205.14135)、[FA2](https://arxiv.org/abs/2307.08691)、[作者 FA2 说明](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)。


## 4. FlashAttention-3：把数据搬运、矩阵乘法与 softmax 重叠起来

FA3 面向 Hopper 的关键组合是：TMA 搬运、WGMMA 异步矩阵乘法、producer/consumer warp specialization、不同 warpgroup 的 ping-pong，以及同一 warpgroup 内 GEMM/softmax 的重叠。
TMA 预取下一 K/V tile 时，消费者处理当前 tile；softmax 依赖该 tile 的 QK 完成，PV 依赖 softmax 完成，下一 tile 的 QK 可与已有工作部分重叠。双缓冲只有在消费者完成后才可重用；需要 barrier 的阶段/计数，不能只把循环改成异步函数。

附录中的 `flash3_pipeline_model` 是**可执行的依赖/双缓冲教学模型**：顺序模拟预取和消费，并输出相同的 attention。它没有实现异步 GPU 指令，因此不应从它的 CPU 时间推断 FA3 性能。FA1/2 也遵循这一边界。官方 Hopper 内核验证在附录。

### 4.1 低精度路径

FP16/BF16 也有舍入误差，FP8 进一步引入量化。block scaling 按块选 scale，减少离群值对其他值精度的挤占。对 Q/K 右乘相同正交矩阵 R 保持 $(QR)(KR)^T=QK^T$；随机符号 Hadamard 可分散离群值。本例只演示正交不变性和**均匀 INT8 风格量化**的误差，不将其冒充 FP8 E4M3/E5M2。


<!-- manim-group:rotation:start -->
<section class="matrix-steps" data-matrix-group="rotation" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 303px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/rotation-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：同一个正交矩阵旋转 query，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/rotation-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/rotation-01-desktop.png" alt="步骤 1：同一个正交矩阵旋转 query" width="912" height="573" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：同一个正交矩阵旋转 query</strong> 这里 Q 与 K 使用同一个 R，区别于 RoPE 中按各自 token 位置旋转。 <a href="/images/notes/attention-from-softmax-to-kda/manim/rotation-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 258px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/rotation-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：key 也使用同一个旋转，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/rotation-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/rotation-02-desktop.png" alt="步骤 2：key 也使用同一个旋转" width="912" height="565" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：key 也使用同一个旋转</strong> 同时旋转保持点积：原点积为 1，旋转后 [0,1]·[-1,1] 仍为 1。 <a href="/images/notes/attention-from-softmax-to-kda/manim/rotation-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 401px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/rotation-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：中间的正交因子相消，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/rotation-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/rotation-03-desktop.png" alt="步骤 3：中间的正交因子相消" width="912" height="586" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：中间的正交因子相消</strong> 要求 RRᵀ=I。Hadamard 旋转也需归一化；后续量化会引入额外误差，不能宣称量化后仍精确相等。 <a href="/images/notes/attention-from-softmax-to-kda/manim/rotation-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:rotation:end -->


误差降低取决于输入，不能断言每个随机样本都改善；生产 FP8 还需精确规定缩放粒度、累加格式、P/V 的量化策略。
来源：[FA3 论文](https://arxiv.org/abs/2407.08608)、[作者硬件说明](https://tridao.me/blog/2024/flash3/)。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-03.png" alt="图 3：满足数据依赖的示意流水线。L 是搬运，QK/PV 是矩阵乘法，SM 是 softmax；横轴不是实测周期。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 3：满足数据依赖的示意流水线。L 是搬运，QK/PV 是矩阵乘法，SM 是 softmax；横轴不是实测周期。</figcaption>
</figure>

走到这里，FlashAttention 的路线已经完整：先减少中间结果的显存往返，再改进并行分工，随后让不同硬件单元尽可能同时工作。它不会替我们决定 KV 如何分配，也不会自动发现两条请求共享了什么。下面把视角从一个 kernel 移到整个推理过程。

## 5. PagedAttention：KV 必须增长，但不必物理连续

decode 第 t 步需要读取此前所有 KV，softmax 复杂度仍是 $O(t(d_k+d_v))$/头。分页解决的是显存管理，而不是把这一步变成 O(1)。

设页容量 P 个 token，token 位置 t 映射为：
$$
logical\_page=\lfloor t/P\rfloor,\quad offset=t\bmod P,
$$
$$
physical\_page=block\_table[logical\_page].
$$

KV 池布局可设 `[num_pages,Hkv,P,dk]` 和 `[num_pages,Hkv,P,dv]`；真实 kernel 常为向量化加载变更内部布局。
一条请求只保存页号表和有效 token 数；读页时用实际尾页长度屏蔽未写槽位，不能把未初始化 KV 读进 softmax。

生命周期：申请空页 → 写 token → 分叉共享并增加引用数 → 修改共享尾页前 COW → 请求结束减引用 → 0 引用归还空闲池。
写入共享且未满的尾页必须先复制；已满旧页不会再写入，只需为新 token 申请新页。
分配失败应不破坏原页表/引用数。这里无 swap、异步 DMA、并发调度，OOM 明确抛出。

每请求尾页浪费最多 P−1 个 token 槽（按逻辑尾页计；共享改变物理总量）。有 m 条不共享请求、总有效 T token 时，分配槽数小于 T+mP；并不是“完全零碎片”。

附录实现从离散页直接用 online softmax 累积，**不先 gather 全部 KV 再算 attention**。gather 只用于测试 oracle。来源：[PagedAttention 论文](https://arxiv.org/abs/2309.06180)。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-04.png" alt="图 4：请求通过页表把逻辑块映射到物理页，逻辑 token 顺序不依赖物理地址顺序。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 4：请求通过页表把逻辑块映射到物理页，逻辑 token 顺序不依赖物理地址顺序。</figcaption>
</figure>

### 5.1 分页与 Flash 可以使用同一种归约思路

分页 attention 不要求先把离散页拼成一个连续大数组。每次读取一页 K/V，计算这一页对 query 的贡献，再合并 online softmax 统计量即可。分页负责找到数据，归约负责得到数学上正确的答案。

另一个需要单独验证的地方是共享：两条生成分支可以共同引用已经写完的页；如果它们共享的尾页尚未写满，任何一条分支继续追加之前都必须复制该页。否则一个分支会悄悄改写另一个分支的上下文。

## 6. RadixAttention：已经算过的前缀，能否直接接着用

压缩 radix tree 的一条边保存一段 token 序列，而不是一个字符。key 必须是**token IDs 及计算上下文**，不能仅对可见字符串做粗糙匹配。相同文本在不同 tokenizer、模型权重、LoRA、位置偏移、多模态输入或 cache 精度下，不保证 KV 相同。

以 token 路径 `[1,2,3,4]` 与 `[1,2,5]` 为例：插入第二条时，把边 `[1,2,3,4]` 分为公共边 `[1,2]`，再分出 `[3,4]` 和 `[5]`。
查询 `[1,2,3,9]` 可命中前三个 token，剩余 `[9]` 重算。即使命中停在边的中间，也能返回匹配部分。

缓存保存的是**每层历史 token 的 K/V**，不是最后一次请求的 attention 输出。已有 prefix 的 query 输出无需重算；新 suffix 的 query 仍要 attend prefix KV。
共享前缀长度 L，新增 U token：单层新增 attention pair 数约 $UL+U(U+1)/2$，而非零；跳过的是 prefix 自身投影及 $L(L+1)/2$ 的因果 attention pair。

本实现支持：压缩边插入/分裂、最长前缀查找、上下文 namespace、活动路径 pin/unpin、叶子 LRU 淘汰、容量控制。
KV 在节点中按边 token 存储；为了把树逻辑与分页逻辑分开，教学版节点持有 ndarray。生产系统通常持有共享 KV 池的索引/页引用，split 需要维护 ownership/refcount。本实现不会宣称数组复制已获得生产共享内存性能。
来源：[SGLang 论文](https://arxiv.org/abs/2312.07104)、[项目作者说明](https://www.lmsys.org/blog/2024-01-17-sglang/)。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-05.png" alt="图 5：压缩 radix 树把公共 token 前缀 [1, 2] 保存为一段边，分叉后的 [3, 4] 与 [5] 分别存储。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 5：压缩 radix 树把公共 token 前缀 [1, 2] 保存为一段边，分叉后的 [3, 4] 与 [5] 分别存储。</figcaption>
</figure>

### 6.1 验证缓存复用，必须让 KV 真正依赖上下文

附录验证运行两层小型 causal attention。第二层的 K/V 来自第一层的上下文输出，因此相同 token 在不同历史中不一定有相同 KV。
每层分别 cache prefix K/V；新请求只处理 suffix，采用绝对 query 起点 L 的 mask。最后与整段两层重算结果比较。
这里只做 forward 推理，权重固定；无 dropout，cache 使用训练态的随机 dropout 结果会破坏一致性。

PagedAttention 和 RadixAttention 至此也可以分开理解：前者解决 KV 放在哪里、怎样共享与回收；后者解决哪些历史计算可以继续使用。它们都保留显式历史，因此历史足够长时，新的 query 仍需要读取很多 K/V。

如果我们愿意改变模型的注意力形式，是否可以让历史不再随着 token 数不断扩张？这引出第二条路线。

## 7. Linear attention：用一个矩阵状态汇总历史

### 7.1 先从你已知的 Attention 出发

一条 query 的输出，本质上是“每个 value 乘一个非负相似度，再除以这些相似度的总和”。softmax 选择的相似度为指数点积。现在先改变相似度的定义，再考虑怎样计算。

令 $\bar q=\phi(q),\bar k=\phi(k)$，用 $\kappa(q,k)=\bar q^T\bar k$ 作为新相似度。本文用逐元素 $\phi(x)=\mathrm{ELU}(x)+1$，即正数分支为 x+1，负数分支为 $e^x$。例如 `[-1,0,2]` 变为 `[e⁻¹,1,3]`；映射后的数均为正，点积因而非负。

“核”在这里就是相似度函数的名称，不要求你先学核方法。一般特征映射可把 $d_k$ 维变成 r 维；这里逐元素映射，所以 r=$d_k$。value 不需要跟着做这个正值映射，它仍可含负数。

**这一步改变了模型**。普通 softmax 的 $\exp(q^Tk/\sqrt{d_k})$ 不能用矩阵结合律直接改写成有限维的 $\phi(q)^T\phi(k)$。本文的 ELU+1 核也不宣称等于它。后面两条计算路径的等价性，是在同一个新核内部成立的。

### 7.2 暂不考虑因果：两条路径给出相同分子

路径一仍像普通 Attention：形成每对 token 的核分数 $A_\kappa=\bar Q\bar K^T$，再乘 V。路径二交换**乘法的括号**：先计算 $S=\bar K^TV$，再让 $\bar Q$ 读取 S。

$$
(\bar Q\bar K^T)V=\bar Q(\bar K^TV)=\bar QS.
$$

矩阵顺序没有交换，改变的只有先做哪一次乘法。路径一中间结果是 $N\times M$，路径二是 $r\times d_v$。后者的行列都没有历史长度 M：这就是固定大小状态出现的位置。


<!-- manim-group:linear-association:start -->
<section class="matrix-steps" data-matrix-group="linear-association" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 302.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：显式路径先生成所有相似度，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-association-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-association-01-desktop.png" alt="步骤 1：显式路径先生成所有相似度" width="1080" height="589" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：显式路径先生成所有相似度</strong> 这里使用非负特征内积，不是 softmax。相似度矩阵仍随 query×key 数增长。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 301px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：改变括号，先汇总 KV 外积，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-association-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-association-02-desktop.png" alt="步骤 2：改变括号，先汇总 KV 外积" width="996" height="651" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：改变括号，先汇总 KV 外积</strong> 收缩历史 token 轴，只留下特征×value 状态。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 302px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：query 读取状态，分子相同，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-association-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-association-03-desktop.png" alt="步骤 3：query 读取状态，分子相同" width="912" height="594" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：query 读取状态，分子相同</strong> 两条路径的分子经 NumPy 检查相同。归一化还需 z=K̄ᵀ1；不能把 softmax 从乘法中移过去。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 510px; --diagram-mobile-width: 375px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：第 1 步：逐 token 缓存与固定状态，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-association-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-association-04-desktop.png" alt="步骤 4：第 1 步：逐 token 缓存与固定状态" width="1020" height="638" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：第 1 步：逐 token 缓存与固定状态</strong> 左侧逐行保存每个 token 的 key/value；右侧将它们叠加成同形 S,z。新 query 在左侧读取所有历史行，在右侧只读当前 S,z；这种压缩不是 softmax 的无损替代。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 510px; --diagram-mobile-width: 375px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：第 3 步：逐 token 缓存与固定状态，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-association-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-association-05-desktop.png" alt="步骤 5：第 3 步：逐 token 缓存与固定状态" width="1020" height="737" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：第 3 步：逐 token 缓存与固定状态</strong> 左侧逐行保存每个 token 的 key/value；右侧将它们叠加成同形 S,z。新 query 在左侧读取所有历史行，在右侧只读当前 S,z；这种压缩不是 softmax 的无损替代。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 510px; --diagram-mobile-width: 375px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-06-desktop.png" target="_blank" rel="noopener" aria-label="步骤 6：第 6 步：逐 token 缓存与固定状态，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-association-06-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-association-06-desktop.png" alt="步骤 6：第 6 步：逐 token 缓存与固定状态" width="1020" height="989" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 6：第 6 步：逐 token 缓存与固定状态</strong> 左侧逐行保存每个 token 的 key/value；右侧将它们叠加成同形 S,z。新 query 在左侧读取所有历史行，在右侧只读当前 S,z；这种压缩不是 softmax 的无损替代。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-association-06-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:linear-association:end -->


### 7.3 分母也要一起变形，不能只讲分子

对 query i，所有相似度的总和为 $\sum_j\bar q_i^T\bar k_j=\bar q_i^Tz$，其中 $z=\sum_j\bar k_j=\bar K^T\mathbf1$。所以完整输出为：

$$
O_i=\frac{\bar q_i^TS}{\bar q_i^Tz+\epsilon}.
$$

S 保存“key 特征与 value 的关联总和”，z 保存“key 特征总量”。如果只有 S 而不除分母，重复加入同一批 token 会让分子越加越大。加入 z 后，同样的重复也增大总权重，因此得到归一化的混合结果。它仍不能恢复被状态叠加丢失的独立 token 身份。

数学讲解可先设 $\epsilon=0$ 且分母为正。代码加 `1e-12` 防止极小分母；这会使归一化权重和略小于 1，是一个明确的数值保护，不应称作完全不改变公式。


<!-- manim-group:linear-normalizer:start -->
<section class="matrix-steps" data-matrix-group="linear-normalizer" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 308px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：第 1 步：key 与 value 外积，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-01-desktop.png" alt="步骤 1：第 1 步：key 与 value 外积" width="996" height="593" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：第 1 步：key 与 value 外积</strong> key 为 2×1 列向量，value 转为 1×3 行向量；输出 2×3，每个格子是对应 key 分量乘 value 分量。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 587px; --diagram-mobile-width: 266px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：第 1 步：逐格累积状态，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-02-desktop.png" alt="步骤 2：第 1 步：逐格累积状态" width="1174" height="573" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：第 1 步：逐格累积状态</strong> 新旧状态形状相同。历史不再分 token 保存，而是叠加进同一组格子；S₀ 的零值留白。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 335px; --diagram-mobile-width: 311.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：第 1 步：累积 key 特征总量，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-03-desktop.png" alt="步骤 3：第 1 步：累积 key 特征总量" width="670" height="587" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：第 1 步：累积 key 特征总量</strong> z 保存 key 特征之和，维度为 2×1；它用于计算 query 对全部历史的总权重。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 288px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：第 1 步：query 读取分子，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-04-desktop.png" alt="步骤 4：第 1 步：query 读取分子" width="1080" height="587" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：第 1 步：query 读取分子</strong> query 收缩状态的 key 特征轴；每列留下一个输出通道。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 372px; --diagram-mobile-width: 288px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：第 1 步：query 读取分母，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-05-desktop.png" alt="步骤 5：第 1 步：query 读取分母" width="744" height="574" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：第 1 步：query 读取分母</strong> 分母是一个标量。本例第 2 步为 1×3+1×3=6；它不是需要求逆的矩阵。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 347.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-06-desktop.png" target="_blank" rel="noopener" aria-label="步骤 6：第 1 步：每个通道除以同一分母，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-06-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-06-desktop.png" alt="步骤 6：第 1 步：每个通道除以同一分母" width="808" height="569" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 6：第 1 步：每个通道除以同一分母</strong> 各通道都除以 1，得到 [3.0, 1.0, 2.0]。这些数直接使用非负特征 k̄、q̄；零分量用于简化示例，并非有限输入经 ELU+1 的精确输出。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-06-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 308px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-07-desktop.png" target="_blank" rel="noopener" aria-label="步骤 7：第 2 步：key 与 value 外积，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-07-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-07-desktop.png" alt="步骤 7：第 2 步：key 与 value 外积" width="996" height="593" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 7：第 2 步：key 与 value 外积</strong> key 为 2×1 列向量，value 转为 1×3 行向量；输出 2×3，每个格子是对应 key 分量乘 value 分量。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-07-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 587px; --diagram-mobile-width: 266px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-08-desktop.png" target="_blank" rel="noopener" aria-label="步骤 8：第 2 步：逐格累积状态，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-08-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-08-desktop.png" alt="步骤 8：第 2 步：逐格累积状态" width="1174" height="572" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 8：第 2 步：逐格累积状态</strong> 新旧状态形状相同。历史不再分 token 保存，而是叠加进同一组格子；S₀ 的零值留白。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-08-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 335px; --diagram-mobile-width: 311.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-09-desktop.png" target="_blank" rel="noopener" aria-label="步骤 9：第 2 步：累积 key 特征总量，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-09-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-09-desktop.png" alt="步骤 9：第 2 步：累积 key 特征总量" width="670" height="587" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 9：第 2 步：累积 key 特征总量</strong> z 保存 key 特征之和，维度为 2×1；它用于计算 query 对全部历史的总权重。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-09-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 288px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-10-desktop.png" target="_blank" rel="noopener" aria-label="步骤 10：第 2 步：query 读取分子，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-10-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-10-desktop.png" alt="步骤 10：第 2 步：query 读取分子" width="1080" height="587" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 10：第 2 步：query 读取分子</strong> query 收缩状态的 key 特征轴；每列留下一个输出通道。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-10-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 372px; --diagram-mobile-width: 288px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-11-desktop.png" target="_blank" rel="noopener" aria-label="步骤 11：第 2 步：query 读取分母，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-11-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-11-desktop.png" alt="步骤 11：第 2 步：query 读取分母" width="744" height="574" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 11：第 2 步：query 读取分母</strong> 分母是一个标量。本例第 2 步为 1×3+1×3=6；它不是需要求逆的矩阵。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-11-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 404px; --diagram-mobile-width: 347.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-12-desktop.png" target="_blank" rel="noopener" aria-label="步骤 12：第 2 步：每个通道除以同一分母，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-12-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-12-desktop.png" alt="步骤 12：第 2 步：每个通道除以同一分母" width="808" height="569" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 12：第 2 步：每个通道除以同一分母</strong> 各通道都除以 6，得到 [2.0, 2.5, 2.0]。这些数直接使用非负特征 k̄、q̄；零分量用于简化示例，并非有限输入经 ELU+1 的精确输出。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-normalizer-12-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:linear-normalizer:end -->


### 7.4 把状态写开：外积不是点积

现在读入一个 token：$\bar k_1=[1,2]^T,v_1=[3,1,2]^T$。外积 $\bar k_1v_1^T$ 是一个 $2\times3$ 矩阵：第一行等于 value，第二行等于两倍 value。它为每个 key 特征通道各存了一份缩放后的 value。

再读入 $\bar k_2=[2,1]^T,v_2=[1,4,2]^T$，把第二个外积加到同形的 S 上。此时 $S_2$ 的两行分别为 `[5,9,6]` 和 `[7,6,6]`，而 $z_2=[3,3]^T$。令 $\bar q_2=[1,1]^T$，从状态读取的分子为 `[12,15,12]`，分母为 6，结果为 `[2,2.5,2]`。

用显式 token 权重再验一次：query 与两个 key 的点积都是 3，因此两个 value 各占一半，$(v_1+v_2)/2=[2,2.5,2]$。两种算法确实得到同一个答案。


<!-- manim-group:linear-state:start -->
<section class="matrix-steps" data-matrix-group="linear-state" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 834px; --diagram-mobile-width: 464px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：六步状态链：写入与读取，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-01-desktop.png" alt="步骤 1：六步状态链：写入与读取" width="1668" height="703" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：六步状态链：写入与读取</strong> 下方外积沿箭头写入加号；上方 query 从更新后的状态读取分子。每个状态都是同样的 2×2；z 同时递推、最后参与归一化。手机分两段展示，重复的 S₃ 表示同一个衔接状态。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 214px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：时间链：S0 → S1，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-02-desktop.png" alt="步骤 2：时间链：S0 → S1" width="922" height="596" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：时间链：S0 → S1</strong> 第 1 个 token 写入同一个 2×2 状态。其 key/value 均为 [1.0, 0.0]；z 同时累加 key，当前为 [1.0, 0.0]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 268px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：时间链：query 读取 S1，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-03-desktop.png" alt="步骤 3：时间链：query 读取 S1" width="912" height="649" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：时间链：query 读取 S1</strong> 本帧展示分子读取；分母为 1。沿时间依次读完 1…6 步，状态始终保持 2×2；不是六个历史 token 的无损存储。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 217px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：时间链：S1 → S2，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-04-desktop.png" alt="步骤 4：时间链：S1 → S2" width="922" height="596" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：时间链：S1 → S2</strong> 第 2 个 token 写入同一个 2×2 状态。其 key/value 均为 [0.0, 1.0]；z 同时累加 key，当前为 [1.0, 1.0]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 271px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：时间链：query 读取 S2，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-05-desktop.png" alt="步骤 5：时间链：query 读取 S2" width="912" height="649" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：时间链：query 读取 S2</strong> 本帧展示分子读取；分母为 2。沿时间依次读完 1…6 步，状态始终保持 2×2；不是六个历史 token 的无损存储。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 217px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-06-desktop.png" target="_blank" rel="noopener" aria-label="步骤 6：时间链：S2 → S3，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-06-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-06-desktop.png" alt="步骤 6：时间链：S2 → S3" width="922" height="597" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 6：时间链：S2 → S3</strong> 第 3 个 token 写入同一个 2×2 状态。其 key/value 均为 [1.0, 1.0]；z 同时累加 key，当前为 [2.0, 2.0]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-06-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 271px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-07-desktop.png" target="_blank" rel="noopener" aria-label="步骤 7：时间链：query 读取 S3，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-07-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-07-desktop.png" alt="步骤 7：时间链：query 读取 S3" width="912" height="651" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 7：时间链：query 读取 S3</strong> 本帧展示分子读取；分母为 4。沿时间依次读完 1…6 步，状态始终保持 2×2；不是六个历史 token 的无损存储。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-07-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 218px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-08-desktop.png" target="_blank" rel="noopener" aria-label="步骤 8：时间链：S3 → S4，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-08-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-08-desktop.png" alt="步骤 8：时间链：S3 → S4" width="922" height="596" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 8：时间链：S3 → S4</strong> 第 4 个 token 写入同一个 2×2 状态。其 key/value 均为 [2.0, 0.0]；z 同时累加 key，当前为 [4.0, 2.0]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-08-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 272px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-09-desktop.png" target="_blank" rel="noopener" aria-label="步骤 9：时间链：query 读取 S4，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-09-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-09-desktop.png" alt="步骤 9：时间链：query 读取 S4" width="912" height="649" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 9：时间链：query 读取 S4</strong> 本帧展示分子读取；分母为 6。沿时间依次读完 1…6 步，状态始终保持 2×2；不是六个历史 token 的无损存储。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-09-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 217px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-10-desktop.png" target="_blank" rel="noopener" aria-label="步骤 10：时间链：S4 → S5，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-10-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-10-desktop.png" alt="步骤 10：时间链：S4 → S5" width="922" height="597" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 10：时间链：S4 → S5</strong> 第 5 个 token 写入同一个 2×2 状态。其 key/value 均为 [0.0, 2.0]；z 同时累加 key，当前为 [4.0, 4.0]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-10-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 271px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-11-desktop.png" target="_blank" rel="noopener" aria-label="步骤 11：时间链：query 读取 S5，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-11-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-11-desktop.png" alt="步骤 11：时间链：query 读取 S5" width="912" height="651" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 11：时间链：query 读取 S5</strong> 本帧展示分子读取；分母为 8。沿时间依次读完 1…6 步，状态始终保持 2×2；不是六个历史 token 的无损存储。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-11-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 218px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-12-desktop.png" target="_blank" rel="noopener" aria-label="步骤 12：时间链：S5 → S6，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-12-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-12-desktop.png" alt="步骤 12：时间链：S5 → S6" width="922" height="597" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 12：时间链：S5 → S6</strong> 第 6 个 token 写入同一个 2×2 状态。其 key/value 均为 [2.0, 1.0]；z 同时累加 key，当前为 [6.0, 5.0]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-12-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 271px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-13-desktop.png" target="_blank" rel="noopener" aria-label="步骤 13：时间链：query 读取 S6，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-state-13-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-state-13-desktop.png" alt="步骤 13：时间链：query 读取 S6" width="912" height="651" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 13：时间链：query 读取 S6</strong> 本帧展示分子读取；分母为 11。沿时间依次读完 1…6 步，状态始终保持 2×2；不是六个历史 token 的无损存储。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-state-13-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:linear-state:end -->


### 7.5 因果性：读取 t 时，只能把前缀放进状态

若先汇总全序列 S 再计算第一个 query，它就读到了未来。因果版本为每个时间步维护前缀：

$$
S_t=S_{t-1}+\bar k_tv_t^T,\quad z_t=z_{t-1}+\bar k_t,\quad
o_t^T=\frac{\bar q_t^TS_t}{\bar q_t^Tz_t+\epsilon}.
$$

初始 $S_0=0,z_0=0$。先写入当前 token，再读出当前输出，对应包含对角线的 causal Attention；如果先读后写，含义就变成只看严格过去。上图第一步输出只能是 $v_1$，直到第二步才允许混合 $v_2$。

**状态固定大小**说的是推理时只须携带最新 S,z。返回全部 N 个输出仍占 $O(Nd_v)$；教学反向代码保存每一步状态，还会占 $O(Nrd_v)$ 激活。固定状态不是训练总内存为常数。

### 7.6 分块：历史状态与本块三角矩阵各做一部分

对当前 C 个 token，前面完整块已压进 $S_{before},z_{before}$；块内尚未发生的 token 则用下三角 mask 排除。分子是“读历史状态”加“读本块可见 value”，分母也相应相加，最后统一相除。块尾再将整个块的写入汇总到 S,z，传给下一块。


<!-- manim-group:linear-chunk:start -->
<section class="matrix-steps" data-matrix-group="linear-chunk" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 240.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：当前块先读取旧状态，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-01-desktop.png" alt="步骤 1：当前块先读取旧状态" width="912" height="587" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：当前块先读取旧状态</strong> 旧状态只汇总本块以前的 token；不能包含本块未来。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 239.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：本块内部保留下三角，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-02-desktop.png" alt="步骤 2：本块内部保留下三角" width="912" height="572" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：本块内部保留下三角</strong> 对角线对应当前 token，下三角对应本块更早 token，右上角严格为零。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 281.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：先相加分子，分母也相加，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-03-desktop.png" alt="步骤 3：先相加分子，分母也相加" width="922" height="642" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：先相加分子，分母也相加</strong> 最后逐行计算 O_i=N_i/l_i。分别归一化两个部分后再相加通常是错的。 <a href="/images/notes/attention-from-softmax-to-kda/manim/linear-chunk-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:linear-chunk:end -->


### 7.7 为什么叫 Linear，代价又在哪里

先看非因果自注意力 $M=N$。显式核分数路径主乘法为 $2N^2r+2N^2d_v$；状态路径形成 S 约 $2Nrd_v$，读 S 再约 $2Nrd_v$，加上 z、分母与逐元素映射的线性工作，主项约 $4Nrd_v$。当 r、$d_v$ 固定时，它关于序列长度 N 是线性的。

因果递推中，一步写外积有 $rd_v$ 个乘法并加到状态上，约 $2rd_v$；query 读状态再约 $2rd_v$；z 与分母另需 $O(r)$。因此仍为 $\Theta(Nrd_v)$，不能把隐藏的维度成本省略成“每步 O(1) 所以一定很快”。若 $r=d_v=128$，一个状态已有 16384 个数，短序列上未必划算；Python 小循环也不等于高吞吐 GPU 实现。

普通 KV cache 给每个历史 token 留一份独立记录，状态 S 把不同 token 写入同一组格子。不同 key 特征相近时，关联会互相干扰；扩大 r 可以增加状态容量，同时增加运算量。模型质量和实际速度必须分别测量，不能由结合律推导直接宣布保持 softmax 的能力。

以上依据 [Linear Transformers 原论文](https://proceedings.mlr.press/v119/katharopoulos20a.html)，数值例子与矩阵展开为本文推导。接下来 DeltaNet 所回答的问题是：同一个 key 的目标 value 变了，还要继续把新旧值机械相加吗？

## 8. DeltaNet：不重复写入已经记住的内容

不归一化的加法状态 $S_t=S_{t-1}+k_tv_t^T$ 会重复叠加相同 key。Delta rule 在每个 token 上最小化瞬时损失：

$$
\ell_t(S)=\tfrac12\|k_t^TS-v_t^T\|_2^2,\quad
\nabla_S\ell_t=k_t(k_t^TS-v_t^T).
$$

梯度步展开为：
$$
\widehat v_t=S_{t-1}^Tk_t,\quad e_t=v_t-\widehat v_t,
$$
$$
S_t=S_{t-1}+\beta_tk_te_t^T=(I-\beta_tk_tk_t^T)S_{t-1}+\beta_tk_tv_t^T,
$$
$$
o_t=S_t^Tq_t.
$$

1. 读取 k 所指方向，得到 `[dv]` 预测。
2. 残差只包含需要补写的信息。
3. `k[:,None] * e[None,:]` 得到 `[dk,dv]` 秩一更新。
4. q 再从新状态读取。这里没有 softmax 概率，也不沿用Linear attention 的 z 分母。



<!-- manim-group:delta-update:start -->
<section class="matrix-steps" data-matrix-group="delta-update" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 216px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：key 读取已有预测，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/delta-update-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/delta-update-01-desktop.png" alt="步骤 1：key 读取已有预测" width="912" height="570" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：key 读取已有预测</strong> key 查询已有状态；这里不是用 query 读最终输出。 <a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 460px; --diagram-mobile-width: 258px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：目标减预测，得到残差，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/delta-update-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/delta-update-02-desktop.png" alt="步骤 2：目标减预测，得到残差" width="920" height="491" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：目标减预测，得到残差</strong> 本例残差 [1,−1]：第一通道不足，第二通道过多。 <a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 414px; --diagram-mobile-width: 263px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：只写残差外积，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/delta-update-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/delta-update-03-desktop.png" alt="步骤 3：只写残差外积" width="828" height="576" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：只写残差外积</strong> 有符号残差允许增加或减少记忆；不是把完整 value 一直累加。 <a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 200px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：把修正写回状态，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/delta-update-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/delta-update-04-desktop.png" alt="步骤 4：把修正写回状态" width="922" height="571" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：把修正写回状态</strong> 本例单位 key、β=1，写入后沿该 key 的预测精确变成目标 [2,1]。 <a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 283px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-05-desktop.png" target="_blank" rel="noopener" aria-label="步骤 5：query 读取更新后的状态，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/delta-update-05-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/delta-update-05-desktop.png" alt="步骤 5：query 读取更新后的状态" width="912" height="582" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 5：query 读取更新后的状态</strong> 最终输出的 query 可以与写入 key 不同；读取的是更新后的 S′。 <a href="/images/notes/attention-from-softmax-to-kda/manim/delta-update-05-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:delta-update:end -->


若 $\|k\|=1,\beta=1$，更新后 $S_t^Tk=v$；与 k 正交的 key 方向保留。一般非正交 keys 会相互干扰，不能理解为无损字典。转移沿 k 的特征值为 $1-\beta\|k\|^2$，正交方向为 1；$\beta\|k\|^2\in[0,2]$ 保证单步该矩阵不扩张。本文用单位范数 k 和 $\beta\in[0,1]$。
来源：[DeltaNet 原论文](https://arxiv.org/abs/2406.06484)、[作者推导](https://sustcsonglin.github.io/blog/2024/deltanet-1/)。

### 8.1 把“更新记忆”看成一次局部学习

Delta rule 的状态本身充当一组快速更新的权重。一次写入的目标不是提高某个相似度，而是纠正当前 key 对应的 value 预测。

这和普通模型训练的时间尺度不同：模型参数决定如何生成 q、k、v 和写入率，状态 S 则在每个序列内部不断变化。序列结束后，状态可以清空；学习到的模型参数仍然保留。这个视角有助于理解为什么这里既能使用回归损失推导更新，又能把整个递推放进更大模型里端到端训练。

## 9. 从 Gated DeltaNet 到 KDA：遗忘需要多精细

统一使用 $D_t=\operatorname{Diag}(\alpha_t)$：
$$
\bar S_t=D_tS_{t-1},\quad e_t=v_t-\bar S_t^Tk_t,
$$
$$
S_t=\bar S_t+\beta_tk_te_t^T=(I-\beta_tk_tk_t^T)D_tS_{t-1}+\beta_tk_tv_t^T.
$$

| 变体 | 门 | 可独立遗忘什么 |
|---|---|---|
| DeltaNet | $D_t=I$ | 只通过 delta 更新改写 key 方向 |
| Gated DeltaNet | $D_t=\alpha_tI$，每头一个标量 | 整头状态同时衰减 |
| KDA | $D_t=\operatorname{Diag}(\alpha_{t,1},…,\alpha_{t,d_k})$ | 每个 key 通道独立衰减 |

**乘法顺序是关键**：KDA 是 $(I-\beta kk^T)D$，一般不等于 $D(I-\beta kk^T)$。
它是 DPLR：$A=D-\beta k(k^TD)$，其中低秩的左右因子由同一 k 和 D 绑定；不是任意两个自由向量。
广播必须是 `alpha[..., :, None] * S`，衰减 key 行，不是 value 列。alpha=0 的通道可以完全清空，beta=0 仍会遗忘。



<!-- manim-group:kda-decay:start -->
<section class="matrix-steps" data-matrix-group="kda-decay" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 241px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：遗忘门按状态行缩放，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-01-desktop.png" alt="步骤 1：遗忘门按状态行缩放" width="912" height="572" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：遗忘门按状态行缩放</strong> 第一行衰减一半，第二行保留。D 为对角矩阵，非对角零值留白。 <a href="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 260.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：在遗忘后的状态上预测，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-02-desktop.png" alt="步骤 2：在遗忘后的状态上预测" width="912" height="574" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：在遗忘后的状态上预测</strong> 随后计算残差 eᵀ=vᵀ−预测，再执行前组相同的残差外积写入。 <a href="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 338px; --diagram-mobile-width: 339px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：合并公式时保持乘法次序，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-03-desktop.png" alt="步骤 3：合并公式时保持乘法次序" width="676" height="586" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：合并公式时保持乘法次序</strong> 一般 key 下低秩修正与 D 不可交换；先 D 遗忘，再用 (I−βkkᵀ) 修正。DeltaNet 取 D=I，标量遗忘各行同门，KDA 各行不同门。 <a href="/images/notes/attention-from-softmax-to-kda/manim/kda-decay-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:kda-decay:end -->


门可由输入产生：$\log\alpha=-\exp(A_{log})\operatorname{softplus}(g+dt_{bias})$，$\beta=\sigma(b)$。
实际模型还含投影、短因果卷积、SiLU、Q/K L2 norm、输出 RMSNorm 与门控。
KDA 的状态固定大小不意味着记忆无限；整个 Kimi Linear 是混合模型，仍含全局注意力层。
来源：[Gated DeltaNet](https://arxiv.org/html/2412.06464v3)、[Kimi Linear §3–4](https://arxiv.org/html/2510.26692v1)。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-07.png" alt="图 7：key-major 状态按行衰减，然后读取旧预测、计算 value 残差并执行秩一写入。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 7：key-major 状态按行衰减，然后读取旧预测、计算 value 残差并执行秩一写入。</figcaption>
</figure>

### 9.1 一个统一实现，把差异限制在门的粒度上

统一状态布局之后，三种递推可以共用同一函数：DeltaNet 传入全 1 的门，Gated DeltaNet 将每头标量扩展到 key 通道，KDA 使用各通道独立的门。这样既便于理解，也方便用退化关系检查实现是否正确。

```python
# alpha 按 key 行遗忘，beta 控制残差写入；return_tape 保存反向所需的每步中间状态。
def delta_recurrent(q,k,v,beta,alpha=None,state=None,return_tape=False):
    assert q.shape==k.shape and q.shape[:-1]==v.shape[:-1]==beta.shape
    # NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
    if alpha is None: alpha=np.ones_like(k)
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])) if state is None else state.copy()
    outs=[]; tape=[]
    for t in range(q.shape[-2]):
        qt,kt,vt,bt,at=q[...,t,:],k[...,t,:],v[...,t,:],beta[...,t],alpha[...,t,:]
        previous=s
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        decayed=at[..., :,None]*s
        # NumPy：einsum("...k,...kv->...v")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        prediction=np.einsum('...k,...kv->...v',kt,decayed)
        error=vt-prediction
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        s=decayed+(bt[...,None]*kt)[..., :,None]*error[...,None,:]
        # NumPy：einsum("...k,...kv->...v")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        outs.append(np.einsum('...k,...kv->...v',qt,s))
        if return_tape: tape.append((previous,decayed,error,s))
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    result=(np.stack(outs,-2),s)
    return result+(tape,) if return_tape else result
```

递推给出了清楚的语义，但逐 token 执行在 GPU 上往往会形成许多小操作。我们已经降低了关于序列长度的算术复杂度，接下来还要回答：怎样把这种递推组织成适合矩阵乘法硬件的计算？

## 10. Chunkwise：把序列递推改写为块内三角系统

这一节用统一推导同时覆盖 DeltaNet/GDN/KDA；块内索引 $i,j=1,…,C$，初始状态 $S_0$。
定义实际写入的伪 value $u_i=\beta_i(v_i-\bar S_i^Tk_i)$ 和逐通道衰减 $g_{i:j}=\prod_{t=i}^j\alpha_t$。
空乘积为全 1。把递推展开：

$$
S_i=\operatorname{Diag}(g_{1:i})S_0+
\sum_{j\le i}\operatorname{Diag}(g_{j+1:i})k_ju_j^T.
$$

在第 i 步**写入之前**，旧写入 j 已经经历了 $j+1,…,i$ 的衰减：
$$
u_i=\beta_i\left[v_i-S_0^T(g_{1:i}\odot k_i)
-\sum_{j<i}F_{ij}u_j\right].
$$
这里 $F_{ij}=k_i^T\operatorname{Diag}(g_{j+1:i})k_j$。令 $K_g[i]=g_{1:i}\odot k_i$、$Q_g[i]=g_{1:i}\odot q_i$，
$L_{ij}=\beta_iF_{ij}$（仅 j<i），$R=\operatorname{Diag}(\beta)(V-K_gS_0)$，则：

$$
\boxed{(I+L)U=R}
$$
$$
O=Q_gS_0+EU,\quad E_{ij}=q_i^T\operatorname{Diag}(g_{j+1:i})k_j\;(j\le i),
$$
$$
S_C=\operatorname{Diag}(g_{1:C})S_0+
(K_{end})^TU,\quad K_{end}[j]=g_{j+1:C}\odot k_j.
$$



<!-- manim-group:chunk-solve:start -->
<section class="matrix-steps" data-matrix-group="chunk-solve" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 323px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：块内依赖形成单位下三角系统，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-01-desktop.png" alt="步骤 1：块内依赖形成单位下三角系统" width="996" height="658" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：块内依赖形成单位下三角系统</strong> 图中系数矩阵含单位对角；正文 L 本身严格下三角。未知 U 逐行求出，不需要构造逆矩阵。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 469px; --diagram-mobile-width: 302.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：前代：第一行没有更早依赖，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-02-desktop.png" alt="步骤 2：前代：第一行没有更早依赖" width="938" height="558" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：前代：第一行没有更早依赖</strong> 逐个 value 通道减去已知历史写入。每行求出后供下一行使用；这是前向代入，不是并行独立逐行除法。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 469px; --diagram-mobile-width: 335px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：前代：只用已求出的第 1…1 行，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-03-desktop.png" alt="步骤 3：前代：只用已求出的第 1…1 行" width="938" height="558" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：前代：只用已求出的第 1…1 行</strong> 逐个 value 通道减去已知历史写入。每行求出后供下一行使用；这是前向代入，不是并行独立逐行除法。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 469px; --diagram-mobile-width: 335px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：前代：只用已求出的第 1…2 行，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-04-desktop.png" alt="步骤 4：前代：只用已求出的第 1…2 行" width="938" height="558" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：前代：只用已求出的第 1…2 行</strong> 逐个 value 通道减去已知历史写入。每行求出后供下一行使用；这是前向代入，不是并行独立逐行除法。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-solve-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:chunk-solve:end -->



<!-- manim-group:chunk-output:start -->
<section class="matrix-steps" data-matrix-group="chunk-output" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 241px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：输出先读取块前历史，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-01-desktop.png" alt="步骤 1：输出先读取块前历史" width="912" height="668" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：输出先读取块前历史</strong> Q_g 含从块首到当前读取时刻的门乘积。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 241px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：再读取本块实际写入，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-02-desktop.png" alt="步骤 2：再读取本块实际写入" width="996" height="656" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：再读取本块实际写入</strong> U 使用上一组前代结果；E 含对角及下三角，是读取系数而非 softmax 概率。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 300px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：同一 token、同一通道相加，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-03-desktop.png" alt="步骤 3：同一 token、同一通道相加" width="922" height="668" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：同一 token、同一通道相加</strong> 历史贡献与本块贡献必须对齐同一输出行。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 261.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：本块写入继续传到块尾，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-04-desktop.png" alt="步骤 4：本块写入继续传到块尾" width="996" height="662" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：本块写入继续传到块尾</strong> 块尾状态再加 Diag(g₁:C)S₀；K_end 已将每次写入衰减到块尾。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-output-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:chunk-output:end -->


这是一个单位下三角系统，依次前代，不需 `inv`。也可以先分别求解
$U_0=(I+L)^{-1}\operatorname{Diag}(\beta)V$ 与
$W=(I+L)^{-1}\operatorname{Diag}(\beta)K_g$，再 $U=U_0-WS_0$；这连接到 WY/UT 表达，避免把每个 token 的 `[dk,dk]` 转移矩阵连乘。



<!-- manim-group:chunk-wy:start -->
<section class="matrix-steps" data-matrix-group="chunk-wy" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 303px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：先求与输入状态无关的写入，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-01-desktop.png" alt="步骤 1：先求与输入状态无关的写入" width="996" height="662" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：先求与输入状态无关的写入</strong> 本例右端 R_V 已包含逐行 β 缩放；用同一个前代系数矩阵求解。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 321px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：同一个系统，再解另一组右端，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-02-desktop.png" alt="步骤 2：同一个系统，再解另一组右端" width="996" height="663" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：同一个系统，再解另一组右端</strong> W 是三角求解结果，不是投影层的训练权重；它描述输入状态对本块写入的影响。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 460px; --diagram-mobile-width: 302px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：扣掉输入状态贡献，得到 U，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-03-desktop.png" alt="步骤 3：扣掉输入状态贡献，得到 U" width="920" height="656" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：扣掉输入状态贡献，得到 U</strong> 线性方程的解对右端是线性的，因此可拆成两次求解。与直接求 solve(I+L,R_V−R_K S₀) 一致。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-wy-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:chunk-wy:end -->

**两种实现**：

- 结合律/GEMM 路径：$G_i=g_{1:i}$，$F=(G\odot K)(K/G)^T$，$E=(G\odot Q)(K/G)^T$ 后加三角 mask。形成块内 Gram 矩阵可用矩阵乘法。
- 稳定教学路径：直接计算区间衰减 $g_{j+1:i}$，无需先形成极小的 G 再求倒数，支持精确零门。它保留 Python 内层循环，正确性好但速度不代表生产 kernel。



<!-- manim-group:chunk-gram:start -->
<section class="matrix-steps" data-matrix-group="chunk-gram" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 219px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：累计门逐格乘 key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-01-desktop.png" alt="步骤 1：累计门逐格乘 key" width="922" height="649" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：累计门逐格乘 key</strong> G 的每行是从块首到该位置的逐通道门乘积；乘法不收缩轴。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 320px; --diagram-mobile-width: 260.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：另一侧逐格除以累计门，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-02-desktop.png" alt="步骤 2：另一侧逐格除以累计门" width="640" height="710" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：另一侧逐格除以累计门</strong> 分母与分子逐格对应，并非矩阵逆。此展开仅适用于 G 非零且除法数值安全。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 540px; --diagram-mobile-width: 323.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：乘转置后形成时间对的门比值，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-03-desktop.png" alt="步骤 3：乘转置后形成时间对的门比值" width="1080" height="650" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：乘转置后形成时间对的门比值</strong> 第 i,j 格包含 G_i/G_j，表示从写入 j 到读取 i 的区间衰减。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 366px; --diagram-mobile-width: 298.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：按用途保留正确三角部分，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-04-desktop.png" alt="步骤 4：按用途保留正确三角部分" width="732" height="663" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：按用途保留正确三角部分</strong> 块内写入依赖严格排除对角，读取允许当前 token；两种 mask 不能互换。 <a href="/images/notes/attention-from-softmax-to-kda/manim/chunk-gram-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:chunk-gram:end -->


GEMM 路径只适用于非零且不过小的块内累计门；实作通常在 log 域算区间差、二级分块，并混用 FP32。不能用随意 clamp 分母冒充数学等价。
分块顺序仍有状态依赖；块内矩阵操作提升 GPU 吞吐，不表示所有时间步都没有依赖。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-08.png" alt="图 8：一个 chunk 的累计衰减 G、严格下三角更新系数 L、因果读取系数 E 与传到块尾的 key。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 8：一个 chunk 的累计衰减 G、严格下三角更新系数 L、因果读取系数 E 与传到块尾的 key。</figcaption>
</figure>

```python
# 按行前代求解 (I+l)out=r；第 i 行只依赖前 i 行，i=0 时空和为零。
def unit_lower_solve(l,r):
    # 求 (I + 严格下三角 l) x = r；批量前代 O(C^2 * dv)。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    out=np.zeros_like(r)
    for i in range(r.shape[-2]):
        # NumPy：einsum("...j,...jv->...v")：对 j 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        out[...,i,:]=r[...,i,:]-np.einsum('...j,...jv->...v',l[...,i,:i],out[...,:i,:])
    return out
```

### 10.1 结合律允许并行，但并行本身不保证便宜

令 $A_t=(I-\beta k k^T)D_t$、$B_t=\beta kv^T$，则 $S_t=A_tS_{t-1}+B_t$。
两个变换按时间复合：
$$
(A_2,B_2)\circ(A_1,B_1)=(A_2A_1,A_2B_1+B_2).
$$


<!-- manim-group:affine-scan:start -->
<section class="matrix-steps" data-matrix-group="affine-scan" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 284.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：先 1 后 2，转移矩阵左乘，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-01-desktop.png" alt="步骤 1：先 1 后 2，转移矩阵左乘" width="912" height="572" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：先 1 后 2，转移矩阵左乘</strong> 将 S₁=A₁S₀+B₁ 代入 S₂=A₂S₁+B₂，得到 A₂A₁；不能交换顺序。 <a href="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 302.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：第一步写入经过第二步转移，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-02-desktop.png" alt="步骤 2：第一步写入经过第二步转移" width="912" height="583" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：第一步写入经过第二步转移</strong> 早先写入也会被后续转移作用；只把 B₁+B₂ 相加是错误的。 <a href="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 241px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：再加上第二步新写入，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-03-desktop.png" alt="步骤 3：再加上第二步新写入" width="922" height="572" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：再加上第二步新写入</strong> 最终把两步表示成 S₂=A₂₁S₀+B₂₁。此复合可结合，但通常不可交换。 <a href="/images/notes/attention-from-softmax-to-kda/manim/affine-scan-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:affine-scan:end -->


复合满足结合律，可作 $O(\log N)$ 深度的并行 prefix scan；但显式 A 是 `[dk,dk]`，矩阵连乘每次 $O(d_k^3)$。
附录的 doubling scan 是 correctness oracle，工作量 $O(N\log N(d_k^3+d_k^2d_v))$、状态存储 $O(N(d_k^2+d_kd_v))$。
它解释为何必须利用对角加低秩、WY/UT 结构，而不能只说“用了 scan 就快”。

这一点与 FlashAttention 恰好相互呼应：算法的渐近复杂度很重要，却不足以决定真实速度。计算是否成为足够大的矩阵乘法、需要保存多少中间状态、是否增加昂贵的三角求解或同步，都会改变实现的表现。

## 11. 从一个 KDA 算子，到可运行的 token-mixing 层

把孤立 Q/K/V 算子接到输入 X 上：

| 步骤 | tensor 流 | 操作 |
|---|---|---|
| 输入投影 | `[B,N,D] → [B,N,Hdk]` 或 `[B,N,Hdv]` | Wq/Wk/Wv |
| 短卷积 | 形状不变 | 每通道独立，左侧 padding，不能读未来 |
| 激活分头 | `[B,N,Hd] → [B,H,N,d]` | SiLU；q/k 再 L2 norm |
| 遗忘门 | `[B,N,D] → [B,N,r_gate] → [B,H,N,dk]` | 低秩投影与 log-decay |
| 写入门 | `[B,N,D] → [B,H,N]` | sigmoid |
| KDA | Q/K/V/alpha/beta → `[B,H,N,dv]` | 递推或分块 |
| 输出 | `[B,H,N,dv] → [B,N,Hdv] → [B,N,D]` | head RMSNorm、sigmoid gate、Wo |



<!-- manim-group:layer-gates:start -->
<section class="matrix-steps" data-matrix-group="layer-gates" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 372px; --diagram-mobile-width: 239.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：门投影先经过窄通道，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-01-desktop.png" alt="步骤 1：门投影先经过窄通道" width="744" height="569" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：门投影先经过窄通道</strong> 门的低秩宽度与 Linear Attention 的特征维不是同一个参数。 <a href="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 414px; --diagram-mobile-width: 266px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：再展开到每个 key 通道，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-02-desktop.png" alt="步骤 2：再展开到每个 key 通道" width="828" height="575" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：再展开到每个 key 通道</strong> raw 值还需正文规定的门参数化，再变成有效衰减系数；不能直接当作概率。 <a href="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 361.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：输出 gate 与 value 通道逐格相乘，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-03-desktop.png" alt="步骤 3：输出 gate 与 value 通道逐格相乘" width="922" height="580" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：输出 gate 与 value 通道逐格相乘</strong> 每条 token 行沿 value 通道归一化后门控；本例归一化值用 ε=0 示意。再合头并乘 W_O。 <a href="/images/notes/attention-from-softmax-to-kda/manim/layer-gates-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:layer-gates:end -->


这是可执行的紧凑参数化示例，维度可配置，非 Kimi 权重加载器。未实现 MoE、MLA、分布式训练。
完整模型流式部署时，除 S 外还须保留 Q/K/V 的短卷积最近 w−1 项缓存；只保留 S 会导致跨段边界输出错误。附录中的 core streaming 测试只覆盖 attention 状态，层级测试覆盖完整序列因果性。

层级结构还有一个实践意义：比较公式的测试应该固定 q/k/v；比较端到端层的测试则必须把卷积、归一化和门的计算一起纳入。前者发现算子错误，后者才能发现跨段卷积缓存丢失、投影维度错误等集成问题。

## 12. 反向传播：前向节省的内存，训练时能否保住

### 12.1 Softmax：无需构造完整 Jacobian

设上游梯度 $G=\partial L/\partial O$：

$$
dV=P^TG,\quad dP=GV^T,
$$
$$
dA=P\odot\left(dP-\operatorname{rowsum}(P\odot dP)\right),
$$
$$
dQ=dAK/\sqrt{d_k},\qquad dK=dA^TQ/\sqrt{d_k}.
$$

逐行 softmax Jacobian 是 $\operatorname{Diag}(p)-pp^T$；向量化表达式避免生成 `[B,H,N,M,M]`。


<!-- manim-group:softmax-jacobian:start -->
<section class="matrix-steps" data-matrix-group="softmax-jacobian" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 414px; --diagram-mobile-width: 285px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：概率外积保留两个 key 轴，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-01-desktop.png" alt="步骤 1：概率外积保留两个 key 轴" width="828" height="594" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：概率外积保留两个 key 轴</strong> 一个 query 的概率向量有两个元素；外积得到两两依赖，不是概率转移矩阵。 <a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 460px; --diagram-mobile-width: 244px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：对角项减去交叉项，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-02-desktop.png" alt="步骤 2：对角项减去交叉项" width="920" height="594" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：对角项减去交叉项</strong> 提高一个 logit 会增加它自己的概率，也会降低其他项的概率，因此非对角导数为负。 <a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 372px; --diagram-mobile-width: 208px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：梯度乘 Jacobian，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-03-desktop.png" alt="步骤 3：梯度乘 Jacobian" width="744" height="560" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：梯度乘 Jacobian</strong> 实际实现用下一组逐元素与归约公式，避免为每条 query 保存 M×M Jacobian。 <a href="/images/notes/attention-from-softmax-to-kda/manim/softmax-jacobian-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:softmax-jacobian:end -->

mask 的梯度为 0，因为对应 P 为 0。投影层进一步有 $dW_Q=\sum_b X_b^T(dQ_{merged})_b$，$dX_Q=dQ_{merged}W_Q^T$，K/V 分支对 dX 求和。


<!-- manim-group:backward-softmax:start -->
<section class="matrix-steps" data-matrix-group="backward-softmax" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 302px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：输出梯度沿概率转置传给 V，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-01-desktop.png" alt="步骤 1：输出梯度沿概率转置传给 V" width="912" height="575" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：输出梯度沿概率转置传给 V</strong> 共享 value 的梯度累加全部 query 的贡献；转置把 query 轴放到被收缩的位置。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 264.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：沿 value 转置传给概率，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-02-desktop.png" alt="步骤 2：沿 value 转置传给概率" width="912" height="575" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：沿 value 转置传给概率</strong> dP 的形状与 P 相同；接着经过 softmax 的局部反向得到 dA。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 261.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：logits 梯度传回 query，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-03-desktop.png" alt="步骤 3：logits 梯度传回 query" width="912" height="657" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：logits 梯度传回 query</strong> 图中乘积尚未除以 √d_k；公式明确最终还要缩放。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 348px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：key 累加全部 query 的分数梯度，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-04-desktop.png" alt="步骤 4：key 累加全部 query 的分数梯度" width="912" height="662" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：key 累加全部 query 的分数梯度</strong> 固定 mask 不可见位置的 dA 为零；某条 query 全遮蔽，不代表共享 key 从其他 query 收不到梯度。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-softmax-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:backward-softmax:end -->



<!-- manim-group:backward-reduce:start -->
<section class="matrix-steps" data-matrix-group="backward-reduce" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 282.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：概率与上游梯度逐格相乘，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-01-desktop.png" alt="步骤 1：概率与上游梯度逐格相乘" width="922" height="560" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：概率与上游梯度逐格相乘</strong> 这一乘法不收缩维度；每个位置保留对应概率与梯度的乘积。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 285px; --diagram-mobile-width: 285.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：沿 key 轴求和，保留单列，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-02-desktop.png" alt="步骤 2：沿 key 轴求和，保留单列" width="570" height="632" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：沿 key 轴求和，保留单列</strong> 每条 query 有自己的标量 D_i；keepdims=True 保留 N×1，供下一步广播。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 418px; --diagram-mobile-width: 241px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：每行减去自己的标量，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-03-desktop.png" alt="步骤 3：每行减去自己的标量" width="836" height="572" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：每行减去自己的标量</strong> 单列 D 沿列广播；第 i 行所有 key 都减同一个 D_i。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 181px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：逐格乘回概率，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-04-desktop.png" alt="步骤 4：逐格乘回概率" width="922" height="562" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：逐格乘回概率</strong> 本例两行 dA 分别为 [0,0] 与 [−1/4,1/4]，与显式 Jacobian 相乘一致。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-reduce-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:backward-reduce:end -->



<!-- manim-group:backward-projection:start -->
<section class="matrix-steps" data-matrix-group="backward-projection" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 498px; --diagram-mobile-width: 288.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：沿 token 轴累计权重梯度，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-01-desktop.png" alt="步骤 1：沿 token 轴累计权重梯度" width="996" height="686" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：沿 token 轴累计权重梯度</strong> 先合并 head 梯度，恢复投影层输出布局；权重被所有 token 共享。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 461px; --diagram-mobile-width: 287.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：权重也被不同 batch 共享，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-02-desktop.png" alt="步骤 2：权重也被不同 batch 共享" width="922" height="663" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：权重也被不同 batch 共享</strong> 不能保留独立 batch 轴当作最终权重梯度；必须对 batch 求和。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 456px; --diagram-mobile-width: 282px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：权重转置把梯度传回输入，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-03-desktop.png" alt="步骤 3：权重转置把梯度传回输入" width="912" height="680" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：权重转置把梯度传回输入</strong> Q、K、V 三条分支都回到同一 X，输入梯度还需把三条分支贡献相加。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-projection-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:backward-projection:end -->


下文有限差分直接检查标量损失 $L=\langle O,G\rangle$；这比只检查两个前向代码的相等更能发现共同错误。

### 12.2 Flash：保存 LSE，反向重建局部概率

前向保存 $O$ 与 $LSE_i=m_i+\log l_i$。反向每个 tile 重算 score，$P_{ij}=\exp(A_{ij}-LSE_i)$。
利用 $D_i=\sum_bG_{ib}O_{ib}=\sum_jP_{ij}dP_{ij}$，直接得到
$dA_{ij}=P_{ij}(dP_{ij}-D_i)$；然后按 tile 累加 dQ/dK/dV。
附录给出确定性单线程实现；真实 GPU 的 dK/dV 跨 query tile 累加要设计 ownership、归约或原子操作，不能让线程无同步写同一位置。

### 12.3 Linear：从未来累积状态梯度

非因果可以一次计算 S,z；因果反向必须从后往前累加状态梯度，因为较早的写入影响所有后续 query。
设 $n_t=\phi(q_t)^TS_t,\ h_t=\phi(q_t)^Tz_t+\epsilon$，上游为 g：
$$
dn_t=g_t/h_t,\quad dh_t=-\langle g_t,n_t\rangle/h_t^2,
$$
$$
d\phi(q_t)=S_tdn_t+dh_tz_t,
$$
$$
dS_t\mathrel{+}=\phi(q_t)dn_t^T,\quad dz_t\mathrel{+}=dh_t\phi(q_t),
$$
$$
d\phi(k_t)=dS_tv_t+dz_t,\quad dv_t=dS_t^T\phi(k_t).
$$


<!-- manim-group:backward-linear:start -->
<section class="matrix-steps" data-matrix-group="backward-linear" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 236px; --diagram-mobile-width: 223px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：先对分式输出求导，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-01-desktop.png" alt="步骤 1：先对分式输出求导" width="472" height="736" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：先对分式输出求导</strong> 本例 l=2,g=[1,0]ᵀ，故 dn=[1/2,0]ᵀ、dl=−1/2。分母路径不能遗漏。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 414px; --diagram-mobile-width: 323px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：分子读取向状态累积外积梯度，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-02-desktop.png" alt="步骤 2：分子读取向状态累积外积梯度" width="828" height="578" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：分子读取向状态累积外积梯度</strong> 归一化状态同时累积 dz += dl·q̄；query 梯度为 S dn + dl·z。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 372.5px; --diagram-mobile-width: 321.5px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：写入反向把状态梯度传给 key，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-03-desktop.png" alt="步骤 3：写入反向把状态梯度传给 key" width="745" height="571" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：写入反向把状态梯度传给 key</strong> 图中展示第一项，最终还要加 dz。dS、dz 从未来向过去累计，必须使用对应时刻的前缀状态。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 372px; --diagram-mobile-width: 278px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：转置状态梯度传给 value，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-04-desktop.png" alt="步骤 4：转置状态梯度传给 value" width="744" height="576" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：转置状态梯度传给 value</strong> 最后对 q̄、k̄ 的梯度还需乘特征映射 φ 的逐元素导数。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-linear-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:backward-linear:end -->


由于加法转移，dS,dz 原样向过去传播。最后乘 ELU+1 的导数；正半轴导数 1，负半轴 exp(x)。参考版保存每步 S,z 以求梯度；其激活内存同样不是常量。

### 12.4 Delta 系列：沿着衰减、预测和写入逐步求导

记上游输出梯度为 $g_t$，从未来传回状态梯度 $H$。
从后往前先加入读出梯度：$d q_t=S_tg_t$，$H\leftarrow H+q_tg_t^T$。
对 $S_t=\bar S_t+\beta_tk_te_t^T$：
$$
d\beta_t=\langle H,k_te_t^T\rangle,\quad
 d k_t^{write}=\beta_tHe_t,\quad d e_t=\beta_tH^Tk_t.
$$
对 $e_t=v_t-\bar S_t^Tk_t$：
$$
dv_t=de_t,\quad dk_t=dk_t^{write}-\bar S_tde_t,\quad
 d\bar S_t=H-k_tde_t^T.
$$
最后：$d\alpha_t=\operatorname{rowsum}(d\bar S_t\odot S_{t-1})$，
$H_{prev}=D_td\bar S_t$。GDN 标量门梯度再对 key 维求和。



<!-- manim-group:backward-delta:start -->
<section class="matrix-steps" data-matrix-group="backward-delta" aria-label="分步矩阵图解">
<figure class="matrix-step" style="--diagram-width: 414px; --diagram-mobile-width: 282px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-01-desktop.png" target="_blank" rel="noopener" aria-label="步骤 1：输出读取向状态回传外积，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-01-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-01-desktop.png" alt="步骤 1：输出读取向状态回传外积" width="828" height="575" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 1：输出读取向状态回传外积</strong> 先累积该时刻输出对状态的贡献；H 还包含未来时间回传的状态梯度。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-01-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 394px; --diagram-mobile-width: 289px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-02-desktop.png" target="_blank" rel="noopener" aria-label="步骤 2：残差写入对 key 的梯度，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-02-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-02-desktop.png" alt="步骤 2：残差写入对 key 的梯度" width="788" height="575" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 2：残差写入对 key 的梯度</strong> 这只是写入路径；key 同时参与预测，随后还需减去 S̄ de。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-02-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 372px; --diagram-mobile-width: 262px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-03-desktop.png" target="_blank" rel="noopener" aria-label="步骤 3：残差接收转置状态梯度，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-03-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-03-desktop.png" alt="步骤 3：残差接收转置状态梯度" width="744" height="583" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 3：残差接收转置状态梯度</strong> 目标 value 的梯度加 de；预测值的梯度为 −de。写入率梯度 dβ=Σ_ab H_ab k_a e_b。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-03-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
<figure class="matrix-step" style="--diagram-width: 460px; --diagram-mobile-width: 282px">
<a href="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-04-desktop.png" target="_blank" rel="noopener" aria-label="步骤 4：扣回预测路径的状态梯度，打开高清图">
<picture>
<source media="(max-width: 768px)" srcset="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-04-mobile.png" />
<img src="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-04-desktop.png" alt="步骤 4：扣回预测路径的状态梯度" width="920" height="575" loading="lazy" decoding="async" />
</picture></a>
<figcaption><strong>步骤 4：扣回预测路径的状态梯度</strong> 若包含 KDA 遗忘，继续 H_prev=D dS̄，并逐行求 dα=rowsum(dS̄⊙S_prev)。 <a href="/images/notes/attention-from-softmax-to-kda/manim/backward-delta-04-desktop.png" target="_blank" rel="noopener">打开高清图</a></figcaption>
</figure>
</section>
<!-- manim-group:backward-delta:end -->


本函数返回的是已归一化 q/k、直接 alpha/beta 的梯度。本文 KDA 采用主定义的 query scale=1；与官方 kernel 比较时须显式统一 scale，某些实现另乘 $d_k^{-1/2}$。
L2 norm 输入 x 的梯度为 $(g-y(y^Tg))/\|x\|$（非零 x、未触发 epsilon 分支）；sigmoid 的梯度为 $g\beta(1-\beta)$；门的神经参数还需链式法则。
参考反向保存每步状态，空间 $O(Nd_kd_v)$；生产实现通过 chunk checkpoint 与重计算降低保存量，**不能把推理常量状态空间当成训练激活空间**。

## 13. 在同一张表里比较：时间、状态、激活与 IO

下表按**单 batch、单 head、单层**计，乘上 BH 得到总量；自注意力 N=M；忽略输入/输出张量与模型权重，另注明之。FLOPs 约定一次乘加计 2 FLOPs。

| 算法/实现 | 长度 N 的混合计算时间 | decode 每 token | 推理持久状态 | 本文实现主要额外工作空间 |
|---|---|---|---|---|
| Dense softmax | $\Theta(N^2(d_k+d_v))$ | $\Theta(N(d_k+d_v))$ | $\Theta(N(d_k+d_v))$ KV | $\Theta(N^2)$ logits/P |
| FA1/2/3 数学 | 同 dense，仍二次 | 同 dense，split-K 可降低延迟 | 同 softmax KV | tile $O(RT+Rd_v+(R+T)d_k+Td_v)$；O/LSE 随 N |
| Linear 递推 | $\Theta(Nrd_v)$ | $\Theta(rd_v)$ | $rd_v+r$ | $O(rd_v+r)$，不计输出 |
| Linear chunk | $O(Nrd_v+NC(r+d_v))$ | 同递推 | 同递推 | $O(C^2+C(r+d_v)+rd_v)$ |
| Delta/GDN/KDA 递推 | $\Theta(Nd_kd_v)$ | $\Theta(d_kd_v)$ | $d_kd_v$ | $O(d_kd_v)$，不保存 tape |
| Delta/GDN/KDA chunk | $O(Nd_kd_v+NC(d_k+d_v))$ | 通常用递推 | $d_kd_v$ | $O(C^2+C(d_k+d_v)+d_kd_v)$ |
| 显式仿射 doubling scan | $O(N\log N(d_k^3+d_k^2d_v))$ | 不适合常规 decode | 构造全部 A/B | $O(N(d_k^2+d_kd_v))$ |
| PagedAttention | attention 数学不变 | $\Theta(N(d_k+d_v))$ 加页寻址 | 按需页池+页表 | 本实现每页 online 统计量；无需全长 scores |
| RadixAttention | 取决于 prefix 命中 | 新 query 仍读全部可见 KV | 所有缓存唯一前缀 KV+树元数据 | 查找匹配路径；此教学版返回 KV 时复制 |

**逐项核算**：

- Softmax 的两个 GEMM 约 $2N^2d_k+2N^2d_v$；causal 可跳过上三角使有效 pair 数为 $N(N+1)/2$。当前 NumPy dense 先完整 GEMM 再 mask，实际不会省掉上三角乘法。
- Linear 每 token 写外积和读 S 各约 $2rd_v$；归一化还需 O(r)。特征映射自身若昂贵，要另加成本。
- Delta 每 token 预测、外积更新、query 读出各为 $\Theta(d_kd_v)$；KDA 门乘状态多加同阶操作。不能仅写 O(N) 而隐藏 d 的平方。
- chunk 每块 Gram/read 矩阵 O($C^2d_k$)，三角前代与 EU 为 O($C^2d_v$)，与输入状态交互 O($Cd_kd_v$)；块数 N/C。这里的复杂度对应**前代**，若用通用 dense inverse 会额外引入 $O(C^3)$。
- 投影并未消失：标准 D→D 多头 QKV/O 投影合计 $O(BND^2)$；门投影、卷积、归一化也要计入模型层时间。
- 标准 GQA：Hq query 头、Hkv KV 头，混合计算依 Hq，KV cache 为 $BNH_{kv}(d_k+d_v)$；复制 KV 的教学 gqa 函数不能体现这一空间节约。
- KV bytes = $n_{layers}BNH_{kv}(d_k+d_v)\times bytes\_per\_element$。KDA state bytes = $n_{KDA}BHd_kd_v\times state\_bytes$；状态常用 FP32、KV 常用 FP16/BF16，不能只比较元素数。
- 固定状态仅是推理核心空间。本文 delta backward 的 tape 为 $O(BHNd_kd_v)$；chunk checkpoint 可只保留块边界状态 $O(BH(N/C)d_kd_v)$，外加输入/门/输出等线性项与块内临时量。
- 本文 NumPy Flash 用普通 ndarray，不能证明真实 HBM IO 上界；其 `outputs` 列表及拼接还会短暂重复存输出。这不影响没有 N×N 概率缓存的性质。
- radix 理想最长匹配比较 O(L)，分裂也涉及 token/索引处理。本教学版 tuple 切片、数组复制和 `nodes()` 遍历增加开销；逐个叶子淘汰的最坏复杂度可达 O(V²)，生产实现会维护专门的淘汰结构。

### 13.1 模型大小示例与 decode 交叉点

单层、单请求、H=8、dk=dv=128：FP16 KV 每 token 占 $8\cdot256\cdot2=4096$ bytes；FP32 KDA 状态占 $8\cdot128^2\cdot4=524288$ bytes。
两者在 N=128 元素存储量级交叉，**不是实测速度交叉点**；GQA、batch、混合层、卷积状态会改变模型总量。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-09.png" alt="图 9：单层单请求下的理论存储量；KV、KDA 状态与 dense score 使用不同指定精度，未计输入、权重和分配器。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 9：单层单请求下的理论存储量；KV、KDA 状态与 dense score 使用不同指定精度，未计输入、权重和分配器。</figcaption>
</figure>

## 14. 怎样验证这些实现，而不被一个数字误导

测量流程：固定种子与 dtype，预热，然后多次执行取中位数。CPU NumPy 的 BLAS、线程数、Python 循环开销与 GPU kernel 完全不同。
这里仅验证参考实现能随 N 运行，并记录此机器的耗时，不把结果写成模型品质比较或 FA1/2/3 的速度排名。
GPU 严格基准应使用 CUDA event、同步、预热、固定 dtype/shape/mask、报告 GPU/软件版本与 peak allocated/reserved，区分 prefill/decode/forward/backward，记录数值误差；吞吐与 TTFT/TPOT 是不同指标。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-10.png" alt="图 10：2026-09-07 初版环境的 NumPy 参考实现 CPU 中位耗时，只用于验证和记录，不代表生产 GPU 内核的速度排名。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 10：2026-09-07 初版环境的 NumPy 参考实现 CPU 中位耗时，只用于验证和记录，不代表生产 GPU 内核的速度排名。</figcaption>
</figure>

### 14.1 哪些错误需要独立的测试

覆盖非整块长度、非方形维度、B/H 广播、prefill/decode、因果性、初始状态、门退化、数值稳定、前向/反向、页共享与树分裂。
未来扰动测试只修改未来的 K/V（或输入 X），要求早期输出不变。不同公式之间不要求输出相同，只有同公式的实现方式才做等价测试。
空序列不是核心函数支持的输入（递推 stack/max 无定义），请求侧应直接跳过空段；本文输入均为正长度。

### 14.2 本文代码的验证结果

| 检查项 | 结果与含义 |
|---|---|
| 129 项原有 CPU 检查 | 全部通过；包括前向等价、有限差分、分块状态、因果性、缓存生命周期 |
| 18 项新增 CPU 检查 | 全部通过；含逐步 softmax、GQA、MLA 吸收与 RoPE、DSA gather 与因果性 |
| PyTorch SDPA CPU | 对照通过；与 GPU 内核测试分开计数 |
| Linear 与 softmax 输出 | 不要求一致；它们使用不同核函数 |
| DeltaNet / GDN / KDA | 同一公式的递推与分块一致；门满足特定条件时验证退化关系 |
| 官方 FlashAttention GPU 内核 | 官方 GPU 内核因无可用 CUDA 而跳过；PyTorch SDPA CPU 对照通过；不能据此声称已验证硬件性能 |

## 15. 带着问题回看这几条路线

遇到一个新的 attention 名称时，可以先追问四件事：它保留了什么数学定义，历史保存在哪里，新 token 需要读取什么，以及优化成本转移到了哪里。

FlashAttention 保留全部 pair 的计算，通过分块、重计算和硬件调度减少中间数据往返。PagedAttention 保留显式 KV，通过逻辑到物理的映射降低碎片和复制成本。RadixAttention 保留前缀的已计算结果，让新请求从可以复用的位置继续。

Linear attention 开始改变记忆的表示方式：不再逐项保存全部历史，而是维护有限的矩阵状态。DeltaNet 让写入由预测残差驱动；Gated DeltaNet 允许整头遗忘；KDA 把遗忘细化到 key 通道。随后，chunkwise 算法又把这些递推整理成硬件更擅长的矩阵运算。

理解这些方法的联系之后，优化选择才有依据：先确定负载与语义约束，再选择公式、内核和缓存策略。一个模型可以同时需要其中多种方法，而每一种方法都需要与它所声称解决的问题对应的测试。

## 附录 A：完整实现与自动测试

以下代码保留为完整参考程序，按展开顺序拼接到一个 `.py` 文件即可运行；依赖为 `numpy`、`matplotlib`。每个代码块包含必要的函数或测试，不能只运行后半部分而跳过初始化。

代码顺序沿用已验证的依赖顺序：先定义数学基准，再定义递推与分块，随后运行 Flash 和 cache 测试。正文则按理解问题的顺序组织，两者目的不同。

图解随文章一同提供。运行代码时会重新生成相应图表。

### NumPy 语法桥梁：先读懂形状，再读运算

`import numpy as np` 只是给 NumPy 取一个短名字。`ndarray` 是有固定形状的多维数表；例如 `q.shape=(2,4,9,3)` 表示 2 个 batch、4 个 head、9 个 token、每个 query 3 个坐标。数学列向量在 NumPy 中常暂存成一维 `[d]`，需要矩阵外积时再显式加入轴。

| 写法 | 一个具体例子 | 意思与常见错误 |
|---|---|---|
| `x.shape[-1]` | `(2,4,9,3)[-1] == 3` | 最后一轴长度，不是最后一行的值 |
| `x[...,t,:]` | `[B,H,N,d] → [B,H,d]` | `...` 保留前导轴，整数下标会移除 token 轴 |
| `x[...,t:t+1,:]` | `[B,H,N,d] → [B,H,1,d]` | 切片保留 token 轴；右端不包含在内 |
| `a[:,None]` | `[r] → [r,1]` | None 新建轴；与 `[1,dv]` 相乘得到外积 |
| `a @ b` | `[N,M] @ [M,dv] → [N,dv]` | 矩阵收缩；`a*b` 是逐元素乘与广播 |
| `sum(axis=-1,keepdims=True)` | `[N,M] → [N,1]` | 每行独立求和，保留 1 才能安全按行除法 |
| `swapaxes(-1,-2)` | `[B,H,M,dk] → [B,H,dk,M]` | 仅转置矩阵面；四维数组 `.T` 会反转所有轴 |
| `stack(out,axis=-2)` | N 份 `[B,H,dv] → [B,H,N,dv]` | 新建 token 轴；concat 是接长已有轴 |
| `einsum('...r,...rv->...v',q,S)` | `[B,H,r]` 与 `[B,H,r,dv]` | 共同 r 被乘后求和，v 和前导轴保留 |
| `np.where(keep,a,-np.inf)` | 同形布尔 mask 与分数 | True 保留分数，False 禁止读取；不是惰性分支 |
| `np.divide(...,out=zero,where=den>0)` | 空行分母为 0 | 只在合法处除；其他格子保留预初始化的零 |

**怎样判断广播是否正确？** 从右往左对齐形状，每一对维度要么相等，要么有一边为 1。例如 `[N,M] / [N,1]` 合法且按行归一化；`[N,M] / [N]` 通常报错，N=M 时更危险：它可能合法却误沿列缩放。

下面每个重要函数先有计算目的说明，具体 NumPy 运算前有中文注释。建议先运行 Softmax 小节，再读 Linear；高级反向和缓存管理可以最后展开。下载 [完整注释参考程序](/code/attention-reference.py) 可避免手动拼接；新增的逐步 softmax、GQA、MLA、DSA 则在 [独立教学程序](/code/attention-variants.py) 与附录 C 中。

```bash
python -m pip install numpy matplotlib
python attention-reference.py
python attention-variants.py
```

<!-- implementation:start -->

<details>
<summary>运行环境、公共函数与检查工具</summary>

```python
import sys, math, time, importlib.util
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from contextlib import contextmanager
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(20260907)
TESTS = []
def check(name, actual, expected, atol=2e-10, rtol=2e-10):
    # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    TESTS.append(name)
def require(name, condition):
    assert bool(condition), name
    TESTS.append(name)
def normalize(x):
    # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
    # NumPy：linalg.norm 沿给定轴求欧氏长度 sqrt(sum(x*x))，不是对矩阵每格取绝对值。
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
def sigmoid(x):
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：logaddexp(a,b) 稳定计算 log(exp(a)+exp(b))，避免直接指数化大正数。
    return np.exp(-np.logaddexp(0., -x))
def softplus(x):
    # NumPy：logaddexp(a,b) 稳定计算 log(exp(a)+exp(b))，避免直接指数化大正数。
    return np.logaddexp(0., x)
def heatmaps(items, title='', cmap='viridis'):
    fig, axes = plt.subplots(1, len(items), figsize=(3.1*len(items), 3.0), squeeze=False)
    for ax, (label, a) in zip(axes[0], items):
        # NumPy：asarray 将输入视为 ndarray；若输入已是合适数组，可复用底层存储。
        a = np.asarray(a)
        ax.imshow(a, cmap=cmap, aspect='auto')
        ax.set_title(label + '  ' + str(a.shape), fontsize=10)
        if a.size <= 64:
            # NumPy：ndenumerate 同时返回坐标元组与该格数值，用来为热力图加文字。
            for (i,j), val in np.ndenumerate(a):
                ax.text(j, i, f'{val:.2g}', ha='center', va='center', fontsize=8,
                        color='white', bbox=dict(facecolor='black', alpha=.2, edgecolor='none'))
        ax.set_xlabel('column'); ax.set_ylabel('row')
    fig.suptitle(title); fig.tight_layout(); plt.show()
print('Python', sys.version.split()[0], 'NumPy', np.__version__)
```

</details>

<details>
<summary>1. Softmax attention：从逐元素到 tensor</summary>

```python
# 输入 a 的最后一轴为 key；固定全遮蔽行返回全零。合法分数应为有限值，mask 用 -inf。
def softmax_masked(a, keep=None):
    if keep is not None:
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        a = np.where(keep, a, -np.inf)
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    m = np.max(a, axis=-1, keepdims=True)
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
    safe_m = np.where(np.isfinite(m), m, 0.)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    e = np.exp(a - safe_m)
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    den = e.sum(-1, keepdims=True)
    # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    return np.divide(e, den, out=np.zeros_like(e), where=den > 0)

# 把 query/key 的局部下标加上全局起点，返回 [n,m] 布尔可见性矩阵。
def position_mask(n, m, q_start=0, k_start=0):
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    return (np.arange(m) + k_start)[None,:] <= (np.arange(n) + q_start)[:,None]

# q/k/v 前导轴是 batch/head；输出 o[...,N,dv] 与概率 p[...,N,M]。
def softmax_attention(q, k, v, causal=False, keep=None, q_start=0):
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
    assert q.shape[-1] == k.shape[-1] and k.shape[-2] == v.shape[-2]
    if causal:
        cm = position_mask(q.shape[-2], k.shape[-2], q_start)
        keep = cm if keep is None else (keep & cm)
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    p = softmax_masked((q @ k.swapaxes(-1,-2)) / math.sqrt(q.shape[-1]), keep)
    return p @ v, p

# 先在通道内部切出 head，再交换 N/H：BN(Hd) → BN H d → BH N d。
def split_heads(x, h):
    b,n,w = x.shape
    assert w % h == 0
    # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
    # NumPy：reshape 只重组轴形状，元素总数必须相等；不会自动完成 token/head 的交换。
    return x.reshape(b,n,h,w//h).transpose(0,2,1,3)

# 先把 token 轴移回 head 前，再合并 head×通道；它与 split_heads 互为逆操作。
def merge_heads(x):
    b,h,n,d = x.shape
    # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
    # NumPy：reshape 只重组轴形状，元素总数必须相等；不会自动完成 token/head 的交换。
    return x.transpose(0,2,1,3).reshape(b,n,h*d)

# 每组 query heads 共享一套 K/V；本函数以 repeat 复制作数学参考，不代表内存优化。
def gqa(q, k, v, **kwargs):
    # 教学版复制；生产 kernel 根据 query head 寻址共享的 KV head。
    assert q.shape[1] % k.shape[1] == 0
    group = q.shape[1] // k.shape[1]
    # NumPy：repeat 沿指定轴重复元素/切片，确实会复制；这里只用于验证共享 KV 的基准。
    return softmax_attention(q, np.repeat(k,group,axis=1),
                            # NumPy：repeat 沿指定轴重复元素/切片，确实会复制；这里只用于验证共享 KV 的基准。
                            np.repeat(v,group,axis=1), **kwargs)[0]

B,H,N,DK,DV = 2,2,9,4,3
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
q = rng.normal(size=(B,H,N,DK)); k = rng.normal(size=q.shape)
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
v = rng.normal(size=(B,H,N,DV))
o,p = softmax_attention(q,k,v,causal=True)
heatmaps([('Q',q[0,0,:4]), ('K transpose',k[0,0,:4].T),
          ('P: causal rows',p[0,0,:4,:4]), ('V',v[0,0,:4]), ('O',o[0,0,:4])],
         'Q @ K.T -> row softmax -> P @ V')
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
# NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
check('softmax rows sum 1',p.sum(-1),np.ones((B,H,N)))
# NumPy：all 检查布尔数组是否全部为真，返回真/假而非数值误差大小。
# NumPy：triu 提取上三角；k=1 只保留严格未来位置，常用于检查因果输出是否为零。
require('causal upper triangle zero', np.all(np.triu(p,1)==0))
# NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
# NumPy：reshape 只重组轴形状，元素总数必须相等；不会自动完成 token/head 的交换。
check('head split merge', merge_heads(split_heads(np.arange(120).reshape(2,5,12),3)),
      # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
      # NumPy：reshape 只重组轴形状，元素总数必须相等；不会自动完成 token/head 的交换。
      np.arange(120).reshape(2,5,12))
# NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
zero,_ = softmax_attention(q,k,v,keep=np.zeros((N,N),dtype=bool))
# NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
check('all masked rows zero',zero,np.zeros_like(zero))
# Prefill 与逐 token decode 必须完全一致。
# NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
decoded = np.concatenate([softmax_attention(q[:,:,t:t+1],k[:,:,:t+1],v[:,:,:t+1])[0]
                          for t in range(N)],axis=2)
check('softmax prefill equals decode',decoded,o)
check('offset mask decode',softmax_attention(q[:,:,-1:],k,v,causal=True,q_start=N-1)[0],o[:,:,-1:])
```

</details>

<details>
<summary>1.1 反向传播：softmax Jacobian 不必显式构造</summary>

```python
# 先经过 O=PV，再经过逐行 softmax，最后经过 scaled QK，按链式法则返回梯度。
def softmax_backward(q,k,v,p,go):
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    dv = p.swapaxes(-1,-2) @ go
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    dp = go @ v.swapaxes(-1,-2)
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    da = p * (dp - (p*dp).sum(-1,keepdims=True))
    scale = q.shape[-1] ** -0.5
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    return (da @ k)*scale, (da.swapaxes(-1,-2) @ q)*scale, dv

# 固定其他元素，仅对一个坐标做正负 eps 扰动，用中心差分检验手写梯度。
def finite_difference(fn, x, eps=1e-6):
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    x = x.copy(); grad = np.zeros_like(x)
    # NumPy：ndindex 遍历所有坐标元组；有限差分一次只扰动一个输入元素。
    for idx in np.ndindex(x.shape):
        old=x[idx]; x[idx]=old+eps; pos=fn(x)
        x[idx]=old-eps; neg=fn(x); x[idx]=old
        grad[idx]=(pos-neg)/(2*eps)
    return grad

# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
sq,sk,sv = [rng.normal(size=s) for s in [(1,1,3,2),(1,1,3,2),(1,1,3,2)]]
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
so,sp=softmax_attention(sq,sk,sv,causal=True); go=rng.normal(size=so.shape)
grads=softmax_backward(sq,sk,sv,sp,go)
for j in range(3):
    arr=[sq,sk,sv]
    def loss(z):
        args=arr.copy(); args[j]=z
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        return np.sum(softmax_attention(*args,causal=True)[0]*go)
    check('softmax finite difference '+str(j),grads[j],finite_difference(loss,arr[j]),atol=2e-8,rtol=2e-6)
```

</details>

<details>
<summary>2. Linear attention：改变核函数后才可换结合顺序</summary>

```python
# ELU+1 非负特征映射；改变核定义是算法选择，不等于保持原 softmax。
def phi(x):
    # 避免 np.where 同时计算 exp(大正数)。
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：minimum 比较对应位置取较小值；例如先将指数输入限制为不大于 0，避免溢出。
    return np.where(x>=0, x+1, np.exp(np.minimum(x,0)))

def linear_dense(q,k,v,causal=True,eps=1e-12):
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    a=phi(q) @ phi(k).swapaxes(-1,-2)
    if causal: a=a*position_mask(q.shape[-2],k.shape[-2])
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    return (a@v)/(a.sum(-1,keepdims=True)+eps)

# 每步写 k 的外积，再读 q；s[...,r,dv]、z[...,r] 可跨调用传递，返回全部 token 输出。
def linear_recurrent(q,k,v,state=None,eps=1e-12):
    q,k=phi(q),phi(k)
    shape=q.shape[:-2]; r=q.shape[-1]; dv=v.shape[-1]
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s,z=(np.zeros(shape+(r,dv)),np.zeros(shape+(r,))) if state is None else (state[0].copy(),state[1].copy())
    out=[]
    for t in range(q.shape[-2]):
        kt,vt,qt=k[...,t,:],v[...,t,:],q[...,t,:]
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        s=s+kt[..., :,None]*vt[...,None,:]; z=z+kt
        # NumPy：einsum("...r,...rv->...v")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        out.append(np.einsum('...r,...rv->...v',qt,s)/(np.sum(qt*z,-1,keepdims=True)+eps))
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    return np.stack(out,-2),(s,z)

# 历史块通过 s,z 贡献，本块通过因果核权重贡献；合并分子分母后统一归一化。
def linear_chunk(q,k,v,chunk=4,eps=1e-12):
    assert chunk>0
    q,k=phi(q),phi(k)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])); z=np.zeros(q.shape[:-2]+(q.shape[-1],))
    outs=[]
    for a in range(0,q.shape[-2],chunk):
        qc,kc,vc=q[...,a:a+chunk,:],k[...,a:a+chunk,:],v[...,a:a+chunk,:]
        # NumPy：tri(n) 得到含对角线的下三角 1/0 矩阵，表示可以读取自己及过去。
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        weights=(qc@kc.swapaxes(-1,-2))*np.tri(qc.shape[-2])
        numerator=qc@s+weights@vc
        # NumPy：einsum("...tr,...r->...t")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        denominator=np.einsum('...tr,...r->...t',qc,z)[...,None]+weights.sum(-1,keepdims=True)
        outs.append(numerator/(denominator+eps))
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        s=s+kc.swapaxes(-1,-2)@vc; z=z+kc.sum(-2)
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    return np.concatenate(outs,-2),(s,z)

lr,ls=linear_recurrent(q,k,v)
check('linear dense recurrence',lr,linear_dense(q,k,v))
for c in [1,4,16]:
    lc,lstate=linear_chunk(q,k,v,c)
    check('linear chunk '+str(c),lc,lr); check('linear chunk state '+str(c),lstate[0],ls[0])
qa,ka=phi(q[0,0,:4]),phi(k[0,0,:4]); va=v[0,0,:4]
heatmaps([('Phi K.T',ka.T),('V',va),('S = K.T @ V',ka.T@va),('Phi Q',qa),('Q @ S',qa@(ka.T@va))],
         'Noncausal numerator: associativity removes N x N')
# NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
# NumPy：abs 对每个元素取绝对值，形状不变。
print('Linear 与 softmax 的最大差异（预期非零）:',np.max(np.abs(lr-o)))
```

</details>

<details>
<summary>2.1 非因果实现与 Linear attention 的完整反向</summary>

```python
def linear_noncausal(q,k,v,eps=1e-12):
    fq,fk=phi(q),phi(k)
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    state=fk.swapaxes(-1,-2)@v; z=fk.sum(-2)
    # NumPy：einsum("...tr,...r->...t")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    return (fq@state)/(np.einsum('...tr,...r->...t',fq,z)[...,None]+eps)

# 先保存每步前缀状态，反向倒序累积 ds,dz；较早写入会影响所有后续输出。
def linear_backward(q,k,v,go,eps=1e-12):
    fq,fk=phi(q),phi(k)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])); z=np.zeros(q.shape[:-2]+(q.shape[-1],))
    tape=[]
    for t in range(q.shape[-2]):
        s=s+fk[...,t,:,None]*v[...,t,None,:]; z=z+fk[...,t,:]
        tape.append((s,z))
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    ds=np.zeros_like(s); dz=np.zeros_like(z)
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    for t in reversed(range(q.shape[-2])):
        s,z=tape[t]; qt=fq[...,t,:]; kt=fk[...,t,:]; vt=v[...,t,:]
        # NumPy：einsum("...r,...rv->...v")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        num=np.einsum('...r,...rv->...v',qt,s)
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        den=(qt*z).sum(-1,keepdims=True)+eps
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        dn=go[...,t,:]/den; dh=-(go[...,t,:]*num).sum(-1,keepdims=True)/(den*den)
        # NumPy：einsum("...rv,...v->...r")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dq[...,t,:]=np.einsum('...rv,...v->...r',s,dn)+dh*z
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        ds=ds+qt[..., :,None]*dn[...,None,:]; dz=dz+dh*qt
        # NumPy：einsum("...rv,...v->...r")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dk[...,t,:]=np.einsum('...rv,...v->...r',ds,vt)+dz
        # NumPy：einsum("...rv,...r->...v")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dv[...,t,:]=np.einsum('...rv,...r->...v',ds,kt)
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：minimum 比较对应位置取较小值；例如先将指数输入限制为不大于 0，避免溢出。
    return dq*np.where(q>=0,1,np.exp(np.minimum(q,0))),dk*np.where(k>=0,1,np.exp(np.minimum(k,0))),dv
check('noncausal linear associativity',linear_noncausal(q,k,v),linear_dense(q,k,v,causal=False))
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
small=[rng.normal(size=(1,1,3,2)) for _ in range(3)]; upstream=rng.normal(size=(1,1,3,2))
lg=linear_backward(*small,upstream)
for j in range(3):
    def loss(a):
        args=small.copy(); args[j]=a
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        return np.sum(linear_recurrent(*args)[0]*upstream)
    check('Linear finite diff '+str(j),lg[j],finite_difference(loss,small[j]),atol=3e-8,rtol=2e-6)
```

</details>

<details>
<summary>3. DeltaNet：把加法记忆改为回归误差修正</summary>

```python
# alpha 按 key 行遗忘，beta 控制残差写入；return_tape 保存反向所需的每步中间状态。
def delta_recurrent(q,k,v,beta,alpha=None,state=None,return_tape=False):
    assert q.shape==k.shape and q.shape[:-1]==v.shape[:-1]==beta.shape
    # NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
    if alpha is None: alpha=np.ones_like(k)
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])) if state is None else state.copy()
    outs=[]; tape=[]
    for t in range(q.shape[-2]):
        qt,kt,vt,bt,at=q[...,t,:],k[...,t,:],v[...,t,:],beta[...,t],alpha[...,t,:]
        previous=s
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        decayed=at[..., :,None]*s
        # NumPy：einsum("...k,...kv->...v")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        prediction=np.einsum('...k,...kv->...v',kt,decayed)
        error=vt-prediction
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        s=decayed+(bt[...,None]*kt)[..., :,None]*error[...,None,:]
        # NumPy：einsum("...k,...kv->...v")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        outs.append(np.einsum('...k,...kv->...v',qt,s))
        if return_tape: tape.append((previous,decayed,error,s))
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    result=(np.stack(outs,-2),s)
    return result+(tape,) if return_tape else result

qn,kn=normalize(q),normalize(k)
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
beta=sigmoid(rng.normal(size=(B,H,N)))
# NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
scalar_alpha=np.exp(-softplus(rng.normal(size=(B,H,N,1)))*.15)
# NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
vector_alpha=np.exp(-softplus(rng.normal(size=(B,H,N,DK)))*.15)
delta,_=delta_recurrent(qn,kn,v,beta)
gated,_=delta_recurrent(qn,kn,v,beta,scalar_alpha)
kda,kda_state=delta_recurrent(qn,kn,v,beta,vector_alpha)
# NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
check('scalar KDA reduces to GDN',delta_recurrent(qn,kn,v,beta,np.broadcast_to(scalar_alpha,kn.shape))[0],gated)
# NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
check('unit decay reduces to DeltaNet',delta_recurrent(qn,kn,v,beta,np.ones_like(kn))[0],delta)
# 一个两通道的数值例子：可以直接观察行衰减、残差与外积。
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
s0=np.array([[2.,1.],[3.,4.]])
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
kt=normalize(np.array([1.,2.])); vt=np.array([2.,-1.]); at=np.array([.9,.2]); bt=.8
sb=at[:,None]*s0; pred=kt@sb; err=vt-pred; update=bt*kt[:,None]*err[None,:]
# NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
heatmaps([('S previous',s0),('D @ S',sb),('pred ; target',np.stack([pred,vt])),
          ('beta k outer error',update),('S new',sb+update)], 'Decay -> predict -> residual -> rank-one write',cmap='coolwarm')
# beta=1 单位 key 的精确覆写。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
onek=normalize(rng.normal(size=(1,1,1,3))); onev=rng.normal(size=(1,1,1,2))
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
initial=rng.normal(size=(1,1,3,2))
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
check('unit key overwrite',delta_recurrent(onek,onek,onev,np.ones((1,1,1)),state=initial)[0],onev)
```

</details>

<details>
<summary>5. 从逐 token 递推到 chunkwise 三角系统</summary>

```python
# 按行前代求解 (I+l)out=r；第 i 行只依赖前 i 行，i=0 时空和为零。
def unit_lower_solve(l,r):
    # 求 (I + 严格下三角 l) x = r；批量前代 O(C^2 * dv)。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    out=np.zeros_like(r)
    for i in range(r.shape[-2]):
        # NumPy：einsum("...j,...jv->...v")：对 j 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        out[...,i,:]=r[...,i,:]-np.einsum('...j,...jv->...v',l[...,i,:i],out[...,:i,:])
    return out

# 构造累计门、更新系数、读取系数和块尾 key；stable 路径允许门精确为零。
def chunk_factors(q,k,alpha,method):
    c,dk=k.shape[-2:]
    # NumPy：cumprod 沿指定轴做前缀乘积；axis=-2 是 token 轴，得到每个通道截至当前的累计门。
    g=np.cumprod(alpha,axis=-2)
    if method=='gemm':
        # NumPy：any 检查是否至少一个元素为真；用来拒绝数值不安全的累计门。
        if np.any(g<1e-100):
            raise ValueError('累计门过小或为零，请用 stable 路径或更小 chunk')
        kr=k/g
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        f=(k*g)@kr.swapaxes(-1,-2)
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        e=(q*g)@kr.swapaxes(-1,-2)
        kend=(g[...,-1:,:]/g)*k
    elif method=='stable':
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        f=np.zeros(k.shape[:-2]+(c,c)); e=np.zeros_like(f)
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        kend=np.zeros_like(k)
        for j in range(c):
            # NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
            decay=np.ones(k.shape[:-2]+(dk,))
            for i in range(j,c):
                if i>j: decay=decay*alpha[...,i,:]
                weighted=decay*k[...,j,:]
                # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
                f[...,i,j]=np.sum(k[...,i,:]*weighted,-1)
                # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
                e[...,i,j]=np.sum(q[...,i,:]*weighted,-1)
            kend[...,j,:]=decay*k[...,j,:]
    else: raise ValueError(method)
    # NumPy：tril 保留最后两轴的下三角；k=-1 排除对角线，适合只依赖更早 token 的更新系数。
    return g, np.tril(f,-1),np.tril(e),kend

# 块内先解伪 value u，再读出与更新块尾；返回值必须与同参数的逐 token 递推一致。
def delta_chunk(q,k,v,beta,alpha=None,chunk=4,state=None,method='stable'):
    assert chunk>0
    # NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
    if alpha is None: alpha=np.ones_like(k)
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])) if state is None else state.copy()
    outs=[]
    for start in range(0,q.shape[-2],chunk):
        sl=slice(start,start+chunk)
        qc,kc,vc,bc,ac=q[...,sl,:],k[...,sl,:],v[...,sl,:],beta[...,sl],alpha[...,sl,:]
        g,f,e,kend=chunk_factors(qc,kc,ac,method)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        l=bc[..., :,None]*f
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        rhs=bc[..., :,None]*(vc-(kc*g)@s)
        u=unit_lower_solve(l,rhs)
        outs.append((qc*g)@s+e@u)
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        s=g[...,-1,:,None]*s+kend.swapaxes(-1,-2)@u
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    return np.concatenate(outs,-2),s

for label,al in [('delta',None),('gdn',scalar_alpha),('kda',vector_alpha)]:
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    ini=rng.normal(size=(B,H,DK,DV))*.1
    ref,rs=delta_recurrent(qn,kn,v,beta,al,ini)
    for c in [1,4,16]:
        for method in ['stable','gemm']:
            actual,ss=delta_chunk(qn,kn,v,beta,al,c,ini,method)
            check(f'{label} chunk={c} {method}',actual,ref)
            check(f'{label} final state={c} {method}',ss,rs)
# 精确零门与极强衰减：稳定区间乘积路径。
az=vector_alpha.copy(); az[:,:,2,:]=0; az[:,:,5,:]=1e-200
check('zero/strong decay chunk',delta_chunk(qn,kn,v,beta,az,4)[0],delta_recurrent(qn,kn,v,beta,az)[0])
g,f,e,kend=chunk_factors(qn[:,:,:4],kn[:,:,:4],vector_alpha[:,:,:4],'stable')
heatmaps([('G: time x key',g[0,0]),('L: strict lower',(beta[:,:,:4,None]*f)[0,0]),
          ('E: causal read',e[0,0]),('K end',kend[0,0])], 'Chunk factors: solve (I+L) U = beta (V - Kg S0)')
```

</details>

<details>
<summary>5.1 通用仿射 scan：为什么数学并行不一定高效</summary>

```python
# 保存所有前缀仿射变换作为正确性基准；不是推荐生产实现，矩阵连乘成本很高。
def delta_affine_scan(q,k,v,beta,alpha,state=None):
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：eye(dk) 生成单位矩阵，只有对角为 1；左乘状态保持原值。
    eye=np.eye(k.shape[-1])
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    aa=(eye-beta[...,None,None]*k[..., :,None]*k[...,None,:])*alpha[...,None,:]
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    bb=beta[...,None,None]*k[..., :,None]*v[...,None,:]
    n=q.shape[-2]; gap=1
    while gap<n:
        # 必须从旧数组读，不能在同一级中原地污染后续输入。
        na=aa.copy(); nb=bb.copy()
        na[...,gap:,:,:]=aa[...,gap:,:,:]@aa[...,:-gap,:,:]
        nb[...,gap:,:,:]=aa[...,gap:,:,:]@bb[...,:-gap,:,:]+bb[...,gap:,:,:]
        aa,bb=na,nb; gap*=2
    if state is not None: bb=bb+aa@state[...,None,:,:]
    # NumPy：einsum("...tk,...tkv->...tv")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
    return np.einsum('...tk,...tkv->...tv',q,bb),bb[...,-1,:,:]
check('affine scan oracle',delta_affine_scan(qn,kn,v,beta,vector_alpha)[0],kda)
```

</details>

<details>
<summary>5.2 Delta/GDN/KDA 手写反向：训练不能只写 forward</summary>

```python
# 按读出→外积写入→残差预测→行衰减的反序求导，h 带着未来状态梯度。
def delta_backward(q,k,v,beta,alpha,tape,go,final_grad=None):
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    db=np.zeros_like(beta); da=np.zeros_like(alpha)
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    h=np.zeros_like(tape[0][0]) if final_grad is None else final_grad.copy()
    for t in reversed(range(q.shape[-2])):
        prev,bar,err,st=tape[t]
        qt,kt,bt,at,gt=q[...,t,:],k[...,t,:],beta[...,t],alpha[...,t,:],go[...,t,:]
        # NumPy：einsum("...kv,...v->...k")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dq[...,t,:]=np.einsum('...kv,...v->...k',st,gt)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        h=h+qt[..., :,None]*gt[...,None,:]
        # NumPy：einsum("...kv,...k,...v->...")：对 k,v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        db[...,t]=np.einsum('...kv,...k,...v->...',h,kt,err)
        # NumPy：einsum("...kv,...v->...k")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        dk[...,t,:]=bt[...,None]*np.einsum('...kv,...v->...k',h,err)
        # NumPy：einsum("...kv,...k->...v")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        de=bt[...,None]*np.einsum('...kv,...k->...v',h,kt)
        dv[...,t,:]=de
        # NumPy：einsum("...kv,...v->...k")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dk[...,t,:]-=np.einsum('...kv,...v->...k',bar,de)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        dbar=h-kt[..., :,None]*de[...,None,:]
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        da[...,t,:]=np.sum(dbar*prev,axis=-1)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        h=at[..., :,None]*dbar
    return dq,dk,dv,db,da,h

# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
args=[normalize(rng.normal(size=(1,1,3,2))),normalize(rng.normal(size=(1,1,3,2))),
      # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
      rng.normal(size=(1,1,3,2)),rng.uniform(.2,.8,(1,1,3)),rng.uniform(.2,.9,(1,1,3,2))]
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
ini=rng.normal(size=(1,1,2,2)); outs,ss,tape=delta_recurrent(*args,state=ini,return_tape=True)
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
gout=rng.normal(size=outs.shape); gs=rng.normal(size=ss.shape)
analytic=delta_backward(*args,tape,gout,gs)
for j in range(6):
    vals=args+[ini]
    def loss(z):
        aa=vals.copy(); aa[j]=z
        yy,st=delta_recurrent(*aa[:5],state=aa[5])
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        return np.sum(yy*gout)+np.sum(st*gs)
    check('KDA backward finite diff '+str(j),analytic[j],finite_difference(loss,vals[j]),atol=3e-8,rtol=2e-6)
# 在输入 beta logits 上做一次真正的梯度下降，证明反向可用于优化。
# NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
logits=np.zeros((1,1,3)); target=rng.normal(size=outs.shape)
def beta_loss(b):
    yy,_=delta_recurrent(args[0],args[1],args[2],sigmoid(b),args[4],state=ini)
    # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
    return .5*np.sum((yy-target)**2)
y,_,tp=delta_recurrent(args[0],args[1],args[2],sigmoid(logits),args[4],state=ini,return_tape=True)
db=delta_backward(args[0],args[1],args[2],sigmoid(logits),args[4],tp,y-target)[3]
new_logits=logits-.01*db*sigmoid(logits)*(1-sigmoid(logits))
require('one gradient step decreases loss', beta_loss(new_logits)<beta_loss(logits))
print('beta gate training loss:',beta_loss(logits),'->',beta_loss(new_logits))
```

</details>

<details>
<summary>6. FlashAttention 的数学核心：online softmax</summary>

```python
# m/l 每条 query 一个数，a 每条 query 一个 value 向量；新最大值改变时一起重标定。
def online_update(m,l,a,x,v):
    # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    new_m=np.maximum(m,np.max(x,axis=-1))
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
    safe=np.where(np.isfinite(new_m),new_m,0.)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    rescale=np.exp(m-safe)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    weights=np.exp(x-safe[...,None])
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    new_l=rescale*l+weights.sum(-1)
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    new_a=rescale[...,None]*a+weights@v
    return new_m,new_l,new_a

# 只计算当前 Q/K 小块，并按全局位置遮蔽未来；外部 keep 同时切取对应行列。
def tile_scores(qc,kc,i,j,causal,keep,q_start=0):
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    scores=qc@kc.swapaxes(-1,-2)/math.sqrt(qc.shape[-1])
    if causal:
        mask=position_mask(qc.shape[-2],kc.shape[-2],i+q_start,j)
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        scores=np.where(mask,scores,-np.inf)
    if keep is not None:
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        scores=np.where(keep[...,i:i+qc.shape[-2],j:j+kc.shape[-2]],scores,-np.inf)
    return scores

# K/V tile 外循环、query tile 内循环；保留归一化输出，在更新时恢复未归一化分子。
def flash1_model(q,k,v,rows=4,cols=3,causal=True,keep=None,q_start=0):
    assert rows>0 and cols>0
    # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    m=np.full(q.shape[:-1],-np.inf); l=np.zeros_like(m)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    out=np.zeros(q.shape[:-1]+(v.shape[-1],))
    for j in range(0,k.shape[-2],cols):
        kc,vc=k[...,j:j+cols,:],v[...,j:j+cols,:]
        for i in range(0,q.shape[-2],rows):
            sl=slice(i,i+rows); qc=q[...,sl,:]
            x=tile_scores(qc,kc,i,j,causal,keep,q_start)
            mm,ll,aa=online_update(m[...,sl],l[...,sl],out[...,sl,:]*l[...,sl,None],x,vc)
            # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
            # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
            # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
            out[...,sl,:]=np.divide(aa,ll[...,None],out=np.zeros_like(aa),where=ll[...,None]>0)
            m[...,sl]=mm; l[...,sl]=ll
    return out

# 每个 query tile 拥有一组 m,l,a；依次消费 KV tiles，末尾归一化一次。
def flash2_model(q,k,v,rows=4,cols=3,causal=True,keep=None,q_start=0,return_lse=False):
    assert rows>0 and cols>0
    outputs=[]; logsum=[]
    for i in range(0,q.shape[-2],rows):
        qc=q[...,i:i+rows,:]
        # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        m=np.full(qc.shape[:-1],-np.inf); l=np.zeros_like(m)
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        a=np.zeros(qc.shape[:-1]+(v.shape[-1],))
        for j in range(0,k.shape[-2],cols):
            kc,vc=k[...,j:j+cols,:],v[...,j:j+cols,:]
            x=tile_scores(qc,kc,i,j,causal,keep,q_start)
            m,l,a=online_update(m,l,a,x,vc)
        # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        outputs.append(np.divide(a,l[...,None],out=np.zeros_like(a),where=l[...,None]>0))
        # NumPy：log 是自然对数；调用前须保证合法正输入，空行另设安全值。
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        logsum.append(m+np.log(np.where(l>0,l,1.)))
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    result=np.concatenate(outputs,-2)
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    return (result,np.concatenate(logsum,-1)) if return_lse else result

for causal in [False,True]:
    keep=rng.random((B,1,N,N))>.25; keep[:,:,0,:]=False
    ref,_=softmax_attention(q,k,v,causal=causal,keep=keep)
    for r,c in [(1,1),(4,3),(16,7)]:
        for fn in [flash1_model,flash2_model]:
            check(f'{fn.__name__} {causal} {r}x{c}',fn(q,k,v,r,c,causal,keep),ref)
# 大 logits 测试；不是先 exp(qk) 再 softmax。
check('flash stable large logits',flash2_model(q*300,k*300,v),softmax_attention(q*300,k*300,v,causal=True)[0])
check('flash decode offset',flash2_model(q[:,:,-1:],k,v,q_start=N-1),o[:,:,-1:])
```

</details>

<details>
<summary>分块图解：每个 tile 只负责一部分键，但输出必须全局归一化</summary>

```python
from matplotlib.patches import Rectangle
fig,axes=plt.subplots(1,2,figsize=(12,4))
# NumPy：tri(n) 得到含对角线的下三角 1/0 矩阵，表示可以读取自己及过去。
mask=np.tri(9); axes[0].imshow(mask,cmap='Blues',vmin=0,vmax=1)
for i in range(0,9,3):
    for j in range(0,9,4):
        axes[0].add_patch(Rectangle((j-.5,i-.5),min(4,9-j),3,fill=False,edgecolor='#d57525',linewidth=2))
for i in range(9):
    for j in range(9):
        axes[0].text(j,i,f'{i},{j}' if j<=i else 'x',fontsize=7,ha='center',va='center')
axes[0].set_title('Causal score tiles: R=3, T=4'); axes[0].set_xlabel('key position j'); axes[0].set_ylabel('query position i')
# 选后3行，确保多个 KV tile 都有真实贡献。
# NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
# NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
qt=q[0,0,6:9]; mm=np.full(3,-np.inf); ll=np.zeros(3); aa=np.zeros((3,DV)); history=[]
for j in range(0,9,4):
    scores=tile_scores(qt,k[0,0,j:j+4],6,j,True,None)
    mm,ll,aa=online_update(mm,ll,aa,scores,v[0,0,j:j+4])
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    history.append(np.stack([mm,ll,aa[:,0]],-1))
# NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
history=np.concatenate(history,axis=0)
axes[1].imshow(history,cmap='coolwarm',aspect='auto')
axes[1].set_xticks([0,1,2],['m','l','a[value=0]'])
axes[1].set_yticks(range(9),[f'tile {j}, query {i}' for j in range(3) for i in range(6,9)])
# NumPy：ndenumerate 同时返回坐标元组与该格数值，用来为热力图加文字。
for (i,j),val in np.ndenumerate(history): axes[1].text(j,i,f'{val:.3f}',ha='center',va='center',fontsize=8)
axes[1].set_title('Running online-softmax statistics'); plt.tight_layout(); plt.show()
check('illustrated tile output',aa/ll[:,None],o[0,0,6:9])
```

</details>

<details>
<summary>6.1 Flash 反向：用 LSE 重建 P，避免保存全 N×N</summary>

```python
# 由 score 与 LSE 重算局部概率，再累加 Q/K/V 梯度；这里使用单线程确定性累加。
def flash_backward(q,k,v,go,out,lse,rows=4,cols=3,causal=True,keep=None):
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    d=(go*out).sum(-1); scale=q.shape[-1]**-.5
    for i in range(0,q.shape[-2],rows):
        qi=q[...,i:i+rows,:]; gi=go[...,i:i+rows,:]
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
        ll=lse[...,i:i+rows]; safe=np.where(np.isfinite(ll),ll,0.)
        for j in range(0,k.shape[-2],cols):
            kj,vj=k[...,j:j+cols,:],v[...,j:j+cols,:]
            x=tile_scores(qi,kj,i,j,causal,keep)
            # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
            # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
            pp=np.exp(x-safe[...,None])
            # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
            dp=gi@vj.swapaxes(-1,-2)
            ds=pp*(dp-d[...,i:i+rows,None])
            dq[...,i:i+rows,:]+=scale*(ds@kj)
            # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
            dk[...,j:j+cols,:]+=scale*(ds.swapaxes(-1,-2)@qi)
            # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
            dv[...,j:j+cols,:]+=pp.swapaxes(-1,-2)@gi
    return dq,dk,dv
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
fo,fl=flash2_model(q,k,v,return_lse=True); gg=rng.normal(size=fo.shape)
fg=flash_backward(q,k,v,gg,fo,fl)
sg=softmax_backward(q,k,v,p,gg)
for j in range(3): check('Flash recomputed backward '+str(j),fg[j],sg[j])
```

</details>

<details>
<summary>7. FlashAttention-3：数学相同，硬件执行不同</summary>

```python
# 用双槽位模拟加载/消费依赖；实际仍是顺序 CPU 执行，不含 CUDA 异步指令。
def flash3_pipeline_model(q,k,v,rows=4,cols=3,causal=True):
    outputs=[]; trace=[]
    for i in range(0,q.shape[-2],rows):
        # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
        qc=q[...,i:i+rows,:]; m=np.full(qc.shape[:-1],-np.inf)
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        l=np.zeros_like(m); a=np.zeros(qc.shape[:-1]+(v.shape[-1],))
        buffers=[None,None]; starts=list(range(0,k.shape[-2],cols))
        def load(step):
            slot=step%2; j=starts[step]
            assert buffers[slot] is None
            buffers[slot]=(j,k[...,j:j+cols,:].copy(),v[...,j:j+cols,:].copy())
            trace.append((i,step,'load',slot))
        load(0)
        for step in range(len(starts)):
            if step+1<len(starts): load(step+1)
            slot=step%2; j,kc,vc=buffers[slot]
            x=tile_scores(qc,kc,i,j,causal,None)
            m,l,a=online_update(m,l,a,x,vc)
            trace.append((i,step,'consume',slot)); buffers[slot]=None
        # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        outputs.append(np.divide(a,l[...,None],out=np.zeros_like(a),where=l[...,None]>0))
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    return np.concatenate(outputs,-2),trace
f3,trace=flash3_pipeline_model(q,k,v)
check('FA3 pipeline math',f3,o)
print('Double-buffer event trace:',trace[:8])
# 自画调度图：横轴为示意时间，非测量时间，不暗示精确 stall 数。
fig,ax=plt.subplots(figsize=(10,3))
loads=[0.,1.,3.,5.]
qk=[1.,2.,4.,6.]; sm=[2.,3.,5.,7.]; pv=[3.,5.,7.,8.]
for j in range(4):
    ax.broken_barh([(loads[j],.7)],(2.1,.6),facecolors='#5896c7')
    ax.broken_barh([(qk[j],1.)],(1.1,.6),facecolors='#edb14c')
    ax.broken_barh([(pv[j],1.)],(1.1,.6),facecolors='#d98749')
    ax.broken_barh([(sm[j],.7)],(.1,.6),facecolors='#5cad8b')
    ax.text(loads[j]+.05,2.4,f'L{j}',fontsize=9)
    ax.text(qk[j]+.05,1.4,f'QK{j}',fontsize=9)
    ax.text(pv[j]+.05,1.4,f'PV{j}',fontsize=9)
    ax.text(sm[j]+.05,.4,f'SM{j}',fontsize=9)
ax.set_yticks([.4,1.4,2.4],['Softmax','Tensor Core work','TMA producer'])
ax.set_xlabel('Illustrative dependency-respecting schedule, not measured cycles')
ax.set_xlim(-.2,9.3); ax.set_ylim(0,3); plt.tight_layout(); plt.show()

def hadamard(n):
    assert n>0 and n&(n-1)==0
    # NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
    h=np.ones((1,1))
    # NumPy：block 按嵌套列表拼出大矩阵；这里构造 Hadamard 的四个正负子块。
    while h.shape[0]<n: h=np.block([[h,h],[h,-h]])
    # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
    return h/np.sqrt(n)
def quantize_uniform(x):
    # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    # NumPy：abs 对每个元素取绝对值，形状不变。
    scale=np.maximum(np.max(np.abs(x),axis=(-2,-1),keepdims=True),1e-12)/127
    # NumPy：clip 把每格限制在给定上下界之间，用来模拟有限的量化范围。
    # NumPy：round 将每格舍入到邻近整数值；返回数组通常仍是浮点 dtype。
    return np.clip(np.round(x/scale),-127,127)*scale
rr=hadamard(DK)*rng.choice([-1.,1.],size=(1,DK))
# NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
check('orthogonal transform invariant logits',(q@rr)@(k@rr).swapaxes(-1,-2),q@k.swapaxes(-1,-2))
# NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
original=q@k.swapaxes(-1,-2)
for label,qa,ka in [('plain',q,k),('rotated',q@rr,k@rr)]:
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    # NumPy：linalg.norm 沿给定轴求欧氏长度 sqrt(sum(x*x))，不是对矩阵每格取绝对值。
    err=np.linalg.norm(quantize_uniform(qa)@quantize_uniform(ka).swapaxes(-1,-2)-original)/np.linalg.norm(original)
    print('Uniform INT8-style demo relative logits error',label,err)
```

</details>

<details>
<summary>8. PagedAttention：逻辑连续，物理不连续</summary>

```python
class PagePool:
    def __init__(self,pages,heads,page_size,dk,dv):
        assert pages>0 and page_size>0
        self.P=page_size; self.H=heads
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        self.k=np.zeros((pages,heads,page_size,dk))
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        self.v=np.zeros((pages,heads,page_size,dv))
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        self.refs=np.zeros(pages,dtype=int)
        self.free=list(reversed(range(pages)))
    def alloc(self):
        if not self.free: raise MemoryError('KV page pool exhausted')
        p=self.free.pop(); assert self.refs[p]==0
        self.refs[p]=1
        self.k[p].fill(0); self.v[p].fill(0)
        return p
    def retain(self,p):
        assert self.refs[p]>0; self.refs[p]+=1
    def release(self,p):
        assert self.refs[p]>0; self.refs[p]-=1
        if self.refs[p]==0: self.free.append(p)
    def audit(self):
        assert len(self.free)==len(set(self.free))
        # NumPy：flatnonzero 返回扁平数组中非零元素的整数位置，用来核对空闲页号。
        assert set(self.free)==set(np.flatnonzero(self.refs==0))
        # NumPy：all 检查布尔数组是否全部为真，返回真/假而非数值误差大小。
        assert np.all(self.refs>=0)

class PagedSequence:
    def __init__(self,pool):
        self.pool=pool; self.table=[]; self.length=0; self.closed=False
    def append(self,k,v):
        assert not self.closed
        p=self.pool
        assert k.shape==(p.H,p.k.shape[-1]) and v.shape==(p.H,p.v.shape[-1])
        offset=self.length%p.P
        if offset==0:
            physical=p.alloc(); self.table.append(physical)
        else:
            physical=self.table[-1]
            if p.refs[physical]>1:
                new=p.alloc()  # 失败时，旧页及引用计数还没有改动。
                p.k[new]=p.k[physical]; p.v[new]=p.v[physical]
                p.release(physical); self.table[-1]=new; physical=new
        p.k[physical,:,offset,:]=k; p.v[physical,:,offset,:]=v
        self.length+=1
    def fork(self):
        assert not self.closed
        child=PagedSequence(self.pool); child.table=self.table.copy(); child.length=self.length
        for p in child.table: self.pool.retain(p)
        return child
    def blocks(self):
        assert not self.closed
        for logical,physical in enumerate(self.table):
            count=min(self.pool.P,self.length-logical*self.pool.P)
            yield self.pool.k[physical,:,:count], self.pool.v[physical,:,:count]
    def gather(self):
        blocks=list(self.blocks())
        if not blocks:
            # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
            return np.zeros((self.pool.H,0,self.pool.k.shape[-1])),np.zeros((self.pool.H,0,self.pool.v.shape[-1]))
        # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
        return np.concatenate([x[0] for x in blocks],1),np.concatenate([x[1] for x in blocks],1)
    def decode(self,q):
        assert q.shape==(self.pool.H,self.pool.k.shape[-1]) and self.length>0
        # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        m=np.full((self.pool.H,1),-np.inf); l=np.zeros_like(m)
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        a=np.zeros((self.pool.H,1,self.pool.v.shape[-1]))
        for kk,vv in self.blocks():
            # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
            scores=q[:,None,:]@kk.swapaxes(-1,-2)/math.sqrt(q.shape[-1])
            m,l,a=online_update(m,l,a,scores,vv)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        return (a/l[...,None])[:,0,:]
    def close(self):
        if self.closed: return
        for p in self.table: self.pool.release(p)
        self.table=[]; self.length=0; self.closed=True

pool=PagePool(12,H,3,DK,DV)
# 先占用一页，构造非连续的物理页分布。
hole=pool.alloc(); seq=PagedSequence(pool)
for t in range(4): seq.append(k[0,:,t],v[0,:,t])
pool.release(hole)
child=seq.fork(); old=seq.gather()
child.append(k[0,:,4],v[0,:,4])
check('COW parent unchanged K',seq.gather()[0],old[0]); check('COW parent unchanged V',seq.gather()[1],old[1])
require('partial last page copied',seq.table[-1]!=child.table[-1])
for t in range(5,N): child.append(k[0,:,t],v[0,:,t])
kk,vv=child.gather()
check('paged gather logical order',kk,k[0])
check('paged decode dense',child.decode(q[0,:,-1]),softmax_attention(q[0,:,-1:, :],kk,vv)[0][:,0])
pool.audit()
fig,ax=plt.subplots(figsize=(9,3))
for i,pid in enumerate(child.table):
    ax.text(i*2+.5,2,f'Logical {i}',ha='center',bbox=dict(boxstyle='round',fc='#e6f1fa'))
    ax.text(i*2+.5,.4,f'Physical {pid}',ha='center',bbox=dict(boxstyle='round',fc='#e7f4eb'))
    ax.annotate('',xy=(i*2+.5,.75),xytext=(i*2+.5,1.8),arrowprops=dict(arrowstyle='->'))
ax.set_xlim(-.5,7); ax.set_ylim(0,2.5); ax.axis('off'); ax.set_title('Page table: address mapping, each block has 3 token slots'); plt.show()
seq.close(); child.close(); pool.audit()
require('all pages released',len(pool.free)==12)
# 满页共享后追加：旧页保持共享，新建下一页。
p2=PagePool(3,1,2,2,2); a=PagedSequence(p2)
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
for _ in range(2): a.append(np.ones((1,2)),np.ones((1,2)))
# NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
b=a.fork(); b.append(np.zeros((1,2)),np.zeros((1,2)))
require('full-page fork no old-page copy',a.table[0]==b.table[0] and len(b.table)==2)
a.close(); b.close(); p2.audit()
# OOM 不损坏原始共享尾页。
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
p3=PagePool(1,1,2,2,2); a=PagedSequence(p3); a.append(np.ones((1,2)),np.ones((1,2))); b=a.fork()
try:
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    b.append(np.zeros((1,2)),np.zeros((1,2)))
    raise AssertionError('Expected OOM')
except MemoryError: pass
require('OOM leaves references intact',p3.refs[0]==2 and b.length==1)
a.close(); b.close(); p3.audit()
```

</details>

<details>
<summary>9. RadixAttention：复用 token 前缀的 KV 计算</summary>

```python
@dataclass(eq=False)
class RadixNode:
    edge: tuple=()
    kv: object=None  # (K,V)，各为 [tokens,heads,dim]
    parent: object=None
    children: dict=field(default_factory=dict)
    pins: int=0
    stamp: int=0

def lcp(a,b):
    i=0
    while i<min(len(a),len(b)) and a[i]==b[i]: i+=1
    return i

class RadixCache:
    def __init__(self,namespace):
        self.namespace=namespace; self.root=RadixNode(); self.clock=0
    def tick(self,node):
        self.clock+=1; node.stamp=self.clock
    def insert(self,tokens,k,v,namespace):
        assert namespace==self.namespace, 'cache namespace mismatch'
        tokens=tuple(tokens); assert len(tokens)==len(k)==len(v)
        node=self.root; pos=0
        while pos<len(tokens):
            rem=tokens[pos:]; child=node.children.get(rem[0])
            if child is None:
                new=RadixNode(rem,(k[pos:].copy(),v[pos:].copy()),node)
                node.children[rem[0]]=new; self.tick(new); return
            n=lcp(rem,child.edge)
            # 若相同 prefix 的 KV 不同，拒绝静默命中。
            # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
            np.testing.assert_allclose(child.kv[0][:n],k[pos:pos+n],atol=1e-10)
            # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
            np.testing.assert_allclose(child.kv[1][:n],v[pos:pos+n],atol=1e-10)
            if n<len(child.edge):
                mid=RadixNode(child.edge[:n],(child.kv[0][:n].copy(),child.kv[1][:n].copy()),node)
                mid.pins=child.pins; mid.stamp=child.stamp
                node.children[rem[0]]=mid
                child.edge=child.edge[n:]; child.kv=(child.kv[0][n:].copy(),child.kv[1][n:].copy())
                child.parent=mid; mid.children[child.edge[0]]=child
                child=mid
            self.tick(child); node=child; pos+=n
    def match(self,tokens,namespace):
        assert namespace==self.namespace, 'cache namespace mismatch'
        tokens=tuple(tokens); node=self.root; pos=0; ks=[]; vs=[]
        while pos<len(tokens):
            child=node.children.get(tokens[pos])
            if child is None: break
            n=lcp(tokens[pos:],child.edge)
            if not n: break
            ks.append(child.kv[0][:n]); vs.append(child.kv[1][:n])
            pos+=n; node=child; self.tick(node)
            if n<len(child.edge): break
        # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
        return pos,(np.concatenate(ks) if ks else None,np.concatenate(vs) if vs else None),node
    @contextmanager
    def lease(self,tokens,namespace):
        matched=self.match(tokens,namespace); node=matched[2]
        cur=node
        while cur is not self.root: cur.pins+=1; cur=cur.parent
        try: yield matched
        finally:
            # 动态沿当前父指针回溯，支持 lease 期间边分裂。
            cur=node
            while cur is not self.root:
                cur.pins-=1; assert cur.pins>=0; cur=cur.parent
    def nodes(self):
        stack=list(self.root.children.values()); result=[]
        while stack:
            n=stack.pop(); result.append(n); stack.extend(n.children.values())
        return result
    @property
    def tokens(self): return sum(len(n.edge) for n in self.nodes())
    def evict_to(self,budget):
        assert budget>=0
        while self.tokens>budget:
            leaves=[n for n in self.nodes() if not n.children and n.pins==0]
            if not leaves: break
            victim=min(leaves,key=lambda n:n.stamp)
            del victim.parent.children[victim.edge[0]]
        return self.tokens<=budget
    def edges(self):
        return sorted([(n.edge,n.pins,len(n.children)) for n in self.nodes()])

ns=('toy-model-v1','tokenizer-v1','position-start-0','no-adapter')
cache=RadixCache(ns)
# 数据只为结构测试；真实 KV 必须包含上下文影响，下一单元会用 causal 层验证。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
embk=rng.normal(size=(20,H,DK)); embv=rng.normal(size=(20,H,DV))
def insert_ids(ids): cache.insert(ids,embk[ids],embv[ids],ns)
insert_ids([1,2,3,4]); insert_ids([1,2,5]); insert_ids([8,9])
require('radix unique prefix storage',cache.tokens==7)
matched,kv,node=cache.match([1,2,3,9],ns)
require('partial compressed-edge match',matched==3)
check('radix KV match order',kv[0],embk[[1,2,3]])
with cache.lease([1,2,3,4],ns):
    require('pinned cache cannot evict all',not cache.evict_to(0))
    require('active prefix survives',cache.match([1,2,3,4],ns)[0]==4)
require('unpin enables eviction',cache.evict_to(0) and cache.tokens==0)
insert_ids([1,2,3,4])
with cache.lease([1,2,3,4],ns):
    insert_ids([1,2,5])
    require('split inherits path pin',any(n.edge==(1,2) and n.pins==1 for n in cache.nodes()))
require('split unpin balanced',all(n.pins==0 for n in cache.nodes()))
try:
    cache.match([1,2],('different-model',))
    raise AssertionError('namespace should fail')
except AssertionError as e:
    assert str(e)=='cache namespace mismatch'
    TESTS.append('namespace isolation')
print('Compressed edges (tokens, pins, children):',cache.edges())
```

</details>

<details>
<summary>9.1 端到端 prefix reuse：真实的上下文相关 KV</summary>

```python
dmodel=8; hh=2; dd=4
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
embedding=rng.normal(size=(30,dmodel))*.2
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
weights=[tuple(rng.normal(size=(dmodel,dmodel))*.2 for _ in range(4)) for _ in range(2)]
def toy_layer(x,w,prefix=None):
    wq,wk,wv,wo=w
    qq=split_heads((x@wq)[None],hh); kk=split_heads((x@wk)[None],hh); vv=split_heads((x@wv)[None],hh)
    length=0 if prefix is None else prefix[0].shape[0]
    if prefix is None: allk,allv=kk,vv
    else:
        pk,pv=prefix
        # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
        # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
        allk=np.concatenate([pk.transpose(1,0,2)[None],kk],axis=2)
        # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
        # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
        allv=np.concatenate([pv.transpose(1,0,2)[None],vv],axis=2)
    yy=softmax_attention(qq,allk,allv,causal=True,q_start=length)[0]
    result=x+merge_heads(yy)[0]@wo
    # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
    return result,(allk[0].transpose(1,0,2),allv[0].transpose(1,0,2))

def full_toy(ids):
    x=embedding[ids]; states=[]
    for w in weights:
        x,kv=toy_layer(x,w); states.append(kv)
    return x,states
prefix=[2,4,6,8]; ids=prefix+[10,12,14]
_,states=full_toy(prefix)
layer_caches=[RadixCache(ns+('layer',i)) for i in range(2)]
for i,c in enumerate(layer_caches): c.insert(prefix,*states[i],ns+('layer',i))
x=embedding[ids[len(prefix):]]
for i,(w,c) in enumerate(zip(weights,layer_caches)):
    hit,cached,_=c.match(ids,ns+('layer',i))
    require('layer prefix hit '+str(i),hit==len(prefix))
    x,newkv=toy_layer(x,w,cached)
    c.insert(ids,*newkv,ns+('layer',i))
full,_=full_toy(ids)
check('two-layer radix cached suffix equals full recomputation',x,full[len(prefix):])
print('Cached prefix:',len(prefix),'new tokens:',len(ids)-len(prefix))
# 真正的压缩 radix 树图，边标签来自插入的 token IDs。
fig,ax=plt.subplots(figsize=(8,3))
positions={'root':(0,2),'prefix':(0,1),'left':(-1.7,0),'right':(1.7,0)}
for a,b,lab in [('root','prefix','[1, 2]'),('prefix','left','[3, 4]'),('prefix','right','[5]')]:
    x1,y1=positions[a]; x2,y2=positions[b]
    ax.annotate('',xy=(x2,y2+.15),xytext=(x1,y1-.1),arrowprops=dict(arrowstyle='->'))
    ax.text((x1+x2)/2+.1,(y1+y2)/2,lab,bbox=dict(fc='white',ec='none'))
for name,(xx,yy) in positions.items(): ax.text(xx,yy,name,ha='center',bbox=dict(boxstyle='round',fc='#e8f2fa'))
ax.set_xlim(-2.8,2.8); ax.set_ylim(-.4,2.5); ax.axis('off'); ax.set_title('Compressed radix tree: tokens on edges'); plt.show()
```

</details>

<details>
<summary>10. 一个可运行的 KDA token-mixing 层</summary>

```python
# 每个特征通道独立卷积，lag=0 是当前 token，lag>0 只读取更早 token。
def short_conv(x,w):
    # x [B,N,D], w [window,D], w[0] 是当前 token 的权重。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    out=np.zeros_like(x)
    for lag in range(min(len(w),x.shape[1])):
        out[:,lag:,:]+=x[:,:x.shape[1]-lag,:]*w[lag]
    return out

def silu(x): return x*sigmoid(x)

def make_kda_weights(dmodel,heads,dk,dv,rank=4,window=3,seed=7):
    rr=np.random.default_rng(seed)
    # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
    def w(a,b): return rr.normal(size=(a,b))/np.sqrt(a)
    return dict(wq=w(dmodel,heads*dk),wk=w(dmodel,heads*dk),wv=w(dmodel,heads*dv),
                cq=w(window,heads*dk),ck=w(window,heads*dk),cv=w(window,heads*dv),
                ad=w(dmodel,rank),au=w(rank,heads*dk),bl=w(dmodel,heads),
                gd=w(dmodel,rank),gu=w(rank,heads*dv),wo=w(heads*dv,dmodel),
                # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
                # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
                alog=np.full((1,heads,1,dk),-2.),dtbias=np.zeros((1,heads,1,dk)),
                # NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
                norm_weight=np.ones((1,heads,1,dv)))

# 投影→短因果卷积→激活/归一化→门→状态更新→归一化/输出门→合头投影。
def kda_layer(x,w,heads,method='recurrent'):
    qq=normalize(split_heads(silu(short_conv(x@w['wq'],w['cq'])),heads))
    kk=normalize(split_heads(silu(short_conv(x@w['wk'],w['ck'])),heads))
    vv=split_heads(silu(short_conv(x@w['wv'],w['cv'])),heads)
    raw=split_heads((x@w['ad'])@w['au'],heads)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    log_alpha=-np.exp(w['alog'])*softplus(raw+w['dtbias'])
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    alpha=np.exp(log_alpha)
    # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
    beta=sigmoid(x@w['bl']).transpose(0,2,1)
    if method=='recurrent': yy,_=delta_recurrent(qq,kk,vv,beta,alpha)
    elif method=='chunk': yy,_=delta_chunk(qq,kk,vv,beta,alpha,chunk=4)
    else: raise ValueError(method)
    # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
    # NumPy：mean 沿指定轴取平均；这里对每个 token 的 value 通道计算均方。
    yy=yy/np.sqrt(np.mean(yy*yy,axis=-1,keepdims=True)+1e-6)*w['norm_weight']
    gate=sigmoid(split_heads((x@w['gd'])@w['gu'],heads))
    return merge_heads(yy*gate)@w['wo']

# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
x=rng.normal(size=(2,9,8)); kw=make_kda_weights(8,2,4,3)
y=kda_layer(x,kw,2); yc=kda_layer(x,kw,2,'chunk')
check('complete KDA layer chunk recurrence',yc,y,atol=1e-9)
x_future=x.copy(); x_future[:,5:]+=20
check('complete KDA layer causal',kda_layer(x_future,kw,2)[:,:5],y[:,:5])
print('KDA layer shape:',x.shape,'->',y.shape)
```

</details>

<details>
<summary>11. 复杂度：必须同时说清时间、状态、训练激活、IO</summary>

```python
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
lengths=np.array([32,64,128,256,512,1024,2048,4096])
kv_bytes=lengths*8*256*2
# NumPy：full_like 复制参照数组的形状和 dtype，再用给定标量填充。
state_bytes=np.full_like(lengths,8*128*128*4)
score_bytes=lengths**2*8*4
fig,ax=plt.subplots(figsize=(8,4))
for label,yy in [('FP16 KV: 8 heads',kv_bytes),('FP32 KDA state',state_bytes),('FP32 dense scores',score_bytes)]:
    ax.loglog(lengths,yy/2**20,marker='o',label=label)
ax.set_xlabel('Sequence length N'); ax.set_ylabel('MiB, single layer/request')
ax.set_title('Analytical storage only: inputs, weights and allocator excluded')
ax.grid(alpha=.2); ax.legend(); plt.tight_layout(); plt.show()
```

</details>

<details>
<summary>12. 实验：可复现的 CPU 时间与功能验证</summary>

```python
def median_ms(fn,repeats=3):
    fn(); samples=[]
    for _ in range(repeats):
        start=time.perf_counter(); fn(); samples.append((time.perf_counter()-start)*1000)
    # NumPy：median 取多次测量的中位数；它描述本机 CPU 计时，不是 GPU 吞吐。
    return float(np.median(samples))
bench=[]
for nn in [32,64,128]:
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    qq=normalize(rng.normal(size=(1,1,nn,8))); kk=normalize(rng.normal(size=qq.shape))
    # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
    # NumPy：full_like 复制参照数组的形状和 dtype，再用给定标量填充。
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    vv=rng.normal(size=(1,1,nn,8)); bb=np.full((1,1,nn),.7); aa=np.full_like(kk,.95)
    funcs={'Dense softmax':lambda:softmax_attention(qq,kk,vv,causal=True)[0],
           'FA2 math model':lambda:flash2_model(qq,kk,vv,16,16),
           'Linear recurrent':lambda:linear_recurrent(qq,kk,vv)[0],
           'KDA recurrent':lambda:delta_recurrent(qq,kk,vv,bb,aa)[0],
           'KDA GEMM chunk':lambda:delta_chunk(qq,kk,vv,bb,aa,16,method='gemm')[0]}
    for name,fn in funcs.items(): bench.append((nn,name,median_ms(fn)))
print('N    implementation            median CPU ms')
for nn,name,ms in bench: print(f'{nn:<4} {name:<25} {ms:9.4f}')
fig,ax=plt.subplots(figsize=(9,4))
for name in funcs:
    points=[(n,t) for n,label,t in bench if label==name]
    ax.plot([x[0] for x in points],[x[1] for x in points],marker='o',label=name)
ax.set_xlabel('N'); ax.set_ylabel('median CPU milliseconds'); ax.set_title('NumPy reference implementations, not GPU kernel benchmarks')
ax.legend(fontsize=8); ax.grid(alpha=.2); plt.tight_layout(); plt.show()
```

</details>

<details>
<summary>12.1 综合回归测试</summary>

```python
# KDA / GDN / DeltaNet 真正跨调用传递状态，而非每次从零开始。
# NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
# NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
for label,al in [('Delta',np.ones_like(kn)),('GDN',np.broadcast_to(scalar_alpha,kn.shape)),('KDA',vector_alpha)]:
    whole,final=delta_recurrent(qn,kn,v,beta,al)
    left,state=delta_recurrent(qn[:,:,:4],kn[:,:,:4],v[:,:,:4],beta[:,:,:4],al[:,:,:4])
    right,state=delta_recurrent(qn[:,:,4:],kn[:,:,4:],v[:,:,4:],beta[:,:,4:],al[:,:,4:],state)
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    check(label+' streaming split',np.concatenate([left,right],2),whole)
    check(label+' streaming final',state,final)
left,state=linear_recurrent(q[:,:,:4],k[:,:,:4],v[:,:,:4])
right,state=linear_recurrent(q[:,:,4:],k[:,:,4:],v[:,:,4:],state)
# NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
check('Linear streaming split',np.concatenate([left,right],2),lr)
# beta=0 时只有通道遗忘；从非零 S0 验证，避免零状态平凡通过。
# NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
sinit=rng.normal(size=(B,H,DK,DV)); bzero=np.zeros_like(beta)
_,sf=delta_recurrent(qn,kn,v,bzero,vector_alpha,sinit)
# NumPy：prod 沿指定轴把元素相乘；这里将多步遗忘门合成为总衰减。
# NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
check('beta zero preserves pure decay',sf,np.prod(vector_alpha,axis=2)[..., :,None]*sinit)
# 不同 query/KV 长度和任意 mask 的 Flash 对照。
qr=q[:,:,:5]; kr=k[:,:,:7]; vr=v[:,:,:7]
keep=rng.random((B,1,5,7))>.3; keep[:,:,2,:]=False
for fn in [flash1_model,flash2_model]:
    check(fn.__name__+' rectangular',fn(qr,kr,vr,3,4,False,keep),softmax_attention(qr,kr,vr,keep=keep)[0])
# MQA/GQA 手动分组 oracle。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
gq=rng.normal(size=(1,4,5,3)); gk=rng.normal(size=(1,2,5,3)); gv=rng.normal(size=(1,2,5,2))
gout=gqa(gq,gk,gv,causal=True)
for head in range(4):
    check('GQA group '+str(head),gout[:,head:head+1],softmax_attention(gq[:,head:head+1],gk[:,head//2:head//2+1],gv[:,head//2:head//2+1],causal=True)[0])
# 因果性：未来值的巨大扰动不改变前5个输出。
vfuture=v.copy(); vfuture[:,:,5:]+=100
for name,fn in [('Softmax',lambda vv:softmax_attention(q,k,vv,causal=True)[0]),
                ('Linear',lambda vv:linear_recurrent(q,k,vv)[0]),
                ('KDA',lambda vv:delta_chunk(qn,kn,vv,beta,vector_alpha,4)[0]),
                ('Flash',lambda vv:flash2_model(q,k,vv))]:
    check(name+' no future leakage',fn(vfuture)[:,:,:5],fn(v)[:,:,:5])
# 覆写与叠加的关联记忆实验。
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
keys=np.array([[[[1.,0.],[1.,0.],[0.,1.]]]])
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
values=np.array([[[[1.,2.],[7.,8.],[3.,4.]]]])
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
_,memory=delta_recurrent(keys,keys,values,np.ones((1,1,3)))
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
check('associative overwrite two keys',memory[0,0],np.array([[7.,8.],[3.,4.]]))
print('All required CPU checks passed:',len(TESTS))
```

</details>

<details>
<summary>13. 可选：PyTorch 与官方 GPU 内核对照</summary>

```python
OPTIONAL=[]
if importlib.util.find_spec('torch') is None:
    OPTIONAL.append(('PyTorch SDPA','SKIP: torch not installed'))
    OPTIONAL.append(('official FA2 / FA3','SKIP: torch/CUDA unavailable'))
else:
    import torch
    import torch.nn.functional as F
    tq,tk,tv=[torch.tensor(a,dtype=torch.float64) for a in (q,k,v)]
    ty=F.scaled_dot_product_attention(tq,tk,tv,dropout_p=0.,is_causal=True)
    # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
    np.testing.assert_allclose(ty.numpy(),o,atol=1e-10,rtol=1e-10)
    OPTIONAL.append(('PyTorch SDPA CPU','PASS'))
    if not torch.cuda.is_available():
        OPTIONAL.append(('official FA2 / FA3','SKIP: CUDA unavailable'))
    else:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch.manual_seed(7)
        a,b,c=[torch.randn(1,128,2,64,device='cuda',dtype=torch.float16) for _ in range(3)]
        with sdpa_kernel(SDPBackend.MATH):
            # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
            ref=F.scaled_dot_product_attention(a.transpose(1,2).float(),b.transpose(1,2).float(),
                                              # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
                                              c.transpose(1,2).float(),is_causal=True).transpose(1,2)
        def verify_gpu(name,fn):
            out=fn(a,b,c,causal=True)
            if isinstance(out,tuple): out=out[0]
            torch.testing.assert_close(out.float(),ref,atol=4e-3,rtol=4e-3)
            for _ in range(3): fn(a,b,c,causal=True)
            torch.cuda.synchronize(); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats(); start.record()
            for _ in range(10): fn(a,b,c,causal=True)
            end.record(); torch.cuda.synchronize()
            OPTIONAL.append((name,f'PASS: {start.elapsed_time(end)/10:.4f} ms; peak allocated {torch.cuda.max_memory_allocated()} bytes'))
        print('GPU:',torch.cuda.get_device_name(),'torch:',torch.__version__,'CUDA:',torch.version.cuda)
        if importlib.util.find_spec('flash_attn') is None:
            OPTIONAL.append(('official FA2','SKIP: flash_attn not installed'))
        else:
            from flash_attn import flash_attn_func
            verify_gpu('official flash_attn (inspect installed version)',flash_attn_func)
        if torch.cuda.get_device_capability()[0]!=9:
            OPTIONAL.append(('official FA3 Hopper','SKIP: this exercise targets Hopper SM90'))
        elif importlib.util.find_spec('flash_attn_interface') is None:
            OPTIONAL.append(('official FA3 Hopper','SKIP: hopper interface not installed'))
        else:
            from flash_attn_interface import flash_attn_func as fa3_func
            verify_gpu('official FA3 Hopper',fa3_func)
for name,status in OPTIONAL: print(name,':',status)
```

</details>

<details>
<summary>验证结果汇总</summary>

```python
print('='*58)
print('REQUIRED CPU CHECKS:',len(TESTS),'PASS; failures: 0')
print('Code covers: Softmax, Linear, DeltaNet, GDN, KDA,')
print('FA1/2/3 educational models, PagedAttention, RadixAttention')
print('Optional checks:',OPTIONAL)
print('Run the complete reference script to reproduce.')
```

</details>

<!-- implementation:end -->

## 附录 B：运行官方 GPU 对照前，需要准备什么

本次验证环境已通过 PyTorch SDPA CPU 对照，但没有可用 CUDA，以下验证在此处记为 SKIP；不算入已通过的 CPU 检查。
本文参考程序的核心无需 GPU 依赖；CPU 运行只需安装 `numpy matplotlib`。
如需 GPU，按 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/) 配置对应 CUDA，再按 [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention) 与 [Hopper 子目录](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) 安装。

官方 FlashAttention Q/K/V 通常为 `[B,N,H,d]`，而本文为 `[B,H,N,d]`，必须 transpose。这里选择方形 causal、FP16、head_dim=64，简化跨版本约束。FA3 通常通过 `flash_attn_interface` 入口，需正确安装对应 hopper 包。
`torch.scaled_dot_product_attention` 的自动 backend 不保证等于特定 FA 版本；下文强制 math backend 作 oracle。
官方模块存在后若执行失败则抛出错误，不能将数值失败当作“没有 GPU”。

## 附录 C：从 3-pass 到 MLA / DSA 的独立小程序

这份程序独立运行，不依赖附录 A 的变量。它包含 18 项检查：三遍与两遍 softmax、融合 value、全遮蔽、手算矩阵、GQA 共享、MLA 吸收及解耦 RoPE、DSA 实际 gather 与全长 mask 基准，以及未来扰动不泄漏。索引器输入为随机特征，用于校验结构；没有声称加载 DeepSeek 权重或复现官方 FP8 性能。

<details>
<summary>展开逐步 softmax、GQA、MLA、DSA 全部代码（也可直接下载上面的程序）</summary>

```python
"""CPU teaching examples: safe/online softmax, GQA, absorbed MLA and gathered DSA.

Run: python attention-variants.py. Only NumPy is required.
This tests algebra, not GPU speed or a pretrained DeepSeek checkpoint.
"""
import numpy as np


def safe_softmax(x):
    """Normalize the last axis; a completely masked row returns zeros."""
    # axis=-1 selects keys. keepdims retains a size-1 axis for row broadcasting.
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    maximum = np.max(x, axis=-1, keepdims=True)
    # np.where(condition, yes, no) selects elementwise, without Python branching.
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
    baseline = np.where(np.isfinite(maximum), maximum, 0.0)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    weights = np.exp(x - baseline)  # exp(-inf)=0 for masked entries.
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    denominator = weights.sum(axis=-1, keepdims=True)
    # where=False leaves the explicitly initialized output at zero: no 0/0.
    # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    return np.divide(weights, denominator, out=np.zeros_like(weights),
                     where=denominator > 0)


def softmax_three_pass(x):
    """One finite, nonempty 1D score vector; no saved exponential array."""
    maximum = -np.inf
    for score in x:  # Pass 1: maximum.
        maximum = max(maximum, float(score))
    denominator = 0.0
    for score in x:  # Pass 2: denominator with the final common baseline.
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        denominator += np.exp(score - maximum)
    # NumPy：empty_like 只分配同形空间、不初始化数值；读取前必须写好每个元素。
    result = np.empty_like(x, dtype=float)  # Every entry is written below.
    for j, score in enumerate(x):  # Pass 3: output probabilities.
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        result[j] = np.exp(score - maximum) / denominator
    return result


def softmax_two_pass(x):
    """有限、非空的一维输入：先合并最大值/分母，再输出概率。mask 版本见 fused_value_scan。"""
    maximum, denominator = -np.inf, 0.0
    for score in x:
        next_maximum = max(maximum, float(score))
        # Old exponentials used the old maximum: rescale their accumulated sum.
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        correction = np.exp(maximum - next_maximum)
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        denominator = correction * denominator + np.exp(score - next_maximum)
        maximum = next_maximum
    # A second read is still needed when we explicitly require all probabilities.
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    return np.exp(x - maximum) / denominator


def fused_value_scan(scores, values):
    """Consume score/value pairs once, including masked scores and empty tiles."""
    maximum, denominator = -np.inf, 0.0
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    numerator = np.zeros(values.shape[-1])  # One accumulator per value channel.
    for score, value in zip(scores, values):
        # NumPy：isneginf 判断是否为负无穷；这种分数代表被 mask 的位置。
        if np.isneginf(score):  # No contribution; avoids -inf - (-inf).
            continue
        next_maximum = max(maximum, float(score))
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        correction = np.exp(maximum - next_maximum)
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        weight = np.exp(score - next_maximum)
        numerator = correction * numerator + weight * value
        denominator = correction * denominator + weight
        maximum = next_maximum
    return numerator / denominator if denominator > 0 else numerator


def gqa_without_repeat(q, k, v):
    """q [Hq,N,dk], k [Hkv,M,dk], v [Hkv,M,dv]; causal self-attention."""
    hq, n, dk = q.shape
    hkv, m, _ = k.shape
    assert n == m and hq % hkv == 0
    # arange generates positions [0,...,n-1]; None adds a row/column singleton.
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    keep = np.arange(m)[None, :] <= np.arange(n)[:, None]
    outputs = []
    for head in range(hq):
        group = head // (hq // hkv)  # Several query heads address one KV head.
        # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
        scores = q[head] @ k[group].T / np.sqrt(dk)
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        probabilities = safe_softmax(np.where(keep, scores, -np.inf))
        outputs.append(probabilities @ v[group])
    # stack inserts a new head axis, unlike concatenate along an existing axis.
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    return np.stack(outputs, axis=0)


def rope_rows(x, positions):
    """Rotate adjacent coordinate pairs of x [tokens, even_dim]."""
    assert x.shape[-1] % 2 == 0
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    frequencies = 10000.0 ** (-np.arange(0, x.shape[-1], 2) / x.shape[-1])
    # Outer product [tokens,1]*[1,pairs] gives one angle per token/pair.
    angles = positions[:, None] * frequencies[None, :]
    # NumPy：cos 逐元素计算余弦，输入角度单位为弧度。
    # NumPy：sin 逐元素计算正弦，与余弦一起构造二维旋转。
    cosine, sine = np.cos(angles), np.sin(angles)
    # NumPy：empty_like 只分配同形空间、不初始化数值；读取前必须写好每个元素。
    result = np.empty_like(x)
    even, odd = x[:, 0::2], x[:, 1::2]  # Slice every second coordinate.
    result[:, 0::2] = even * cosine - odd * sine
    result[:, 1::2] = even * sine + odd * cosine
    return result


def mla(qc, qr, c, kr, uk, uv, wo, absorbed=True, selections=None):
    """One batch, causal MLA; C and the queries are already projected/normalized.

    qc [H,N,dc], qr [H,N,dR], c [M,r], kr [M,dR],
    uk [H,r,dc], uv [H,r,dv], wo [H*dv,D]. qr/kr already include RoPE.
    selections optionally lists source positions for each query (shared by heads).
    """
    heads, n, dc = qc.shape
    dv, model_width = uv.shape[-1], wo.shape[-1]
    assert n == c.shape[0] and qr.shape[-1] == kr.shape[-1]
    # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
    scale = np.sqrt(dc + qr.shape[-1])  # Preserve ORIGINAL concatenated width.
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    output = np.zeros((n, model_width))
    probabilities = []
    for t in range(n):
        # With sparse selections, gather happens BEFORE the main dot products.
        # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
        indices = np.arange(t + 1) if selections is None else selections[t]
        # NumPy：unique 去重并排序；这里只用来断言所选历史位置没有重复。
        assert len(indices) > 0 and len(np.unique(indices)) == len(indices)
        # NumPy：all 检查布尔数组是否全部为真，返回真/假而非数值误差大小。
        assert np.all((indices >= 0) & (indices <= t))  # No future leakage.
        cs, rs = c[indices], kr[indices]  # Advanced integer indexing gathers rows.
        per_head = []
        for head in range(heads):
            if absorbed:
                q_latent = qc[head, t] @ uk[head].T  # [dc]@[dc,r] -> [r].
                content = q_latent @ cs.T  # [r]@[r,k] -> one score per selected key.
            else:
                expanded_keys = cs @ uk[head]  # [k,r]@[r,dc] -> [k,dc].
                content = qc[head, t] @ expanded_keys.T
            position = qr[head, t] @ rs.T
            prob = safe_softmax((content + position) / scale)
            if absorbed:
                latent_sum = prob @ cs  # Aggregate k histories to ONE [r] vector.
                head_output = latent_sum @ uv[head]  # Expand only the aggregate.
            else:
                expanded_values = cs @ uv[head]
                head_output = prob @ expanded_values
            # This row block is the portion of WO associated with this head.
            wo_head = wo[head * dv:(head + 1) * dv]
            output[t] += head_output @ wo_head
            per_head.append(prob)
        # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
        probabilities.append(np.stack(per_head))
    return output, probabilities


def dsa_indices(q_index, k_index, head_weights, top_k):
    """FP64 semantic indexer: qI [HI,N,dI], kI [N,dI], w [N,HI].

    This does not implement the official FP8 kernel or learned projections.
    Returned integer positions are shared across all MAIN MLA heads.
    """
    assert top_k > 0
    n = k_index.shape[0]
    selected = []
    for t in range(n):
        # [HI,dI]@[dI,t+1] -> one score row for each indexer head.
        dots = q_index[:, t, :] @ k_index[:t + 1].T
        # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
        positive_dots = np.maximum(dots, 0.0)  # ReLU; weights themselves may be negative.
        # [HI,1] broadcasts over candidate positions; sum axis=0 removes HI.
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        scores = (head_weights[t, :, None] * positive_dots).sum(axis=0)
        count = min(top_k, t + 1)
        # argsort returns integer LOCATIONS, not sorted score values.
        # Stable full sort is an easy CPU oracle; production uses efficient top-k.
        # NumPy：argsort 返回排序后的整数位置；不返回分数本身，负号可将升序变为降序。
        by_score = np.argsort(-scores, kind='stable')[:count]
        # NumPy：sort 返回按数值升序排列的新数组；对选中位置排序仅调整一致的历史读取顺序。
        selected.append(np.sort(by_score))  # Temporal order helps inspect the gather.
    return selected


def run_checks():
    rng = np.random.default_rng(20260908)  # Local generator; no global state changes.
    passed = []
    def close(name, actual, expected):
        # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        passed.append(name)

    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    for scores in [np.array([0., 1., 2.]), np.array([1000., -1000., 999.])]:
        close('three-pass stable', softmax_three_pass(scores), safe_softmax(scores))
        close('two-pass stable', softmax_two_pass(scores), safe_softmax(scores))
        # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
        vals = rng.normal(size=(3, 4))
        close('fused value scan', fused_value_scan(scores, vals), safe_softmax(scores) @ vals)
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
    for scores in [np.array([-np.inf, 0., 2.]), np.full(3, -np.inf)]:
        # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
        vals = rng.normal(size=(3, 4))
        close('masked first/all scores', fused_value_scan(scores, vals), safe_softmax(scores) @ vals)
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    close('weighted sum hand example', np.array([.25, .75]) @ np.array([[1., 2.], [5., 6.]]), [4., 5.])
    # NumPy：outer(a,b) 生成 a_i*b_j 的二维表，两个向量轴都保留，是外积而非点积。
    memory = np.outer([1., 2.], [3., 1., 2.]) + np.outer([2., 1.], [1., 4., 2.])
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    close('linear state hand example', np.array([1., 1.]) @ memory / 6., [2., 2.5, 2.])

    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    gq = rng.normal(size=(4, 5, 3)); gk = rng.normal(size=(2, 5, 3)); gv = rng.normal(size=(2, 5, 2))
    shared = gqa_without_repeat(gq, gk, gv)
    # NumPy：repeat 沿指定轴重复元素/切片，确实会复制；这里只用于验证共享 KV 的基准。
    copied = gqa_without_repeat(gq, np.repeat(gk, 2, axis=0), np.repeat(gv, 2, axis=0))
    close('GQA shared addressing vs repeated oracle', shared, copied)

    # Deliberately unequal dc=4, r=3, dv=2: catches hidden square-shape mistakes.
    heads, n, dc, r, dv, dr, dim = 2, 7, 4, 3, 2, 2, 5
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    qc = rng.normal(size=(heads, n, dc)); c = rng.normal(size=(n, r))
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    qr = np.stack([rope_rows(rng.normal(size=(n, dr)), np.arange(n)) for _ in range(heads)])
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    kr = rope_rows(rng.normal(size=(n, dr)), np.arange(n))
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    uk = rng.normal(size=(heads, r, dc)); uv = rng.normal(size=(heads, r, dv))
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    wo = rng.normal(size=(heads * dv, dim))
    args = (qc, qr, c, kr, uk, uv, wo)
    dense, _ = mla(*args, absorbed=False)
    close('MLA expanded vs absorbed with decoupled RoPE', mla(*args)[0], dense)
    close('value and WO weight absorption', (c @ uv[0]) @ wo[:dv], c @ (uv[0] @ wo[:dv]))

    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    iq = rng.normal(size=(3, n, 4)); ik = rng.normal(size=(n, 4)); iw = rng.normal(size=(n, 3))
    selections = dsa_indices(iq, ik, iw, top_k=3)
    sparse, _ = mla(*args, selections=selections)
    # Independent full-score-then-mask oracle; intentionally does NOT save work.
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    reference = np.zeros_like(sparse)
    for t in range(n):
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        mask = np.zeros(n, dtype=bool); mask[selections[t]] = True
        for head in range(heads):
            # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
            logits = (qc[head, t] @ (c @ uk[head]).T + qr[head, t] @ kr.T) / np.sqrt(dc + dr)
            # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
            prob = safe_softmax(np.where(mask, logits, -np.inf))
            reference[t] += (prob @ (c @ uv[head])) @ wo[head * dv:(head + 1) * dv]
    close('DSA gathered computation vs full masked oracle', sparse, reference)
    all_keys = dsa_indices(iq, ik, iw, top_k=n)
    close('DSA selecting full prefix reduces to dense MLA', mla(*args, selections=all_keys)[0], dense)
    # Perturb future cache and indexer keys; earlier selections/outputs must match.
    future_c, future_ik, future_kr = c.copy(), ik.copy(), kr.copy()
    future_c[4:] += 100.; future_ik[4:] -= 100.; future_kr[4:] += 30.
    future_selections = dsa_indices(iq, future_ik, iw, top_k=3)
    changed = mla(qc, qr, future_c, future_kr, uk, uv, wo, selections=future_selections)[0]
    close('DSA no future leakage', changed[:4], sparse[:4])
    # A negative head weight makes even a positive ReLU dot contribute negatively.
    # NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    signed = dsa_indices(np.ones((1, 2, 2)), np.array([[2., 2.], [1., 1.]]), -np.ones((2, 1)), 1)
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    close('signed indexer head weight', signed[1], np.array([1]))
    # Known top-k history slots used in the explanatory diagram.
    # NumPy：sort 返回按数值升序排列的新数组；对选中位置排序仅调整一致的历史读取顺序。
    # NumPy：argsort 返回排序后的整数位置；不返回分数本身，负号可将升序变为降序。
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    close('diagram top-k indices', np.sort(np.argsort(-np.array([8, 2, 9, 1, 7]))[:3]), [0, 2, 4])
    print(f'VARIANT CHECKS: {len(passed)} PASS; failures: 0')
    return passed


if __name__ == '__main__':
    run_checks()
```

</details>

## 参考资料


| 内容 | 一手来源 |
|---|---|
| 视频图解参考 | [Jia-Bin Huang：How Attention Got So Efficient](https://youtu.be/Y-o545eYjXM)；章节时间由提问所附信息定位 |
| GQA / MQA | [GQA](https://arxiv.org/abs/2305.13245)、[MQA](https://arxiv.org/abs/1911.02150) |
| MLA / 解耦 RoPE | [DeepSeek-V2](https://arxiv.org/abs/2405.04434)、[V3](https://arxiv.org/abs/2412.19437) |
| DSA | [Exp 官方代码](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)、[V3.2 §2.1](https://arxiv.org/html/2512.02556v1) |
| Online normalizer | [Milakov & Gimelshein, 2018](https://arxiv.org/abs/1805.02867) |
| NumPy API | [广播](https://numpy.org/doc/stable/user/basics.broadcasting.html)、[einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)、[divide](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) |
| Transformer / softmax | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) |
| Linear attention | [Katharopoulos et al., ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html) |
| DeltaNet 并行训练 | [Yang et al., 2024](https://arxiv.org/abs/2406.06484)；[作者解释](https://sustcsonglin.github.io/blog/2024/deltanet-1/) |
| Gated DeltaNet | [论文](https://arxiv.org/abs/2412.06464)；[NVlabs 官方实现](https://github.com/NVlabs/GatedDeltaNet) |
| FlashAttention 1 | [论文](https://arxiv.org/abs/2205.14135) |
| FlashAttention 2 | [论文](https://arxiv.org/abs/2307.08691) |
| FlashAttention 3 | [论文](https://arxiv.org/abs/2407.08608)；[作者说明](https://tridao.me/blog/2024/flash3/) |
| 官方 Flash kernels | [Dao-AILab 仓库](https://github.com/Dao-AILab/flash-attention) |
| PagedAttention | [Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180) |
| RadixAttention / SGLang | [论文](https://arxiv.org/abs/2312.07104)；[LMSYS 作者文章](https://www.lmsys.org/blog/2024-01-17-sglang/) |
| KDA / Kimi Linear | [报告 §3、§4 及附录](https://arxiv.org/html/2510.26692v1)；[官方项目](https://github.com/MoonshotAI/Kimi-Linear) |

本 文章中的图与参考代码为教学构建；论文报告的加速倍数未作为本机测量结果。所有“PASS”只针对此文具体代码路径和测试输入，不替代生产性能或模型质量评估。
