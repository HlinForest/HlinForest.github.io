---
title: "Attention 的计算与优化：从 Softmax、FlashAttention 到 KDA"
description: "从注意力矩阵的计算与存储成本出发，串起 FlashAttention 1/2/3、PagedAttention、RadixAttention，以及 Linear、DeltaNet、Gated DeltaNet 与 Kimi Delta Attention 的推导、实现和验证。"
publishedAt: "2026-09-07"
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

> **实现与验证范围**：参考代码依赖 NumPy 和 Matplotlib，可在 CPU 运行。配套实现的 129 项 CPU 检查已通过。FA1/2/3 的 Python 代码是算法与调度教学模型，真实 CUDA 异步指令与硬件性能仍需官方内核验证；当前验证环境没有 PyTorch/CUDA，对应项目明确跳过。KDA 在这里指 **Kimi Delta Attention**。

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

全遮蔽行本来没有概率分布，本文约定输出和梯度均为 0。padding 的 query 若要清零，也需对该 query 的整行作 mask。
Dropout 是训练时对 P 的额外随机运算；本文核心函数取 dropout=0，不把带 dropout 的训练结果与无 dropout 推理结果混比。

来源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。以下维度展开与实现为教学推导。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-01.png" alt="图 1：Q 与 K 的 key 维收缩，产生 token×token 权重，再与 V 收缩得到输出。图中展示一个 batch/head 的前四个 token。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 1：Q 与 K 的 key 维收缩，产生 token×token 权重，再与 V 收缩得到输出。图中展示一个 batch/head 的前四个 token。</figcaption>
</figure>

### 2.1 一段可以作为基准的实现

这里将 causal mask 和外部 mask 合并，再执行数值稳定的 softmax。完整代码中的 `softmax_masked` 负责全遮蔽行的零输出约定。所有优化实现都先与这一基准比较。

```python
def softmax_masked(a, keep=None):
    if keep is not None:
        a = np.where(keep, a, -np.inf)
    m = np.max(a, axis=-1, keepdims=True)
    safe_m = np.where(np.isfinite(m), m, 0.)
    e = np.exp(a - safe_m)
    den = e.sum(-1, keepdims=True)
    return np.divide(e, den, out=np.zeros_like(e), where=den > 0)

def softmax_attention(q, k, v, causal=False, keep=None, q_start=0):
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
    assert q.shape[-1] == k.shape[-1] and k.shape[-2] == v.shape[-2]
    if causal:
        cm = position_mask(q.shape[-2], k.shape[-2], q_start)
        keep = cm if keep is None else (keep & cm)
    p = softmax_masked((q @ k.swapaxes(-1,-2)) / math.sqrt(q.shape[-1]), keep)
    return p @ v, p
```

### 2.2 训练与生成：同一公式，两种负载

训练或 prefill 时，一次输入整段序列。单头需要两个主要矩阵乘法：$QK^T$ 和 $PV$，总计算量为 $\Theta(N^2(d_k+d_v))$。推理生成时，每次只新增一个 query，但它仍需读取全部可见历史，因此单步成本是 $\Theta(N(d_k+d_v))$。

缓存历史 K/V 可以避免反复计算旧 token 的投影，却没有取消新 query 对历史的读取。也正因为如此，prefill 常有较大的矩阵乘法并行空间，decode 则更容易受 KV 读取带宽和请求 batch 大小影响。

我们现在有了两个直接问题：能不能保留同一个公式，却不把 N×N 中间矩阵完整写入显存？能不能保留全部 KV，却更合理地存放和复用它们？先看第一个问题。

## 3. FlashAttention：不保存整张概率矩阵，怎样保证结果正确

标准 attention 中 $N^2$ 概率矩阵占用显存且来回读写。Flash 把 score 划为 `R×T` tile，在快速存储中消费后丢弃。**计算仍为二次，改变的是 IO 和执行组织。**

每个 query 行保留三元组 $(m,l,a)$：当前最大 logit、以 m 为基准的指数和、未归一化输出向量。
初始 $(-\infty,0,0)$。新 tile logits 为 X：
$$
m'=\max(m,\max_jX_j),\quad c=\exp(m-m'),\quad P'=\exp(X-m'),
$$
$$
l'=cl+\sum_jP'_j,\quad a'=ca+P'V,\quad O=a'/l'.
$$

旧的指数基准变了，旧 l 和 a **必须一起**乘 c。不可以对每个 tile 做 softmax 后直接相加。
全遮蔽 tile 不贡献指数和；初始全遮蔽时要绕开 $-\infty-(-\infty)$。
等价地，两个 disjoint KV 分区的统计量也可合并，这支持 split-K decode 和分页遍历。

### 3.1 用三个统计量完成全局归一化

online softmax 最值得记住的不是循环顺序，而是一个不变量：`a / l` 始终是截至当前已处理键集合的正确输出。新块带来更大的最大值时，只要旧分子和分母一起重标定，就可以继续累积。

```python
def online_update(m,l,a,x,v):
    new_m=np.maximum(m,np.max(x,axis=-1))
    safe=np.where(np.isfinite(new_m),new_m,0.)
    rescale=np.exp(m-safe)
    weights=np.exp(x-safe[...,None])
    new_l=rescale*l+weights.sum(-1)
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

softmax 的 $\exp(q^Tk)$ 不能直接拆成 $q^Tk$。定义新核 $\kappa(q,k)=\phi(q)^T\phi(k)$ 后，非因果情形：

$$
O_i=\frac{\phi(q_i)^T\underbrace{\sum_j\phi(k_j)v_j^T}_{S=\Phi(K)^TV}}
{\phi(q_i)^T\underbrace{\sum_j\phi(k_j)}_z+\epsilon}.
$$

`[N,r] @ ([r,N] @ [N,dv])` 先得到 `[r,dv]`，避开 `[N,N]`。本例选 $\phi(x)=\operatorname{ELU}(x)+1$，输出正值；并不宣称其等于 softmax 核。分母避免总权重随序列长任意变化，但无法消除压缩状态中的关联干扰。

**因果不能直接使用全序列的 S**，否则看到未来。要递推：
$$
S_t=S_{t-1}+\phi(k_t)v_t^T,\quad z_t=z_{t-1}+\phi(k_t),\quad
O_t=\phi(q_t)^TS_t/(\phi(q_t)^Tz_t+\epsilon).
$$

这里更新后读出，包含当前 token。若先读出后写入，会变为严格过去 attention。分块时，块内作 masked kernel，块间保留 S,z；两者相加再统一归一化，不能先各自归一化再相加。
来源：[Linear Transformers](https://proceedings.mlr.press/v119/katharopoulos20a.html)。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-06.png" alt="图 6：非因果 Linear attention 的分子先计算 K 的特征转置与 V 的乘积，再由 Q 读取。因果情况必须改用前缀状态。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 6：非因果 Linear attention 的分子先计算 K 的特征转置与 V 的乘积，再由 Q 读取。因果情况必须改用前缀状态。</figcaption>
</figure>

### 7.1 固定状态带来的收益，也定义了它的边界

显式 KV cache 会为每个历史 token 保留一份独立表示；Linear attention 把这些关联叠加进同一矩阵。每增加一个 token，状态大小保持不变，但不同关联可能竞争同一片表示空间。

这里的“linear”是关于序列长度的复杂度：当特征维 r 和 value 维固定时，混合计算随 N 线性增长。它不表示计算只是一个普通线性层，也不保证与 softmax 模型具有相同的表达能力。

这时一个很自然的问题出现了：同一个 key 的 value 发生变化时，是否还应该把新的关联直接叠加进去？

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
def delta_recurrent(q,k,v,beta,alpha=None,state=None,return_tape=False):
    assert q.shape==k.shape and q.shape[:-1]==v.shape[:-1]==beta.shape
    if alpha is None: alpha=np.ones_like(k)
    alpha=np.broadcast_to(alpha,k.shape)
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])) if state is None else state.copy()
    outs=[]; tape=[]
    for t in range(q.shape[-2]):
        qt,kt,vt,bt,at=q[...,t,:],k[...,t,:],v[...,t,:],beta[...,t],alpha[...,t,:]
        previous=s
        decayed=at[..., :,None]*s
        prediction=np.einsum('...k,...kv->...v',kt,decayed)
        error=vt-prediction
        s=decayed+(bt[...,None]*kt)[..., :,None]*error[...,None,:]
        outs.append(np.einsum('...k,...kv->...v',qt,s))
        if return_tape: tape.append((previous,decayed,error,s))
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
-\sum_{j<i}\underbrace{k_i^T\operatorname{Diag}(g_{j+1:i})k_j}_{F_{ij}}u_j\right].
$$
令 $K_g[i]=g_{1:i}\odot k_i$、$Q_g[i]=g_{1:i}\odot q_i$，
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

这是一个单位下三角系统，依次前代，不需 `inv`。也可以先分别求解
$U_0=(I+L)^{-1}\operatorname{Diag}(\beta)V$ 与
$W=(I+L)^{-1}\operatorname{Diag}(\beta)K_g$，再 $U=U_0-WS_0$；这连接到 WY/UT 表达，避免把每个 token 的 `[dk,dk]` 转移矩阵连乘。

**两种实现**：

- 结合律/GEMM 路径：$G_i=g_{1:i}$，$F=(G\odot K)(K/G)^T$，$E=(G\odot Q)(K/G)^T$ 后加三角 mask。形成块内 Gram 矩阵可用矩阵乘法。
- 稳定教学路径：直接计算区间衰减 $g_{j+1:i}$，无需先形成极小的 G 再求倒数，支持精确零门。它保留 Python 内层循环，正确性好但速度不代表生产 kernel。

GEMM 路径只适用于非零且不过小的块内累计门；实作通常在 log 域算区间差、二级分块，并混用 FP32。不能用随意 clamp 分母冒充数学等价。
分块顺序仍有状态依赖；块内矩阵操作提升 GPU 吞吐，不表示所有时间步都没有依赖。

<figure>
  <img src="/images/notes/attention-from-softmax-to-kda/figure-08.png" alt="图 8：一个 chunk 的累计衰减 G、严格下三角更新系数 L、因果读取系数 E 与传到块尾的 key。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 8：一个 chunk 的累计衰减 G、严格下三角更新系数 L、因果读取系数 E 与传到块尾的 key。</figcaption>
</figure>

```python
def unit_lower_solve(l,r):
    # 求 (I + 严格下三角 l) x = r；批量前代 O(C^2 * dv)。
    out=np.zeros_like(r)
    for i in range(r.shape[-2]):
        out[...,i,:]=r[...,i,:]-np.einsum('...j,...jv->...v',l[...,i,:i],out[...,:i,:])
    return out
```

### 10.1 结合律允许并行，但并行本身不保证便宜

令 $A_t=(I-\beta k k^T)D_t$、$B_t=\beta kv^T$，则 $S_t=A_tS_{t-1}+B_t$。
两个变换按时间复合：
$$
(A_2,B_2)\circ(A_1,B_1)=(A_2A_1,A_2B_1+B_2).
$$
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
mask 的梯度为 0，因为对应 P 为 0。投影层进一步有 $dW_Q=X^TdQ_{merged}$，$dX_Q=dQ_{merged}W_Q^T$，K/V 分支对 dX 求和。
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

本函数返回的是已归一化 q/k、直接 alpha/beta 的梯度。
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
  <img src="/images/notes/attention-from-softmax-to-kda/figure-10.png" alt="图 10：此环境中的 NumPy 参考实现 CPU 中位耗时，只用于验证和记录，不代表生产 GPU 内核的速度排名。" loading="lazy" style="display:block;width:100%;height:auto;" />
  <figcaption>图 10：此环境中的 NumPy 参考实现 CPU 中位耗时，只用于验证和记录，不代表生产 GPU 内核的速度排名。</figcaption>
</figure>

### 14.1 哪些错误需要独立的测试

覆盖非整块长度、非方形维度、B/H 广播、prefill/decode、因果性、初始状态、门退化、数值稳定、前向/反向、页共享与树分裂。
未来扰动测试只修改未来的 K/V（或输入 X），要求早期输出不变。不同公式之间不要求输出相同，只有同公式的实现方式才做等价测试。
空序列不是核心函数支持的输入（递推 stack/max 无定义），请求侧应直接跳过空段；本文输入均为正长度。

### 14.2 本文代码的验证结果

| 检查项 | 结果与含义 |
|---|---|
| 129 项必需 CPU 检查 | 全部通过；包括前向等价、有限差分、分块状态、因果性、缓存生命周期 |
| Linear 与 softmax 输出 | 不要求一致；它们使用不同核函数 |
| DeltaNet / GDN / KDA | 同一公式的递推与分块一致；门满足特定条件时验证退化关系 |
| 官方 FlashAttention GPU 内核 | 当前环境无 PyTorch/CUDA，跳过；不能据此声称已验证硬件性能 |

## 15. 带着问题回看这几条路线

遇到一个新的 attention 名称时，可以先追问四件事：它保留了什么数学定义，历史保存在哪里，新 token 需要读取什么，以及优化成本转移到了哪里。

FlashAttention 保留全部 pair 的计算，通过分块、重计算和硬件调度减少中间数据往返。PagedAttention 保留显式 KV，通过逻辑到物理的映射降低碎片和复制成本。RadixAttention 保留前缀的已计算结果，让新请求从可以复用的位置继续。

Linear attention 开始改变记忆的表示方式：不再逐项保存全部历史，而是维护有限的矩阵状态。DeltaNet 让写入由预测残差驱动；Gated DeltaNet 允许整头遗忘；KDA 把遗忘细化到 key 通道。随后，chunkwise 算法又把这些递推整理成硬件更擅长的矩阵运算。

理解这些方法的联系之后，优化选择才有依据：先确定负载与语义约束，再选择公式、内核和缓存策略。一个模型可以同时需要其中多种方法，而每一种方法都需要与它所声称解决的问题对应的测试。

## 附录 A：完整实现与自动测试

以下代码保留为完整参考程序，按展开顺序拼接到一个 `.py` 文件即可运行；依赖为 `numpy`、`matplotlib`。每个代码块包含必要的函数或测试，不能只运行后半部分而跳过初始化。

代码顺序沿用已验证的依赖顺序：先定义数学基准，再定义递推与分块，随后运行 Flash 和 cache 测试。正文则按理解问题的顺序组织，两者目的不同。

图解随文章一同提供。运行代码时会重新生成相应图表。

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
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    TESTS.append(name)
def require(name, condition):
    assert bool(condition), name
    TESTS.append(name)
def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))
def softplus(x):
    return np.logaddexp(0., x)
def heatmaps(items, title='', cmap='viridis'):
    fig, axes = plt.subplots(1, len(items), figsize=(3.1*len(items), 3.0), squeeze=False)
    for ax, (label, a) in zip(axes[0], items):
        a = np.asarray(a)
        ax.imshow(a, cmap=cmap, aspect='auto')
        ax.set_title(label + '  ' + str(a.shape), fontsize=10)
        if a.size <= 64:
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
def softmax_masked(a, keep=None):
    if keep is not None:
        a = np.where(keep, a, -np.inf)
    m = np.max(a, axis=-1, keepdims=True)
    safe_m = np.where(np.isfinite(m), m, 0.)
    e = np.exp(a - safe_m)
    den = e.sum(-1, keepdims=True)
    return np.divide(e, den, out=np.zeros_like(e), where=den > 0)

def position_mask(n, m, q_start=0, k_start=0):
    return (np.arange(m) + k_start)[None,:] <= (np.arange(n) + q_start)[:,None]

def softmax_attention(q, k, v, causal=False, keep=None, q_start=0):
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
    assert q.shape[-1] == k.shape[-1] and k.shape[-2] == v.shape[-2]
    if causal:
        cm = position_mask(q.shape[-2], k.shape[-2], q_start)
        keep = cm if keep is None else (keep & cm)
    p = softmax_masked((q @ k.swapaxes(-1,-2)) / math.sqrt(q.shape[-1]), keep)
    return p @ v, p

def split_heads(x, h):
    b,n,w = x.shape
    assert w % h == 0
    return x.reshape(b,n,h,w//h).transpose(0,2,1,3)

def merge_heads(x):
    b,h,n,d = x.shape
    return x.transpose(0,2,1,3).reshape(b,n,h*d)

def gqa(q, k, v, **kwargs):
    # 教学版复制；生产 kernel 根据 query head 寻址共享的 KV head。
    assert q.shape[1] % k.shape[1] == 0
    group = q.shape[1] // k.shape[1]
    return softmax_attention(q, np.repeat(k,group,axis=1),
                            np.repeat(v,group,axis=1), **kwargs)[0]

B,H,N,DK,DV = 2,2,9,4,3
q = rng.normal(size=(B,H,N,DK)); k = rng.normal(size=q.shape)
v = rng.normal(size=(B,H,N,DV))
o,p = softmax_attention(q,k,v,causal=True)
heatmaps([('Q',q[0,0,:4]), ('K transpose',k[0,0,:4].T),
          ('P: causal rows',p[0,0,:4,:4]), ('V',v[0,0,:4]), ('O',o[0,0,:4])],
         'Q @ K.T -> row softmax -> P @ V')
check('softmax rows sum 1',p.sum(-1),np.ones((B,H,N)))
require('causal upper triangle zero', np.all(np.triu(p,1)==0))
check('head split merge', merge_heads(split_heads(np.arange(120).reshape(2,5,12),3)),
      np.arange(120).reshape(2,5,12))
zero,_ = softmax_attention(q,k,v,keep=np.zeros((N,N),dtype=bool))
check('all masked rows zero',zero,np.zeros_like(zero))
# Prefill 与逐 token decode 必须完全一致。
decoded = np.concatenate([softmax_attention(q[:,:,t:t+1],k[:,:,:t+1],v[:,:,:t+1])[0]
                          for t in range(N)],axis=2)
check('softmax prefill equals decode',decoded,o)
check('offset mask decode',softmax_attention(q[:,:,-1:],k,v,causal=True,q_start=N-1)[0],o[:,:,-1:])
```

</details>

<details>
<summary>1.1 反向传播：softmax Jacobian 不必显式构造</summary>

```python
def softmax_backward(q,k,v,p,go):
    dv = p.swapaxes(-1,-2) @ go
    dp = go @ v.swapaxes(-1,-2)
    da = p * (dp - (p*dp).sum(-1,keepdims=True))
    scale = q.shape[-1] ** -0.5
    return (da @ k)*scale, (da.swapaxes(-1,-2) @ q)*scale, dv

def finite_difference(fn, x, eps=1e-6):
    x = x.copy(); grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old=x[idx]; x[idx]=old+eps; pos=fn(x)
        x[idx]=old-eps; neg=fn(x); x[idx]=old
        grad[idx]=(pos-neg)/(2*eps)
    return grad

sq,sk,sv = [rng.normal(size=s) for s in [(1,1,3,2),(1,1,3,2),(1,1,3,2)]]
so,sp=softmax_attention(sq,sk,sv,causal=True); go=rng.normal(size=so.shape)
grads=softmax_backward(sq,sk,sv,sp,go)
for j in range(3):
    arr=[sq,sk,sv]
    def loss(z):
        args=arr.copy(); args[j]=z
        return np.sum(softmax_attention(*args,causal=True)[0]*go)
    check('softmax finite difference '+str(j),grads[j],finite_difference(loss,arr[j]),atol=2e-8,rtol=2e-6)
```

</details>

<details>
<summary>2. Linear attention：改变核函数后才可换结合顺序</summary>

```python
def phi(x):
    # 避免 np.where 同时计算 exp(大正数)。
    return np.where(x>=0, x+1, np.exp(np.minimum(x,0)))

def linear_dense(q,k,v,causal=True,eps=1e-12):
    a=phi(q) @ phi(k).swapaxes(-1,-2)
    if causal: a=a*position_mask(q.shape[-2],k.shape[-2])
    return (a@v)/(a.sum(-1,keepdims=True)+eps)

def linear_recurrent(q,k,v,state=None,eps=1e-12):
    q,k=phi(q),phi(k)
    shape=q.shape[:-2]; r=q.shape[-1]; dv=v.shape[-1]
    s,z=(np.zeros(shape+(r,dv)),np.zeros(shape+(r,))) if state is None else (state[0].copy(),state[1].copy())
    out=[]
    for t in range(q.shape[-2]):
        kt,vt,qt=k[...,t,:],v[...,t,:],q[...,t,:]
        s=s+kt[..., :,None]*vt[...,None,:]; z=z+kt
        out.append(np.einsum('...r,...rv->...v',qt,s)/(np.sum(qt*z,-1,keepdims=True)+eps))
    return np.stack(out,-2),(s,z)

def linear_chunk(q,k,v,chunk=4,eps=1e-12):
    assert chunk>0
    q,k=phi(q),phi(k)
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])); z=np.zeros(q.shape[:-2]+(q.shape[-1],))
    outs=[]
    for a in range(0,q.shape[-2],chunk):
        qc,kc,vc=q[...,a:a+chunk,:],k[...,a:a+chunk,:],v[...,a:a+chunk,:]
        weights=(qc@kc.swapaxes(-1,-2))*np.tri(qc.shape[-2])
        numerator=qc@s+weights@vc
        denominator=np.einsum('...tr,...r->...t',qc,z)[...,None]+weights.sum(-1,keepdims=True)
        outs.append(numerator/(denominator+eps))
        s=s+kc.swapaxes(-1,-2)@vc; z=z+kc.sum(-2)
    return np.concatenate(outs,-2),(s,z)

lr,ls=linear_recurrent(q,k,v)
check('linear dense recurrence',lr,linear_dense(q,k,v))
for c in [1,4,16]:
    lc,lstate=linear_chunk(q,k,v,c)
    check('linear chunk '+str(c),lc,lr); check('linear chunk state '+str(c),lstate[0],ls[0])
qa,ka=phi(q[0,0,:4]),phi(k[0,0,:4]); va=v[0,0,:4]
heatmaps([('Phi K.T',ka.T),('V',va),('S = K.T @ V',ka.T@va),('Phi Q',qa),('Q @ S',qa@(ka.T@va))],
         'Noncausal numerator: associativity removes N x N')
print('Linear 与 softmax 的最大差异（预期非零）:',np.max(np.abs(lr-o)))
```

</details>

<details>
<summary>2.1 非因果实现与 Linear attention 的完整反向</summary>

```python
def linear_noncausal(q,k,v,eps=1e-12):
    fq,fk=phi(q),phi(k)
    state=fk.swapaxes(-1,-2)@v; z=fk.sum(-2)
    return (fq@state)/(np.einsum('...tr,...r->...t',fq,z)[...,None]+eps)

def linear_backward(q,k,v,go,eps=1e-12):
    fq,fk=phi(q),phi(k)
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])); z=np.zeros(q.shape[:-2]+(q.shape[-1],))
    tape=[]
    for t in range(q.shape[-2]):
        s=s+fk[...,t,:,None]*v[...,t,None,:]; z=z+fk[...,t,:]
        tape.append((s,z))
    ds=np.zeros_like(s); dz=np.zeros_like(z)
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    for t in reversed(range(q.shape[-2])):
        s,z=tape[t]; qt=fq[...,t,:]; kt=fk[...,t,:]; vt=v[...,t,:]
        num=np.einsum('...r,...rv->...v',qt,s)
        den=(qt*z).sum(-1,keepdims=True)+eps
        dn=go[...,t,:]/den; dh=-(go[...,t,:]*num).sum(-1,keepdims=True)/(den*den)
        dq[...,t,:]=np.einsum('...rv,...v->...r',s,dn)+dh*z
        ds=ds+qt[..., :,None]*dn[...,None,:]; dz=dz+dh*qt
        dk[...,t,:]=np.einsum('...rv,...v->...r',ds,vt)+dz
        dv[...,t,:]=np.einsum('...rv,...r->...v',ds,kt)
    return dq*np.where(q>=0,1,np.exp(np.minimum(q,0))),dk*np.where(k>=0,1,np.exp(np.minimum(k,0))),dv
check('noncausal linear associativity',linear_noncausal(q,k,v),linear_dense(q,k,v,causal=False))
small=[rng.normal(size=(1,1,3,2)) for _ in range(3)]; upstream=rng.normal(size=(1,1,3,2))
lg=linear_backward(*small,upstream)
for j in range(3):
    def loss(a):
        args=small.copy(); args[j]=a
        return np.sum(linear_recurrent(*args)[0]*upstream)
    check('Linear finite diff '+str(j),lg[j],finite_difference(loss,small[j]),atol=3e-8,rtol=2e-6)
```

</details>

<details>
<summary>3. DeltaNet：把加法记忆改为回归误差修正</summary>

```python
def delta_recurrent(q,k,v,beta,alpha=None,state=None,return_tape=False):
    assert q.shape==k.shape and q.shape[:-1]==v.shape[:-1]==beta.shape
    if alpha is None: alpha=np.ones_like(k)
    alpha=np.broadcast_to(alpha,k.shape)
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])) if state is None else state.copy()
    outs=[]; tape=[]
    for t in range(q.shape[-2]):
        qt,kt,vt,bt,at=q[...,t,:],k[...,t,:],v[...,t,:],beta[...,t],alpha[...,t,:]
        previous=s
        decayed=at[..., :,None]*s
        prediction=np.einsum('...k,...kv->...v',kt,decayed)
        error=vt-prediction
        s=decayed+(bt[...,None]*kt)[..., :,None]*error[...,None,:]
        outs.append(np.einsum('...k,...kv->...v',qt,s))
        if return_tape: tape.append((previous,decayed,error,s))
    result=(np.stack(outs,-2),s)
    return result+(tape,) if return_tape else result

qn,kn=normalize(q),normalize(k)
beta=sigmoid(rng.normal(size=(B,H,N)))
scalar_alpha=np.exp(-softplus(rng.normal(size=(B,H,N,1)))*.15)
vector_alpha=np.exp(-softplus(rng.normal(size=(B,H,N,DK)))*.15)
delta,_=delta_recurrent(qn,kn,v,beta)
gated,_=delta_recurrent(qn,kn,v,beta,scalar_alpha)
kda,kda_state=delta_recurrent(qn,kn,v,beta,vector_alpha)
check('scalar KDA reduces to GDN',delta_recurrent(qn,kn,v,beta,np.broadcast_to(scalar_alpha,kn.shape))[0],gated)
check('unit decay reduces to DeltaNet',delta_recurrent(qn,kn,v,beta,np.ones_like(kn))[0],delta)
# 一个两通道的数值例子：可以直接观察行衰减、残差与外积。
s0=np.array([[2.,1.],[3.,4.]])
kt=normalize(np.array([1.,2.])); vt=np.array([2.,-1.]); at=np.array([.9,.2]); bt=.8
sb=at[:,None]*s0; pred=kt@sb; err=vt-pred; update=bt*kt[:,None]*err[None,:]
heatmaps([('S previous',s0),('D @ S',sb),('pred ; target',np.stack([pred,vt])),
          ('beta k outer error',update),('S new',sb+update)], 'Decay -> predict -> residual -> rank-one write',cmap='coolwarm')
# beta=1 单位 key 的精确覆写。
onek=normalize(rng.normal(size=(1,1,1,3))); onev=rng.normal(size=(1,1,1,2))
initial=rng.normal(size=(1,1,3,2))
check('unit key overwrite',delta_recurrent(onek,onek,onev,np.ones((1,1,1)),state=initial)[0],onev)
```

</details>

<details>
<summary>5. 从逐 token 递推到 chunkwise 三角系统</summary>

```python
def unit_lower_solve(l,r):
    # 求 (I + 严格下三角 l) x = r；批量前代 O(C^2 * dv)。
    out=np.zeros_like(r)
    for i in range(r.shape[-2]):
        out[...,i,:]=r[...,i,:]-np.einsum('...j,...jv->...v',l[...,i,:i],out[...,:i,:])
    return out

def chunk_factors(q,k,alpha,method):
    c,dk=k.shape[-2:]
    g=np.cumprod(alpha,axis=-2)
    if method=='gemm':
        if np.any(g<1e-100):
            raise ValueError('累计门过小或为零，请用 stable 路径或更小 chunk')
        kr=k/g
        f=(k*g)@kr.swapaxes(-1,-2)
        e=(q*g)@kr.swapaxes(-1,-2)
        kend=(g[...,-1:,:]/g)*k
    elif method=='stable':
        f=np.zeros(k.shape[:-2]+(c,c)); e=np.zeros_like(f)
        kend=np.zeros_like(k)
        for j in range(c):
            decay=np.ones(k.shape[:-2]+(dk,))
            for i in range(j,c):
                if i>j: decay=decay*alpha[...,i,:]
                weighted=decay*k[...,j,:]
                f[...,i,j]=np.sum(k[...,i,:]*weighted,-1)
                e[...,i,j]=np.sum(q[...,i,:]*weighted,-1)
            kend[...,j,:]=decay*k[...,j,:]
    else: raise ValueError(method)
    return g, np.tril(f,-1),np.tril(e),kend

def delta_chunk(q,k,v,beta,alpha=None,chunk=4,state=None,method='stable'):
    assert chunk>0
    if alpha is None: alpha=np.ones_like(k)
    alpha=np.broadcast_to(alpha,k.shape)
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])) if state is None else state.copy()
    outs=[]
    for start in range(0,q.shape[-2],chunk):
        sl=slice(start,start+chunk)
        qc,kc,vc,bc,ac=q[...,sl,:],k[...,sl,:],v[...,sl,:],beta[...,sl],alpha[...,sl,:]
        g,f,e,kend=chunk_factors(qc,kc,ac,method)
        l=bc[..., :,None]*f
        rhs=bc[..., :,None]*(vc-(kc*g)@s)
        u=unit_lower_solve(l,rhs)
        outs.append((qc*g)@s+e@u)
        s=g[...,-1,:,None]*s+kend.swapaxes(-1,-2)@u
    return np.concatenate(outs,-2),s

for label,al in [('delta',None),('gdn',scalar_alpha),('kda',vector_alpha)]:
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
def delta_affine_scan(q,k,v,beta,alpha,state=None):
    alpha=np.broadcast_to(alpha,k.shape)
    eye=np.eye(k.shape[-1])
    aa=(eye-beta[...,None,None]*k[..., :,None]*k[...,None,:])*alpha[...,None,:]
    bb=beta[...,None,None]*k[..., :,None]*v[...,None,:]
    n=q.shape[-2]; gap=1
    while gap<n:
        # 必须从旧数组读，不能在同一级中原地污染后续输入。
        na=aa.copy(); nb=bb.copy()
        na[...,gap:,:,:]=aa[...,gap:,:,:]@aa[...,:-gap,:,:]
        nb[...,gap:,:,:]=aa[...,gap:,:,:]@bb[...,:-gap,:,:]+bb[...,gap:,:,:]
        aa,bb=na,nb; gap*=2
    if state is not None: bb=bb+aa@state[...,None,:,:]
    return np.einsum('...tk,...tkv->...tv',q,bb),bb[...,-1,:,:]
check('affine scan oracle',delta_affine_scan(qn,kn,v,beta,vector_alpha)[0],kda)
```

</details>

<details>
<summary>5.2 Delta/GDN/KDA 手写反向：训练不能只写 forward</summary>

```python
def delta_backward(q,k,v,beta,alpha,tape,go,final_grad=None):
    alpha=np.broadcast_to(alpha,k.shape)
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    db=np.zeros_like(beta); da=np.zeros_like(alpha)
    h=np.zeros_like(tape[0][0]) if final_grad is None else final_grad.copy()
    for t in reversed(range(q.shape[-2])):
        prev,bar,err,st=tape[t]
        qt,kt,bt,at,gt=q[...,t,:],k[...,t,:],beta[...,t],alpha[...,t,:],go[...,t,:]
        dq[...,t,:]=np.einsum('...kv,...v->...k',st,gt)
        h=h+qt[..., :,None]*gt[...,None,:]
        db[...,t]=np.einsum('...kv,...k,...v->...',h,kt,err)
        dk[...,t,:]=bt[...,None]*np.einsum('...kv,...v->...k',h,err)
        de=bt[...,None]*np.einsum('...kv,...k->...v',h,kt)
        dv[...,t,:]=de
        dk[...,t,:]-=np.einsum('...kv,...v->...k',bar,de)
        dbar=h-kt[..., :,None]*de[...,None,:]
        da[...,t,:]=np.sum(dbar*prev,axis=-1)
        h=at[..., :,None]*dbar
    return dq,dk,dv,db,da,h

args=[normalize(rng.normal(size=(1,1,3,2))),normalize(rng.normal(size=(1,1,3,2))),
      rng.normal(size=(1,1,3,2)),rng.uniform(.2,.8,(1,1,3)),rng.uniform(.2,.9,(1,1,3,2))]
ini=rng.normal(size=(1,1,2,2)); outs,ss,tape=delta_recurrent(*args,state=ini,return_tape=True)
gout=rng.normal(size=outs.shape); gs=rng.normal(size=ss.shape)
analytic=delta_backward(*args,tape,gout,gs)
for j in range(6):
    vals=args+[ini]
    def loss(z):
        aa=vals.copy(); aa[j]=z
        yy,st=delta_recurrent(*aa[:5],state=aa[5])
        return np.sum(yy*gout)+np.sum(st*gs)
    check('KDA backward finite diff '+str(j),analytic[j],finite_difference(loss,vals[j]),atol=3e-8,rtol=2e-6)
# 在输入 beta logits 上做一次真正的梯度下降，证明反向可用于优化。
logits=np.zeros((1,1,3)); target=rng.normal(size=outs.shape)
def beta_loss(b):
    yy,_=delta_recurrent(args[0],args[1],args[2],sigmoid(b),args[4],state=ini)
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
def online_update(m,l,a,x,v):
    new_m=np.maximum(m,np.max(x,axis=-1))
    safe=np.where(np.isfinite(new_m),new_m,0.)
    rescale=np.exp(m-safe)
    weights=np.exp(x-safe[...,None])
    new_l=rescale*l+weights.sum(-1)
    new_a=rescale[...,None]*a+weights@v
    return new_m,new_l,new_a

def tile_scores(qc,kc,i,j,causal,keep,q_start=0):
    scores=qc@kc.swapaxes(-1,-2)/math.sqrt(qc.shape[-1])
    if causal:
        mask=position_mask(qc.shape[-2],kc.shape[-2],i+q_start,j)
        scores=np.where(mask,scores,-np.inf)
    if keep is not None:
        scores=np.where(keep[...,i:i+qc.shape[-2],j:j+kc.shape[-2]],scores,-np.inf)
    return scores

def flash1_model(q,k,v,rows=4,cols=3,causal=True,keep=None,q_start=0):
    assert rows>0 and cols>0
    m=np.full(q.shape[:-1],-np.inf); l=np.zeros_like(m)
    out=np.zeros(q.shape[:-1]+(v.shape[-1],))
    for j in range(0,k.shape[-2],cols):
        kc,vc=k[...,j:j+cols,:],v[...,j:j+cols,:]
        for i in range(0,q.shape[-2],rows):
            sl=slice(i,i+rows); qc=q[...,sl,:]
            x=tile_scores(qc,kc,i,j,causal,keep,q_start)
            mm,ll,aa=online_update(m[...,sl],l[...,sl],out[...,sl,:]*l[...,sl,None],x,vc)
            out[...,sl,:]=np.divide(aa,ll[...,None],out=np.zeros_like(aa),where=ll[...,None]>0)
            m[...,sl]=mm; l[...,sl]=ll
    return out

def flash2_model(q,k,v,rows=4,cols=3,causal=True,keep=None,q_start=0,return_lse=False):
    assert rows>0 and cols>0
    outputs=[]; logsum=[]
    for i in range(0,q.shape[-2],rows):
        qc=q[...,i:i+rows,:]
        m=np.full(qc.shape[:-1],-np.inf); l=np.zeros_like(m)
        a=np.zeros(qc.shape[:-1]+(v.shape[-1],))
        for j in range(0,k.shape[-2],cols):
            kc,vc=k[...,j:j+cols,:],v[...,j:j+cols,:]
            x=tile_scores(qc,kc,i,j,causal,keep,q_start)
            m,l,a=online_update(m,l,a,x,vc)
        outputs.append(np.divide(a,l[...,None],out=np.zeros_like(a),where=l[...,None]>0))
        logsum.append(m+np.log(np.where(l>0,l,1.)))
    result=np.concatenate(outputs,-2)
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
mask=np.tri(9); axes[0].imshow(mask,cmap='Blues',vmin=0,vmax=1)
for i in range(0,9,3):
    for j in range(0,9,4):
        axes[0].add_patch(Rectangle((j-.5,i-.5),min(4,9-j),3,fill=False,edgecolor='#d57525',linewidth=2))
for i in range(9):
    for j in range(9):
        axes[0].text(j,i,f'{i},{j}' if j<=i else 'x',fontsize=7,ha='center',va='center')
axes[0].set_title('Causal score tiles: R=3, T=4'); axes[0].set_xlabel('key position j'); axes[0].set_ylabel('query position i')
# 选后3行，确保多个 KV tile 都有真实贡献。
qt=q[0,0,6:9]; mm=np.full(3,-np.inf); ll=np.zeros(3); aa=np.zeros((3,DV)); history=[]
for j in range(0,9,4):
    scores=tile_scores(qt,k[0,0,j:j+4],6,j,True,None)
    mm,ll,aa=online_update(mm,ll,aa,scores,v[0,0,j:j+4])
    history.append(np.stack([mm,ll,aa[:,0]],-1))
history=np.concatenate(history,axis=0)
axes[1].imshow(history,cmap='coolwarm',aspect='auto')
axes[1].set_xticks([0,1,2],['m','l','a[value=0]'])
axes[1].set_yticks(range(9),[f'tile {j}, query {i}' for j in range(3) for i in range(6,9)])
for (i,j),val in np.ndenumerate(history): axes[1].text(j,i,f'{val:.3f}',ha='center',va='center',fontsize=8)
axes[1].set_title('Running online-softmax statistics'); plt.tight_layout(); plt.show()
check('illustrated tile output',aa/ll[:,None],o[0,0,6:9])
```

</details>

<details>
<summary>6.1 Flash 反向：用 LSE 重建 P，避免保存全 N×N</summary>

```python
def flash_backward(q,k,v,go,out,lse,rows=4,cols=3,causal=True,keep=None):
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    d=(go*out).sum(-1); scale=q.shape[-1]**-.5
    for i in range(0,q.shape[-2],rows):
        qi=q[...,i:i+rows,:]; gi=go[...,i:i+rows,:]
        ll=lse[...,i:i+rows]; safe=np.where(np.isfinite(ll),ll,0.)
        for j in range(0,k.shape[-2],cols):
            kj,vj=k[...,j:j+cols,:],v[...,j:j+cols,:]
            x=tile_scores(qi,kj,i,j,causal,keep)
            pp=np.exp(x-safe[...,None])
            dp=gi@vj.swapaxes(-1,-2)
            ds=pp*(dp-d[...,i:i+rows,None])
            dq[...,i:i+rows,:]+=scale*(ds@kj)
            dk[...,j:j+cols,:]+=scale*(ds.swapaxes(-1,-2)@qi)
            dv[...,j:j+cols,:]+=pp.swapaxes(-1,-2)@gi
    return dq,dk,dv
fo,fl=flash2_model(q,k,v,return_lse=True); gg=rng.normal(size=fo.shape)
fg=flash_backward(q,k,v,gg,fo,fl)
sg=softmax_backward(q,k,v,p,gg)
for j in range(3): check('Flash recomputed backward '+str(j),fg[j],sg[j])
```

</details>

<details>
<summary>7. FlashAttention-3：数学相同，硬件执行不同</summary>

```python
def flash3_pipeline_model(q,k,v,rows=4,cols=3,causal=True):
    outputs=[]; trace=[]
    for i in range(0,q.shape[-2],rows):
        qc=q[...,i:i+rows,:]; m=np.full(qc.shape[:-1],-np.inf)
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
        outputs.append(np.divide(a,l[...,None],out=np.zeros_like(a),where=l[...,None]>0))
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
    h=np.ones((1,1))
    while h.shape[0]<n: h=np.block([[h,h],[h,-h]])
    return h/np.sqrt(n)
def quantize_uniform(x):
    scale=np.maximum(np.max(np.abs(x),axis=(-2,-1),keepdims=True),1e-12)/127
    return np.clip(np.round(x/scale),-127,127)*scale
rr=hadamard(DK)*rng.choice([-1.,1.],size=(1,DK))
check('orthogonal transform invariant logits',(q@rr)@(k@rr).swapaxes(-1,-2),q@k.swapaxes(-1,-2))
original=q@k.swapaxes(-1,-2)
for label,qa,ka in [('plain',q,k),('rotated',q@rr,k@rr)]:
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
        self.k=np.zeros((pages,heads,page_size,dk))
        self.v=np.zeros((pages,heads,page_size,dv))
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
        assert set(self.free)==set(np.flatnonzero(self.refs==0))
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
            return np.zeros((self.pool.H,0,self.pool.k.shape[-1])),np.zeros((self.pool.H,0,self.pool.v.shape[-1]))
        return np.concatenate([x[0] for x in blocks],1),np.concatenate([x[1] for x in blocks],1)
    def decode(self,q):
        assert q.shape==(self.pool.H,self.pool.k.shape[-1]) and self.length>0
        m=np.full((self.pool.H,1),-np.inf); l=np.zeros_like(m)
        a=np.zeros((self.pool.H,1,self.pool.v.shape[-1]))
        for kk,vv in self.blocks():
            scores=q[:,None,:]@kk.swapaxes(-1,-2)/math.sqrt(q.shape[-1])
            m,l,a=online_update(m,l,a,scores,vv)
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
for _ in range(2): a.append(np.ones((1,2)),np.ones((1,2)))
b=a.fork(); b.append(np.zeros((1,2)),np.zeros((1,2)))
require('full-page fork no old-page copy',a.table[0]==b.table[0] and len(b.table)==2)
a.close(); b.close(); p2.audit()
# OOM 不损坏原始共享尾页。
p3=PagePool(1,1,2,2,2); a=PagedSequence(p3); a.append(np.ones((1,2)),np.ones((1,2))); b=a.fork()
try:
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
            np.testing.assert_allclose(child.kv[0][:n],k[pos:pos+n],atol=1e-10)
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
embedding=rng.normal(size=(30,dmodel))*.2
weights=[tuple(rng.normal(size=(dmodel,dmodel))*.2 for _ in range(4)) for _ in range(2)]
def toy_layer(x,w,prefix=None):
    wq,wk,wv,wo=w
    qq=split_heads((x@wq)[None],hh); kk=split_heads((x@wk)[None],hh); vv=split_heads((x@wv)[None],hh)
    length=0 if prefix is None else prefix[0].shape[0]
    if prefix is None: allk,allv=kk,vv
    else:
        pk,pv=prefix
        allk=np.concatenate([pk.transpose(1,0,2)[None],kk],axis=2)
        allv=np.concatenate([pv.transpose(1,0,2)[None],vv],axis=2)
    yy=softmax_attention(qq,allk,allv,causal=True,q_start=length)[0]
    result=x+merge_heads(yy)[0]@wo
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
def short_conv(x,w):
    # x [B,N,D], w [window,D], w[0] 是当前 token 的权重。
    out=np.zeros_like(x)
    for lag in range(min(len(w),x.shape[1])):
        out[:,lag:,:]+=x[:,:x.shape[1]-lag,:]*w[lag]
    return out

def silu(x): return x*sigmoid(x)

def make_kda_weights(dmodel,heads,dk,dv,rank=4,window=3,seed=7):
    rr=np.random.default_rng(seed)
    def w(a,b): return rr.normal(size=(a,b))/np.sqrt(a)
    return dict(wq=w(dmodel,heads*dk),wk=w(dmodel,heads*dk),wv=w(dmodel,heads*dv),
                cq=w(window,heads*dk),ck=w(window,heads*dk),cv=w(window,heads*dv),
                ad=w(dmodel,rank),au=w(rank,heads*dk),bl=w(dmodel,heads),
                gd=w(dmodel,rank),gu=w(rank,heads*dv),wo=w(heads*dv,dmodel),
                alog=np.full((1,heads,1,dk),-2.),dtbias=np.zeros((1,heads,1,dk)),
                norm_weight=np.ones((1,heads,1,dv)))

def kda_layer(x,w,heads,method='recurrent'):
    qq=normalize(split_heads(silu(short_conv(x@w['wq'],w['cq'])),heads))
    kk=normalize(split_heads(silu(short_conv(x@w['wk'],w['ck'])),heads))
    vv=split_heads(silu(short_conv(x@w['wv'],w['cv'])),heads)
    raw=split_heads((x@w['ad'])@w['au'],heads)
    log_alpha=-np.exp(w['alog'])*softplus(raw+w['dtbias'])
    alpha=np.exp(log_alpha)
    beta=sigmoid(x@w['bl']).transpose(0,2,1)
    if method=='recurrent': yy,_=delta_recurrent(qq,kk,vv,beta,alpha)
    elif method=='chunk': yy,_=delta_chunk(qq,kk,vv,beta,alpha,chunk=4)
    else: raise ValueError(method)
    yy=yy/np.sqrt(np.mean(yy*yy,axis=-1,keepdims=True)+1e-6)*w['norm_weight']
    gate=sigmoid(split_heads((x@w['gd'])@w['gu'],heads))
    return merge_heads(yy*gate)@w['wo']

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
lengths=np.array([32,64,128,256,512,1024,2048,4096])
kv_bytes=lengths*8*256*2
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
    return float(np.median(samples))
bench=[]
for nn in [32,64,128]:
    qq=normalize(rng.normal(size=(1,1,nn,8))); kk=normalize(rng.normal(size=qq.shape))
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
for label,al in [('Delta',np.ones_like(kn)),('GDN',np.broadcast_to(scalar_alpha,kn.shape)),('KDA',vector_alpha)]:
    whole,final=delta_recurrent(qn,kn,v,beta,al)
    left,state=delta_recurrent(qn[:,:,:4],kn[:,:,:4],v[:,:,:4],beta[:,:,:4],al[:,:,:4])
    right,state=delta_recurrent(qn[:,:,4:],kn[:,:,4:],v[:,:,4:],beta[:,:,4:],al[:,:,4:],state)
    check(label+' streaming split',np.concatenate([left,right],2),whole)
    check(label+' streaming final',state,final)
left,state=linear_recurrent(q[:,:,:4],k[:,:,:4],v[:,:,:4])
right,state=linear_recurrent(q[:,:,4:],k[:,:,4:],v[:,:,4:],state)
check('Linear streaming split',np.concatenate([left,right],2),lr)
# beta=0 时只有通道遗忘；从非零 S0 验证，避免零状态平凡通过。
sinit=rng.normal(size=(B,H,DK,DV)); bzero=np.zeros_like(beta)
_,sf=delta_recurrent(qn,kn,v,bzero,vector_alpha,sinit)
check('beta zero preserves pure decay',sf,np.prod(vector_alpha,axis=2)[..., :,None]*sinit)
# 不同 query/KV 长度和任意 mask 的 Flash 对照。
qr=q[:,:,:5]; kr=k[:,:,:7]; vr=v[:,:,:7]
keep=rng.random((B,1,5,7))>.3; keep[:,:,2,:]=False
for fn in [flash1_model,flash2_model]:
    check(fn.__name__+' rectangular',fn(qr,kr,vr,3,4,False,keep),softmax_attention(qr,kr,vr,keep=keep)[0])
# MQA/GQA 手动分组 oracle。
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
keys=np.array([[[[1.,0.],[1.,0.],[0.,1.]]]])
values=np.array([[[[1.,2.],[7.,8.],[3.,4.]]]])
_,memory=delta_recurrent(keys,keys,values,np.ones((1,1,3)))
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
    np.testing.assert_allclose(ty.numpy(),o,atol=1e-10,rtol=1e-10)
    OPTIONAL.append(('PyTorch SDPA CPU','PASS'))
    if not torch.cuda.is_available():
        OPTIONAL.append(('official FA2 / FA3','SKIP: CUDA unavailable'))
    else:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch.manual_seed(7)
        a,b,c=[torch.randn(1,128,2,64,device='cuda',dtype=torch.float16) for _ in range(3)]
        with sdpa_kernel(SDPBackend.MATH):
            ref=F.scaled_dot_product_attention(a.transpose(1,2).float(),b.transpose(1,2).float(),
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

当前验证环境没有 PyTorch/CUDA，以下验证在此处记为 SKIP；不算入已通过的 CPU 检查。
本文参考程序的核心无需 GPU 依赖；CPU 运行只需安装 `numpy matplotlib`。
如需 GPU，按 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/) 配置对应 CUDA，再按 [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention) 与 [Hopper 子目录](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) 安装。

官方 FlashAttention Q/K/V 通常为 `[B,N,H,d]`，而本文为 `[B,H,N,d]`，必须 transpose。这里选择方形 causal、FP16、head_dim=64，简化跨版本约束。FA3 通常通过 `flash_attn_interface` 入口，需正确安装对应 hopper 包。
`torch.scaled_dot_product_attention` 的自动 backend 不保证等于特定 FA 版本；下文强制 math backend 作 oracle。
官方模块存在后若执行失败则抛出错误，不能将数值失败当作“没有 GPU”。

## 参考资料


| 内容 | 一手来源 |
|---|---|
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
