# Attention 笔记修订：来源核验与数学审读

核验日期：2026-09-08。审读对象：`src/content/blog/attention-from-softmax-to-kda.md` 修订前版本。本文记录来源、边界和可执行的改进建议，不代表已观看完整视频或已运行代码测试。

## 视频证据边界

直接打开 [Jia-Bin Huang 的视频页面](https://www.youtube.com/watch?v=Y-o545eYjXM) 获得标题 **How Attention Got So Efficient [GQA/MLA/DSA]**，页面正文未返回字幕、描述或讲解逐字稿。2025-11-26 发布日期及 00:00–27:44 章节表来自用户提供的信息，本次未独立取得视频元数据或字幕验证这些细节。

用户提供的图片确实可以作为“query 行、key 列、因果三角形、权重、value、output 分阶段呈现”的视觉参考。依据用户章节表，视频主线是 KV cache、MQA、GQA、MLA、RoPE 和 DSA；不能把本文 Linear attention 或 FlashAttention 推导标成该视频的逐段讲解。适当表述是：**借鉴用户参考图的图解颗粒度，数学定义分别核对原始论文**。

## 已核对的一手来源

| 来源 | 支撑的范围 | 笔记修订建议 |
|---|---|---|
| [Linear Transformers，§3.2–3.4，式 4–12、18–20](https://arxiv.org/html/2006.16236v3) | 非负特征核、结合律、ELU+1、因果前缀状态、归一化状态 | 区分“改变核”与“同一核的等价重排”；明确先更新再读出包含当前 token。 |
| [Online normalizer calculation for softmax，Algorithm 2–3](https://arxiv.org/html/1805.02867v1) | 三遍 safe softmax，在线更新最大值与分母 | 三遍依次求 max、指数和、最终概率；online normalizer 合并前两遍，输出全部概率仍需第二遍。 |
| [FlashAttention，Algorithm 1、Appendix B.3](https://arxiv.org/html/2205.14135v2) | QK/PV 分块、统计量重标定、mask、dropout、重计算 | 融合 value 累加后可以直接产出 attention 输出；保持二次 pair 计算量，不保存完整概率矩阵。Dropout 作用于 softmax 后的权重，保留值乘 1/(1-p)。 |
| [FlashAttention-2 作者说明](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2) | 减少非 matmul FLOPs、沿序列并行、warp 工作划分 | FA2 改进执行组织，不能描述为变成线性 attention。 |
| [DeltaNet 论文](https://arxiv.org/abs/2406.06484) 与 [Kimi Linear §2–4、Appendix B–C](https://arxiv.org/html/2510.26692v1) | Delta 梯度步、KDA 对角门顺序、chunk 三角系统、层结构 | 本文 key-major 状态约定与 Kimi 论文一致。必须先衰减，再以衰减后的状态计算预测与残差。 |
| [NumPy divide 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) | `out`、`where`、广播 | `where=False` 的元素保留 `out` 原值；因此必须提供初始化为零的 `out`，而不是依靠未初始化数组。 |

## 数学审读结论与修订点

1. **Linear 主公式正确，但 epsilon 需要说明。** 对正核，分母是 query 对所有可见 key 的总权重。理想公式不加 epsilon 时归一化权重和为 1；添加 epsilon 后为 `den/(den+epsilon)`，因此是数值保护后的微小修改。显式二次参考实现和状态实现必须使用相同 epsilon 才能比较。以上最后一项是从分式直接得到的代数推论。

2. **不能跨 softmax 使用结合律。** `(Φ(Q)Φ(K)ᵀ)V = Φ(Q)(Φ(K)ᵀV)` 正确；`softmax(QKᵀ)V = Q(KᵀV)` 不成立。建议图中分别标出 `[N,N]` token–token 权重与 `[r,dv]` feature–value 状态，展示一次外积如何写入状态、query 如何读出，以及 `z` 分母在哪里计算。

3. **全遮蔽行是约定，而非普通 softmax。** `[-∞,-∞]` 没有正常概率分布，直接减最大值会出现 `-∞-(-∞)`。现有 `safe_m` 与带 `out/where` 的除法可实现零输出约定（有限输入、mask 使用负无穷的范围内）。需要解释“无可见 key”与“训练 dropout 恰好丢掉全部连接”是两种原因；后者沿用 dropout 定义，不对剩余权重重新归一化。

4. **online softmax 递推正确，但不变量有条件。** 原文“a/l 始终是正确输出”应改成“已有可见 key、l>0 时 a/l 是正确输出；l=0 时采用零输出约定”。新最大值改变指数基准时，分子和分母必须同时乘 `exp(old_m-new_m)`。建议用两个不等长/不同最大值的块演算，展示不能把块内 softmax 输出直接相加。

5. **safe softmax 的 pass 计数需要指定对象。** 三遍是对一条 logits 向量的算法级遍历，并非任意 NumPy 表达式恰好执行三个 HBM pass。若缓存所有指数会改变读写位置和存储开销。融合 PV 的算法是只需要最终加权和，不再要求逐个输出概率；不能说任意独立 softmax 都能无缓存地一遍输出最终概率。

6. **Delta、KDA 与 chunk 主公式未发现代数错误。** 对 `S∈R^(dk×dv)`，转移为 `(I−βkkᵀ)D`；`D(I−βkkᵀ)` 一般不同。`L_ij=β_i k_iᵀDiag(g_{j+1:i})k_j` 使用严格下三角，读取系数 `E` 包含对角；`(I+L)U=Diag(β)(V−K_gS0)` 的符号、衰减区间和块尾写回均一致。累计门比值形式只在相关门非零时成立；直接区间乘积路径支持零门。以上结合论文定义逐项代数核对。

7. **KDA 缩放应明确为教学选择。** 论文主定义输出为 `Sᵀq`，Appendix C 的具体伪代码额外执行 `q *= dk**-0.5`。本文算子按主定义不缩放本身没有错误，但比较官方 kernel 时必须保持相同 `scale`，不能默认为数值直接相等。

8. **投影反向的 batch 归约需写清。** 原文 `dW_Q=XᵀdQ_merged` 可以理解为单 batch 或已合并 batch/token 的二维矩阵；对正文三维 `[B,N,D]`，应写 `dW_Q=Σ_b X_bᵀdQ_b`，或先 reshape 成 `[BN,D]`。`X.T` 在 NumPy 三维数组中会逆序所有轴，不能作为这个矩阵转置的直接代码。这是维度和链式法则核对所得。

9. **运算量建议从单输出元素数起。** `[m,k]@[k,n]` 每个输出含 k 次乘法和 k−1 次加法，总计 `mn(2k−1)≈2mnk` FLOPs。对单头，QKᵀ 和 PV 分别约 `2N²dk`、`2N²dv`；对标准等宽多头的 Q/K/V/O 投影合计约 `8BND²`。这些为本文约定下的逐项计数；mask 在完整 GEMM 后执行并不节省上三角乘法。

## 建议验算点

最有区分力的例子是：全遮蔽 query；只含遮蔽位置的首 tile；后 tile 最大值增大；dropout 后权重和不等于 1；显式特征核对状态式；改变未来 V 不影响此前 causal 输出；非对易 D 与 kkᵀ；chunk 内零门；batched 投影梯度的求和。本文仅提出审读建议，未把建议当作测试通过记录。

## 增补：GQA、MLA、DSA 的教学推导依据

本节响应后续增加的 GQA、MLA、DSA 请求。继续沿用用户提供的 09:42–27:44 章节顺序作为组织参考；未新取得视频字幕。[Exp 官方发布页](https://api-docs.deepseek.com/news/news250929/) 链接的 PDF 在本次浏览接口中未能提取，故 DSA 公式以 [DeepSeek-V3.2 正式报告 §2.1](https://arxiv.org/html/2512.02556v1) 和 [Exp 官方推理代码](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py) 交叉核对。正式报告明确其架构与 Exp 相同，但报告本身发表于视频所标日期之后，不能据此宣称这些文字出自视频。

### GQA：head 数改变在哪里

[GQA 原论文 §2](https://arxiv.org/html/2305.13245v3) 定义每组 query heads 共享一组 K/V；单组是 MQA，每个 query 独立一组则是 MHA。组内 K/V 投影权重取均值并继续预训练是论文的 checkpoint 转换方法，不是对任意 MHA 做保持输出不变的重排。

以下是依据定义整理的行向量/shape 推导。零基 query head 编号 h，`g=Hq/Hkv`，所属 KV 组 `u(h)=floor(h/g)`：

```text
Q: [B,Hq,N,dk]       K: [B,Hkv,M,dk]       V: [B,Hkv,M,dv]
P_h = softmax(Q_h K_u(h)^T / sqrt(dk) + mask) : [B,N,M]
O_h = P_h V_u(h) : [B,N,dv]
```

例：8 个 Q heads、2 个 KV heads，Q0–Q3 连到 K0/V0，Q4–Q7 连到 K1/V1；每个 Q head 仍独立生成自己的 P。KV 元素量从 `BNHq(dk+dv)` 变成 `BNHkv(dk+dv)`，而混合计算仍按 Hq 计数。图应让“存两组”与“读出八个不同结果”同时可见。

### MLA：从展开 K/V 到直接读取 latent

基础定义来自 [DeepSeek-V2 §2.1.2–2.1.3](https://arxiv.org/html/2405.04434v5)；[V3 §2.1.1](https://arxiv.org/html/2412.19437v2) 进一步明确只缓存 KV latent 与 RoPE key。下面统一改写成行向量，符号与论文列向量权重互为转置。

| 量 | 形状 | 作用 |
|---|---|---|
| X | `[N,D]` | 一个层的 token 表示 |
| C | `[N,r]` | 所有头共用的 KV latent |
| U_i^K、U_i^V | `[r,dc]`、`[r,dv]` | 第 i 头的 K/V 升维矩阵 |
| q_ti^C | `[1,dc]` | 第 t 个 token、第 i 头的内容 query |
| q_ti^R、k_s^R | `[1,dR]`、`[1,dR]` | 旋转后的 query 位置部分、跨头共用 key 位置部分 |
| W_i^O | `[dv,D]` | Wo 对应第 i 头的行块 |

先忽略 RoPE，`C=X W_DKV`，`K_i=C U_i^K`，`V_i=C U_i^V`。真实模型 latent 后有 RMSNorm；把规范化后的结果记作 C 即可。**不能跨越 RMSNorm，把输入端所有矩阵随意相乘成单个线性矩阵。**

下面等式完全来自矩阵结合律，可在小数值例子中逐项验算：

```text
内容分数：q_ti^C (C U_i^K)^T
        = (q_ti^C (U_i^K)^T) C^T
        = qbar_ti C^T
shape:   [1,dc] @ [dc,N] → [1,N]
重排后:  ([1,dc] @ [dc,r]) @ [r,N] → [1,N]

每头输出：p_ti (C U_i^V) = (p_ti C) U_i^V = z_ti U_i^V
shape:   ([1,N] @ [N,r]) @ [r,dv] → [1,dv]

最终输出：Concat_i(z_ti U_i^V) Wo
        = Σ_i z_ti (U_i^V W_i^O)
shape:   Σ_i [1,r] @ ([r,dv] @ [dv,D]) → [1,D]
```

因此每头的 key 升维挪到当前 query 侧，value 升维挪到加权汇总之后，历史无需逐 token 展开成全部头的 K/V。不同头的概率 p 不同，所以 z_ti 仍不同；不能先把所有头合成同一个 z。若 query 由 `c_t^Q U_i^Q` 生成，可预组合 `U_i^Q (U_i^K)^T`，但 query latent 前的 RMSNorm 仍需保留。

### 解耦 RoPE：为什么必须另外留一小条 key

把行向量旋转约定为 `q R_t`、`k R_s`。若直接旋转内容 K，打分出现 `q R_t R_s^T (U_i^K)^T c_s^T`；中间矩阵随历史位置 s 改变，无法合成一个适用于全部 s 的固定 query 投影。矩阵不可交换，移动括号不能越过位置矩阵。该困难和解耦方案见 [DeepSeek-V2 §2.1.3](https://arxiv.org/html/2405.04434v5)。

把内容、位置两路拼接后的点积展开，得到可直接画图的两条支路：

```text
a_tis = (qbar_ti c_s^T + q_ti^R (k_s^R)^T) / sqrt(dc+dR)
       内容 latent 点积       位置点积
p_ti  = softmax_s(a_tis + causal_mask)
z_ti  = p_ti C
```

吸收后 qbar 的宽度是 r，**缩放仍是 `sqrt(dc+dR)`，不能误改成 `sqrt(r+dR)`**，因为这里只重排了原始点积。每层每 token 缓存 `r+dR` 个元素；缓存随 N 线性增长。query latent 不需要作为历史 KV 留存。RoPE key 来自输入 X 的独立投影，所有主 attention heads 共享；query 的 RoPE 投影仍分头。上述形状与 cache 定义亦由 [V3 的蓝框公式](https://arxiv.org/html/2412.19437v2) 支撑。

### DSA：先廉价挑 token，再在同一组 token 上算主 attention

报告定义与训练方案见 [DeepSeek-V3.2 §2.1](https://arxiv.org/html/2512.02556v1)：

```text
I_ts = Σ_h w_th^I ReLU(q_th^I · k_s^I)
S_t  = TopKIndices({I_ts : s ≤ t})
```

对每个 query token，索引器所有 heads 汇总成一个分数向量；一个选择集合 S_t 供该 query 的全部主 MLA heads 共享。主 attention 在 S_t 内计算自己的 logits、softmax 和 value 加权和，索引器分数不替代这些权重。训练先保留 dense attention，仅训练 indexer；随后使用稀疏主 attention。teacher 为主 attention 概率跨头相加、沿 key 位置归一化的分布 p；loss 方向为 `KL(stopgrad(p) || softmax(I))`。稀疏阶段双方仅在 S_t 归一化。Indexer 输入 detach，indexer 仅受该 KL，主模型仅受 LM loss。主 attention 部分为 O(Nk)，indexer 仍 O(N²)。

以下代码细节核对 [Exp model.py 的 Indexer 类](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L401)：query 来自主网的归一化 query latent；key 是输入 X 的线性投影加 LayerNorm；head 权重来自 X 的线性投影，未加正值约束。Q/K 的部分通道先 RoPE，再共同作归一化 Hadamard，随后 FP8 量化；index key 与量化 scale 要缓存。索引器不读取当前主 attention 输出或 V，主 attention 因此可以在选择之后计算。官方简明参考前向仍计算全长主 logits 再 mask，不应把该参考代码的时间称作已兑现稀疏节省。

依据上述定义可作以下教学推论：

- `ReLU(dot)` 非负不代表总 I 非负，因为 w 可以为负。I 也不是概率；TopK 无需先 softmax，训练 KL 才使用 softmax(I)。
- 前缀不足 k 个可见 token 时保留全部可见位置；若固定形状 TopK 包含了 `-∞` 的 future 槽位，下游仍必须因果遮蔽。每个位置 t 的可见数量不同，不能只对整段长度取 `min(k,N)` 就取消因果 mask。
- teacher 指 softmax 后的概率，不是可能有负数的原始 QK logits。若各头概率均规范化，跨头求和再 L1 归一化相当于按头平均。
- 稀疏 teacher 由当前稀疏主网在 S_t 上的概率产生；不需要为每一步额外构造全长 dense teacher。TopK 的离散选择不靠主 LM loss 的常规导数训练；独立 KL 提供 indexer 的监督。
- 同一个正交 Hadamard H 满足 `(qH)(kH)^T=qk^T`，这是量化前的精确代数事实。量化后只有近似，排名可能改变；不能保证任意输入的误差都下降。Hadamard 放在 RoPE 之后，不能未加证明就交换两者。
- DSA 依然保留可寻址历史 latent 和 index keys，属于动态稀疏注意力，不是固定大小 recurrent 状态的 Linear attention。

建议图解顺序：八个 Q 连到两组 KV → 共享 latent 展开多个 K/V → 把 UK、UV 移到 query/汇总后 → 内容与 RoPE 两条分数支路相加 → 小 indexer 的多个分数行经 ReLU、带符号 head 加权、TopK → 全部主 heads 共用被选列 → dense/sparse 两阶段训练箭头与 stop-gradient。这个顺序是本文的教学设计，不作为已验证的视频逐帧还原。
