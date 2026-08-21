# 公式、形状、资源与调试速查

## 符号

| 符号 | 含义 |
|---|---|
| $B$ | microbatch sequences/GPU |
| $T$ | sequence/context length |
| $V$ | vocabulary size |
| $D$ | `d_model` |
| $H,H_{kv}$ | query heads、KV heads |
| $D_h=D/H$ | head dimension |
| $F$ | FFN hidden dimension |
| $L$ | number of layers |
| $N$ | parameters（注明 total/non-embedding/active） |
| $D_{tok}$ | total training tokens |
| $P$ | world size |

## 语言模型

$$p(x_{1:T})=\prod_t p(x_t|x_{<t})$$

$$\mathcal L=-\frac1M\sum_m\log p(y_m|x_m),\qquad PPL=e^{\mathcal L}$$

稳定交叉熵：

$$\mathrm{CE}(z,y)=m+\log\sum_j e^{z_j-m}-z_y,\ m=\max_jz_j.$$

## Transformer shape

```text
ids              [B,T]
x                [B,T,D]
q/k/v            [B,H,T,Dh]
scores/probs     [B,H,T,T]
attn out         [B,H,T,Dh] -> [B,T,D]
ffn hidden       [B,T,F]
logits           [B,T,V]
labels/mask      [B,T]
```

$$S=QK^T/\sqrt{D_h}+M,\quad P=softmax(S),\quad O=PV$$

$$RMSNorm(x)=g\odot x/\sqrt{mean(x^2)+\epsilon}$$

$$SwiGLU(x)=W_2(SiLU(W_gx)\odot W_vx)$$

单层参数（dense SwiGLU，无 bias）：

$$N_{layer}\approx4D^2+3DF$$

embedding/head：共享约 $VD$，不共享约 $2VD$。

## 训练

Global batch tokens：

$$B_{tok}=B\times T\times grad\_accum\times P.$$

AdamW：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
$$\theta\leftarrow(1-\eta\lambda)\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)$$

Global norm clipping：

$$g\leftarrow g\min(1,c/(\|g\|_2+\epsilon)).$$

Cosine after warmup：

$$\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos(\pi u)).$$

## FLOPs 与显存

Linear `[M,K]@[K,N]` forward 约 $2MKN$ FLOPs；训练 forward+dX+dW 约 forward 的 3 倍。

Dense LM 粗略训练：

$$C\approx6ND_{tok}.$$

Arithmetic intensity / roofline：

$$I=FLOPs/bytes,\quad Perf\le\min(Peak, I\cdot BW).$$

Adam 混合精度 model states 常见量级约 16 bytes/parameter（实现不同）；另加 activations、临时张量、通信 bucket、碎片。

KV cache：

$$M_{KV}=2LT H_{kv}D_h\cdot bytes\cdot batch.$$

## FlashAttention online softmax

合入新 block：

$$m'=\max(m,m_b)$$
$$\ell'=e^{m-m'}\ell+\sum_{j\in b}e^{s_j-m'}$$
$$o'=e^{m-m'}o+\sum_{j\in b}e^{s_j-m'}v_j$$
$$O=o/\ell$$

Backward：

$$dV=P^TdO,\ dP=dOV^T$$
$$dS=P\odot(dP-rowSum(P\odot dP))$$
$$dQ=dSK/\sqrt{D_h},\ dK=dS^TQ/\sqrt{D_h}$$

## 分布式

DDP gradient：

$$g=\frac1P\sum_rg_r$$

Ring all-reduce 每 rank bytes 约：

$$2\frac{P-1}{P}M.$$

通信模型：

$$T_{comm}\approx\alpha\#messages+\beta\#bytes.$$

强扩展效率：

$$E=\frac{T_1}{PT_P}.$$

## Scaling

$$L(N,D)=E+A/N^\alpha+B/D^\beta$$

IsoFLOP：每 $C_i$ 令 $D=C_i/(6N)$，找 U 型 $L$ 的 $N_{opt}$，再拟合：

$$N_{opt}=aC^\alpha,\qquad D_{opt}=bC^\beta.$$

必须报告外推比、CI、holdout error 和系统可行性。

## 数据去重

$$J(A,B)=|A\cap B|/|A\cup B|$$

MinHash：

$$P[h_{min}(A)=h_{min}(B)]=J(A,B).$$

LSH candidate probability：

$$1-(1-s^r)^b.$$

## DPO

令

$$\Delta_\theta=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x),$$
$$\Delta_{ref}=\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x),$$

则：

$$L_{DPO}=-\log\sigma(\beta(\Delta_\theta-\Delta_{ref})).$$

只累计 response token；reference 冻结；sum/mean 长度处理会改变目标。

## Policy gradient / GRPO

$$\nabla J=E[R\nabla\log\pi(y|x)]$$

$$A_i=(R_i-\bar R)/(std(R)+\epsilon)$$

$$r_t=\exp(\log\pi_\theta-\log\pi_{old})$$

$$L^{clip}=E[\min(rA,clip(r,1-\epsilon,1+\epsilon)A)]$$

注意 group 全同 reward、std/length normalization、prompt/padding mask、old-policy 版本和 off-policy ESS。

## 第一响应式调试

| 症状 | 第一批检查 |
|---|---|
| loss 不降 | target shift、mask、参数更新、LR、数据 |
| NaN | softmax/logsumexp、norm fp32、LR、fp16 scale |
| OOM | 参数/optimizer、activation、临时 score、allocator |
| GPU 慢 | 同步计时、预热、shape、HBM、launch、data wait |
| 多卡挂 | collective 顺序、某 rank exception、async wait |
| 多卡不等价 | sampler、loss reduction、grad avg、seed |
| DPO 退化 | template/mask、ref、beta、长度/KL |
| GRPO reward 涨但准确不涨 | verifier/parser hack、format、length |

## 核心术语

- **base model**：主要经预训练的续写模型。
- **instruction/chat model**：经 SFT/偏好/RL 适配交互。
- **pretraining / mid-training / post-training**：广覆盖训练 / 聚焦高质域的中期训练 / 下游行为与 RL。
- **MFU**：模型理论 FLOPs 相对硬件峰值；不等于整体成本效率。
- **on-policy**：数据由当前/足够接近当前 policy 产生。
- **verifiable reward**：可通过答案、测试、执行器客观检查的 reward。
- **data lineage**：最终样本到来源/每步变换的可追溯链。
- **decontamination**：移除/标注与评测集重合的训练数据。
- **prefill/decode**：并行处理 prompt / 串行生成新 token。
- **GQA/MQA/MLA**：逐级减少/压缩 KV cache 的 attention 设计。
