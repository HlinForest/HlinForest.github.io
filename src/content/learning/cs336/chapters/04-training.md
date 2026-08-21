# 04 优化、训练循环与实验科学

对应官方 Lecture 3、Assignment 1 后半，以及 scaling 前置。

## 4.1 一个训练 step 的准确含义

给连续 token 数组，随机取起点 $s$ 和长度 $T+1$：

```text
input  = tokens[s : s+T]
target = tokens[s+1 : s+T+1]
```

模型输出 `[B,T,V]` logits，与 `[B,T]` targets 计算 token 平均交叉熵。训练顺序：

```text
sample batch -> forward -> loss / grad_accum
-> backward -> (repeat microbatches)
-> unscale if needed -> clip gradients
-> optimizer.step -> lr scheduler -> zero_grad
-> log/eval/checkpoint
```

gradient accumulation 的目标是模拟更大的 global batch。若每个 microbatch 的 loss 已取平均，应再除 accumulation steps；否则梯度会被放大。分布式框架还会对 worker 梯度求平均/和，必须确认约定。

## 4.2 SGD、Momentum 与 AdamW

### SGD

$$\theta_t=\theta_{t-1}-\eta g_t.$$

简单但一个全局学习率难适配不同尺度/稀疏度参数。

### Momentum

$$m_t=\beta m_{t-1}+(1-\beta)g_t,qquad \theta_t=\theta_{t-1}-\eta m_t.$$

它对持续方向累积、对震荡方向抵消，相当于梯度低通滤波。

### Adam

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,$$
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},$$
$$\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

偏置修正因为零初始化让早期矩估计偏小。`epsilon` 放在根号内外不是等价实现；checkpoint 要保存 step 和两阶状态。

### AdamW 为什么“解耦”

把 L2 惩罚加进 Adam 梯度会被自适应分母缩放，参数的衰减量依赖历史梯度。AdamW 独立做：

$$\theta_t\leftarrow(1-\eta\lambda)\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

通常不衰减 norm scale、bias、可能还有 embedding；必须显式构造参数组，而不是用名字模糊猜测。AdamW 的优势是优化行为和正则语义更清楚，不意味着 weight decay 越大越好。

## 4.3 Muon 与矩阵参数优化

Muon 类方法对二维 hidden-layer 权重的 momentum 更新做正交化/谱方向归一，再施加更新；embedding、norm、bias、输出头通常仍用 AdamW。直觉是 Adam 的逐元素预条件忽略矩阵几何，而正交化让各奇异方向更新更均衡。

它可能在固定 token/墙钟下提高样本效率，但要计入 Newton-Schulz 迭代、通信和超参数映射；不能仅比较“同 learning rate”。公开 Assignment 1 实现加入 Muon 是很好的扩展，但课程核心仍是先把 AdamW、预算和验证做对。

## 4.4 学习率 schedule

训练早期参数和 optimizer state 未稳定，大 LR 容易破坏表示，因此 warmup：

$$\eta_t=\eta_{max}\frac{t}{T_w},\quad t<T_w.$$

cosine decay：

$$\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right].$$

WSD（warmup-stable-decay）长时间保持稳定 LR，最后单独 decay，便于从同一稳定训练段切出不同 token budget 的 checkpoint，适合持续训练/多预算实验。schedule 的自变量最好是已见 token 或 optimizer step；换 world size/accumulation 后要保持含义一致。

### 调 learning rate 的正确顺序

1. 固定模型、数据顺序、global batch tokens、训练 token budget。
2. 对数尺度 sweep，例如相邻 2-3 倍。
3. 看早期 loss 下降速度、梯度/更新范数和稳定性。
4. 在合理区间细化；用验证 loss 而非训练 loss 选。
5. 规模变化后验证 LR 是否可迁移；不要盲用小模型最优值。

## 4.5 Batch size 与噪声

小 batch 梯度噪声大，step 多；大 batch 并行好、噪声小，但每 token 的统计收益最终递减。critical batch size 大致是继续增 batch 不再减少达到目标 loss 所需 token/step 的转折，并随 loss/训练阶段变化。

报告 batch 时只写“batch=32”不够：要写 sequence length、microbatch、accumulation、world size 和 global tokens。扩大 batch 常需调整 LR 和 warmup，但“线性缩放规则”不是普遍定律。

## 4.6 Gradient clipping

global norm：

$$\|g\|_2=\sqrt{\sum_p\sum_i g_{p,i}^2},\qquad
g\leftarrow g\cdot\min\left(1,\frac{c}{\|g\|_2+\epsilon}\right).$$

混合精度时先 unscale 再 clip；分片训练要得到真正 global norm。clipping 是稳定保险，不应长期每步把梯度砍小；若 clip rate 很高，应检查 LR、数据异常、loss reduction 和数值问题。

## 4.7 数据采样与文档边界

把文档用 EOT 拼成一维 token 流能高效随机切片。模型可跨 EOT 看到前文，但学习 EOT 是边界；有的训练会 reset attention/position，含义不同。关键是：

- target 永远是 input 后一 token；
- train/val 文档级切分，不能同一文档片段泄漏；
- memory-map 大数组，避免全载入 RAM；
- worker/进程使用独立且可复现 RNG；
- 分布式 sampler 不重复或漏数据，除非混合策略明确允许。

packing 多个短样本时，用 block-diagonal causal mask 防止跨样本 attention；同时 label mask 忽略 padding 和非目标 prompt token。

## 4.8 Checkpoint 是完整状态，不只是权重

可精确恢复至少保存：

- model state；
- optimizer state 与 step；
- scheduler state；
- grad scaler（fp16）；
- Python/NumPy/PyTorch CPU/CUDA RNG；
- dataloader/shuffle cursor 或可重建信息；
- 配置、代码版本、tokenizer/data hash；
- 已训练 tokens/steps 和 best metric。

恢复测试：连续训练 $K$ 步的结果，应与训练 $k$ 步→保存→新进程加载→再 $K-k$ 步在容许确定性范围内一致。只比较“能加载”远远不够。

大规模训练还要处理原子写、分片 checkpoint、异步保存和旧 checkpoint 保留策略。先写临时文件，完成后 rename，避免宕机留下看似有效的半文件。

## 4.9 生成：概率分布不是 argmax

给最后位置 logits：

- temperature：$p_i\propto\exp(z_i/\tau)$；$\tau<1$ 更尖，$\tau>1$ 更多样。
- top-k：只保留概率最高 k 个。
- top-p/nucleus：按概率排序，保留累计概率首次达到 $p$ 的最小集合。
- greedy 是每步 argmax，确定但可能重复/退化。

采样顺序通常是 temperature → filter → renormalize → multinomial。必须保留至少一个 token，并正确停止于 EOT/max tokens。生成质量同时受训练 loss、数据、prompt、采样、上下文和 checkpoint 影响，不能用一个好故事判断模型。

## 4.10 训练监控

最小仪表盘：

- train/validation loss 与 PPL；
- tokens/sec、step time、MFU、数据等待时间；
- GPU peak memory；
- LR、gradient norm、clip fraction；
- parameter/activation RMS，必要时 update/weight ratio；
- NaN/Inf count；
- 固定 prompt 的周期样本；
- consumed tokens 和累计 FLOPs/墙钟。

曲线解读：

| 现象 | 优先怀疑 |
|---|---|
| loss 从不下降 | label 错位、参数没更新、LR 太小、mask 错 |
| 第一步正常随后 NaN | LR/数值、fp16 overflow、norm/softmax |
| train 降 val 不降 | 过拟合、分布错位、泄漏/切分问题 |
| loss 周期尖峰 | 数据 shard、LR、异常长/脏样本、恢复状态 |
| 吞吐周期下降 | eval/checkpoint、dataloader、通信 straggler |
| 好生成但 PPL 差 | cherry-pick、采样差异、验证处理不一致 |

## 4.11 消融怎样做才有意义

“删 RoPE/RMSNorm/SwiGLU 后跑一次”只是练习。有效消融需要：

1. 明确假设和主指标；
2. 固定 tokenizer、数据顺序、token budget；
3. 决定固定参数、FLOPs 还是墙钟，并解释；
4. 若组件改变参数量，用宽度调整做 matched-budget 对照；
5. 多 seed 或 bootstrap；报告均值、方差和失败率；
6. 同时记录吞吐/显存，避免质量小升但成本大增；
7. 不只给最终点，也看训练轨迹和不同数据域。

公开作业中值得学习的是把 LR、batch、norm、position、activation 分开记录并保留 W&B 报告；不应照搬其单次结论。小 TinyStories 模型上的排序可能无法外推到多十亿参数与不同语料。

## 4.12 论文思路

### Kingma & Ba, *Adam* (2014)

用一阶/二阶梯度指数移动平均构造逐参数自适应步长，并用偏置修正解决早期零初始化偏差。它对稀疏/非平稳梯度实用、调参相对稳健；代价是两份状态和可能不同于 SGD 的泛化/缩放行为。

### Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2017)

指出自适应优化器里 L2 正则与真正 weight decay 不等价，把参数收缩从梯度预条件中分离。方法简单，却使 weight decay 的意义和调参更一致。

### Loshchilov & Hutter, *SGDR* (2016/2017)

用 cosine annealing 和周期 restart 改善优化探索。LLM 训练多取其中 warmup + cosine decay，而不一定 restart；核心是平滑降低后期步长。

### Large-batch / critical batch 研究

把梯度噪声尺度与有用 batch 上限联系：batch 增大先提高并行效率，过临界点后样本效率收益饱和。实际 critical batch 随模型和训练阶段变化，所以动态 batch 是自然方向。

## 4.13 训练完成定义

一次“完成”的小 LM 实验应包含：可复现配置；通过单元测试；一个 batch 过拟合；train/val 曲线；峰值显存/吞吐；checkpoint 恢复测试；至少一组受控消融；固定和随机 prompt 样本；失败运行记录。只有最终 checkpoint 和一句“loss=…”不构成工程证据。
