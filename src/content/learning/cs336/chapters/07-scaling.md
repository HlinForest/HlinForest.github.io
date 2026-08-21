# 07 Scaling Laws：小实验怎样指导昂贵的大训练

对应官方 Lecture 9、11 与 Assignment 3。

## 7.1 Scaling law 解决的不是“模型越大越好”

固定训练计算预算 $C$，参数量 $N$ 大意味着每 token 贵、只能看较少数据；$N$ 小能看更多 token，但容量不足。问题是：

$$\min_{N,D_{tok},h}\;L(N,D_{tok},h)quad
\text{s.t. } C(N,D_{tok},h)\le C_{budget},$$

$h$ 包含深宽、LR、batch、schedule、数据 mix 等。Scaling law 用较小运行拟合平滑规律，预测大预算的计算最优配置、预期 loss 与不确定性。

## 7.2 最常用的三个关系

粗略 dense Transformer 训练计算：

$$C\approx 6ND_{tok}.$$

Kaplan 风格分别观察 loss 随 $N,D,C$ 的幂律下降并趋向不可约项。Chinchilla 的联合参数化常写：

$$L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.$$

- $E$：该数据分布/目标下的不可约 loss；
- $A/N^\alpha$：容量不足项；
- $B/D^\beta$：数据不足项。

幂律在 log-log 上近线性，但不能对带 $E$ 的原 loss 直接随便取 log 做普通线性回归；需非线性拟合/稳健损失和多初值。

## 7.3 IsoFLOP 方法

对多个小计算预算 $C_i$：

1. 选一组不同 $N_{ij}$。
2. 令 $D_{ij}=C_i/(6N_{ij})$，使每组近似同 FLOPs。
3. 每个模型用合适 LR/batch 训练到预算结束，得到 final loss。
4. 对每个 $C_i$ 的 `loss vs log(N)` 拟合平滑 U 型下包络，求 $N_{opt}(C_i)$。
5. $D_{opt}=C_i/(6N_{opt})$。
6. 拟合：

$$N_{opt}=aC^\alpha,qquad D_{opt}=bC^\beta.$$

在 log 空间回归并外推目标 $C^*$。U 型直觉：左侧模型太小即使多数据也欠拟合；右侧模型太大、预算不足以优化/看数据。

### 为什么不能每组只跑两个点

最优点必须被左右包围；否则你拟合的是边界趋势，不是 U 型极小值。先做廉价 pilot 定范围，再自适应在当前极小值附近加点。Assignment 3 的真正难点是实验设计，而不是 `curve_fit`。

## 7.4 Kaplan 与 Chinchilla 为什么不同

### Kaplan et al. 2020

系统改变模型、数据与计算，发现 test loss 呈平滑幂律；其计算最优分析倾向随计算更快增大模型、相对少增数据。它给行业“可预测扩展”的信心。

### Hoffmann et al. 2022 / Chinchilla

指出早期分析中不同规模未充分调 LR/训练过程，且很多大模型训练 token 太少。用三种方法，尤其 IsoFLOP 与联合 loss 拟合，得到参数和数据应更接近同比例扩大；70B Chinchilla 用更多 tokens 超过更大但欠训练的 Gopher。

不是谁“数学对、谁错”，而是数据、计算定义、优化调参和观测范围不同。今天常把 Chinchilla ratio 当起点，但推理成本、数据重复、质量与下游能力会改变最优点。

## 7.5 计算最优训练不等于产品最优

若只最小化一次训练后的 loss，Chinchilla-like ratio 合理。产品总成本还包括长期推理：小模型多训 tokens（overtraining）可能训练上非最优，却在海量 decode 请求中更便宜。2024 的 overtrained scaling 研究把 inference demand 纳入目标，说明最优 token/parameter 比随部署场景改变。

同样，训练最优 loss 不等于下游任务最优。某些能力、数据域或 post-training 可在不同规模出现不同斜率；应用应拟合真正关心的 metric/cost frontier。

## 7.6 Hyperparameter scaling

架构和优化超参数不随规模稳定，会污染 scaling fit。

### 深宽与 heads

在参数预算固定时，过深可能优化困难/串行，过宽可能矩阵好算但表达/层级不足。head dim 常保持在硬件/经验友好范围，头数随 D 增长。所有候选应满足 tile 对齐，避免系统低效被误判为模型规律。

### Learning rate 与 batch

最优 LR、warmup、critical batch 随规模/loss 变化。若大模型用小模型 LR 直接发散，它的高 final loss 不是容量规律。应对不同规模做廉价 LR sweep，或采用能稳定迁移的参数化。

### μP

Maximal Update Parametrization 给不同类型参数规定宽度相关初始化和 LR scaling，使宽度增长时激活、特征学习与更新保持非平凡极限。目标是把小模型调出的超参数迁移到大模型。它要求实现严格遵循 parameterization；普通 SP checkpoint/LR 不能随意混用。

### WSD

稳定阶段训练一条长轨迹，在不同预算点各自接 decay，可以用较少重复训练估计多 budget 曲线。MiniCPM 的公开 recipe 把 μP、WSD 和 scaling 结合，是“省 scaling 实验计算”的案例。

## 7.7 拟合的统计与工程细节

### 数据表每行至少包含

`N_nonembed, N_total, tokens, FLOPs_est, wall_time, layers, D, F, heads, LR, batch_tokens, schedule, data_mix, seed, min/last val loss, failed_reason`。

不要把 NaN/OOM 运行无声删除；失败边界也是最优设计空间信息。

### loss 取哪个点

预算截止 final loss 符合 compute-optimal 问题；取训练中的 minimum 会偏爱高噪声/过拟合并引入选择偏差。validation 数据、频率和 tokenization 必须相同。

### 稳健拟合

- 对 loss 用 Huber 等稳健残差，避免单次异常支配；
- 多随机初始化，检查局部极小；
- 参数加合理正约束；
- bootstrap runs/IsoFLOP groups 得置信区间；
- leave-one-budget-out 检验外推；
- 比较多种合理形式，不只报最佳 $R^2$；
- 画残差随 $N,D,C$，系统结构说明模型缺项。

### 外推比

若最大实验预算只到目标的 $1/1000$，即使拟合线很好也危险。报告 `target C / max observed C`、预测区间和最近邻结构。Scaling law 是决策工具，不是自然定律。

## 7.8 FLOPs 预算与墙钟预算

Assignment 3 的 2026 版本按 B200 hours 接口约束，墙钟受 shape、kernel、并行和故障影响。两个同 FLOPs 配置可能因 MFU 不同消耗完全不同时间。实际规划可有两套模型：

1. **algorithmic scaling**：loss vs theoretical FLOPs；
2. **systems scaling**：achieved tokens/s/MFU vs shape/world size；

合成 `loss vs dollars/time`。有时稍差的算法配置因硬件利用高，在固定日期前反而最好。

## 7.9 数据质量改变 scaling

参数 $E,A,B,\alpha,\beta$ 不是模型架构常数；换 tokenizer、data mix、过滤和重复率就会变。高质量数据可让相同 token 提供更多学习信号，但容量小、数据易时更早饱和。DeepSeek 的 scaling 研究特别提醒不同数据集的 law 不同。

因此用 TinyStories 的曲线预测网页/代码大模型是类别错误；课程小实验学习的是方法，而非生成前沿数字。

## 7.10 实际决策流程

给 48 B200-hours 大训练、12 B200-hours 调研预算，可这样做：

1. benchmark 候选 shape，排除 MFU 极差或内存不可行配置。
2. 选 4-6 个 pilot compute budgets，覆盖至少 1-2 个数量级。
3. 每组 5-8 个 N，确保极小值被包围；先低保真再加密。
4. 每规模做 LR/batch 小 sweep，记录最佳及失败。
5. 用相同数据/验证/调度规则跑 IsoFLOP。
6. 拟合 U 型极小与 power laws；bootstrap。
7. 计算目标 N/D 的区间，再映射到可整除的层、宽、heads。
8. 在接近目标比例的中等预算做 holdout 验证。
9. 留出硬件故障、checkpoint、数据延迟和重跑余量，不把 100% 预算押满。

## 7.11 论文思路

### Kaplan et al., *Scaling Laws for Neural Language Models* (2020)

问题：性能能否从小模型可靠预测大模型。方法：跨多数量级改变参数、数据、计算，拟合 power law，并分析计算最优前沿。贡献是展示曲线平滑、弱依赖具体细节；局限是当时优化/数据范围导致最优 N/D 结论后来被修正。

### Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022)

用 fixed-model sweep、IsoFLOP、联合参数 loss 三种方法交叉验证；强调每个规模调 LR，结论是 N 和 D 更均衡扩展。Chinchilla 实证说明“更小模型 + 更多数据”可在同计算下更强。

### Yang et al., *Tensor Programs V / μTransfer* (2022)

问题：标准参数化下宽度变化会改变激活/更新尺度，小模型超参难迁移。方法：μP 为各参数张量规定初始化、乘子和 LR scaling，使超参可从小 proxy 迁移。收益是减少大模型 sweep；局限是实现敏感，宽度之外的深度、数据和优化变化并未全部解决。

### MiniCPM / WSD (2024)

把 μP、scaling law 与 warmup-stable-decay 结合，先长时间稳定训练，再对不同预算 checkpoint decay，减少重复实验。核心是“训练轨迹复用”；需确认中途切 decay 与独立最优 schedule 的差距。

## 7.12 Scaling 作业验收

交付物应包括原始 run table、可复现拟合脚本、IsoFLOP 图、每预算极小值与误差、power-law 参数置信区间、holdout 预测误差、目标配置离散化过程、系统可行性和风险清单。只给一个预测 N 与 loss 不是完整 scaling analysis。
