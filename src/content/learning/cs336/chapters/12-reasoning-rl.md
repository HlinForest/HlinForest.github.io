# 12 推理强化学习：从 REINFORCE、PPO 到 GRPO/RLVR

对应官方 Lecture 16 与 2026 Assignment 5 主作业。

## 12.1 RLVR 与 RLHF 的区别

RLHF 的 reward 多为人类偏好代理，容易过优化。RL with Verifiable Rewards (RLVR) 选能自动验证的领域：数学最终答案、代码测试、定理检查、工具执行。它可廉价生成大量准确 reward，但只验证结果时可能奖励猜答案、格式漏洞或 hack verifier。

流程：prompt $x$ → policy 采样完整 response $y$ → verifier 给 $R(x,y)$ → 用 policy gradient 增加高 reward 序列概率。没有正确 reasoning label，这是它区别 SFT 的地方。

## 12.2 Policy gradient

目标：

$$J(\theta)=E_{y\sim\pi_\theta(\cdot|x)}[R(x,y)].$$

score-function trick：

$$\nabla_\theta J=E[R(x,y)\nabla_\theta\log\pi_\theta(y|x)].$$

序列 logprob 是 token 和：

$$\nabla\log\pi(y|x)=\sum_t\nabla\log\pi(y_t|x,y_{<t}).$$

单样本 loss 可写 $-A\sum_t\log\pi(y_t|\cdot)$，$A$ 是 reward/advantage。若 response 正确，提升所有采样 token 的概率，包括可能无关/错误的中间步骤；credit assignment 粗糙。

## 12.3 Baseline 为什么不引入偏差

若 baseline $b(x)$ 不依赖采样动作 $y$：

$$E[b(x)\nabla\log\pi(y|x)]=b(x)\nabla\sum_y\pi(y|x)=0.$$

所以用 $A=R-b$ 不改变期望，只降方差。PPO 学 value function 估 baseline；GRPO 用同 prompt 的 group rewards 构造相对 baseline，省 value model。

baseline 若包含当前样本，仍可有细微缩放/偏差讨论；leave-one-out 用其他 $G-1$ 样本均值更接近独立 baseline。

## 12.4 Importance sampling 与 on-policy

rollout 由旧策略 $\pi_{old}$ 采样，更新新策略 $\pi_\theta$：

$$r_t(\theta)=\frac{\pi_\theta(y_t|s_t)}{\pi_{old}(y_t|s_t)}
=\exp(\log\pi_\theta-\log\pi_{old}).$$

若策略已变很多，ratio 方差巨大，旧样本不代表新分布。on-policy 数据新鲜但生成贵；off-policy 可复用 rollout、提高系统效率，却需 clipping/权重控制。

## 12.5 PPO clipping

$$L^{clip}=E_t\left[\min(r_tA_t,
\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right].$$

若更新会把有利动作概率升得过多或不利动作降得过多，clipped 分支限制激励，构造局部 trust region。PPO 并不保证严格 KL 上限，仍需监控 approx KL、clip fraction、entropy。

## 12.6 GRPO

同 prompt 采样 $G$ 个 responses，reward $R_i$。经典 group-normalized advantage：

$$A_i=\frac{R_i-\bar R}{\mathrm{std}(R)+\epsilon}.$$

再把同一个 sequence advantage 分给其 response tokens，用 PPO-like ratio/clipping。优点：无需 value model、同题相对难度被中心化；代价：每 prompt 要多个 samples、组内 reward 全同则信号为零、std 归一带来题目加权效应。

### std 归一的隐含权重

二值 reward 下，接近全对/全错的组 std 小；加 epsilon 后行为复杂，且只有一正/一负的题可能产生较大标准化 advantage。Dr. GRPO 类工作指出 length/std normalization 可引入 bias，提出只减 group mean、不除 std 等简化。不要因为公式叫“normalized”就假定更正确。

## 12.7 Token loss 与 masking

对 response token mask $m_{it}$，常见：

$$L=-\frac{1}{\sum m}\sum_{i,t}m_{it}\,ell_{it}(r_{it},A_i).$$

可能先每序列平均再组平均，或全 token 平均；两者对长 response 权重不同。

- **per-token average across batch**：长序列贡献更多 token，但每 token同权。
- **per-sequence mean**：每响应总权重相同，短响应每 token 权重大。
- **不长度归一的 sum**：长响应总梯度大，可能产生 length bias。

MaxRL/长度归一变体改变的正是优化目标/方差，必须结合 reward length correlation 检验。padding、prompt、截断 token 必须 mask；EOS 是否包含要一致。

## 12.8 2026 Assignment 5 的实验脉络

新版主作业用 OLMo-2-0425-1B + GSM8K：

1. zero/few-shot/CoT baseline；
2. on-policy GRPO；
3. RFT、Dr. GRPO、MaxRL 等 policy-gradient 选择；
4. off-policy GRPO 与 clipping。

这比 2025 的 Qwen2.5-Math + MATH + SFT/expert iteration 路线更聚焦 RL estimator 科学。笔记保留两者经验：SFT/Expert Iteration 是稳定强起点；纯 RL 用验证信号探索未给 demonstration 的解法。

## 12.9 Rejection Fine-Tuning / Expert Iteration

当前 policy 每题采样多个，保留 verifier 正确的 responses，作为 SFT 数据训练；重复迭代。它把 sparse reward 变成监督：简单稳定、可离线；但只学已成功样本，分布越来越自我强化，缺少显式负例/importance correction。

适合先产生 bootstrap，再 RL。记录每题成功率和去重，避免易题/重复答案淹没难题。

## 12.10 系统瓶颈：生成比反向贵

RL loop 包含 inference engine 与 training engine：

```text
sample prompts -> vLLM rollouts -> verifier
-> tokenize/mask/log old probs -> train microbatches
-> sync new weights -> repeat
```

关键权衡：

- 同步 rollout 最 on-policy，但 GPU 等待；
- 异步提高利用率，但 policy lag/off-policy；
- 训练与推理共卡要频繁释放/加载、同步权重；
- 分池要传权重/adapter 和调度；
- group 长度不齐，最长 response 造成 straggler；
- vLLM sampling 的 tokenizer/template/logprob 必须与 trainer 一致。

系统效率和算法 on-policyness 是同一问题的两端。记录 rollout policy version；限制样本 age；按版本重算/保存 old logprob。

## 12.11 Reward/verifier hacking

数学 parser 只找 `\boxed{}`，模型可能输出多个框、注入 parser、利用等价判断 bug；代码可超时、读测试、调用外部资源。verifier 要 sandbox、资源限制、canonicalize、多测试、拒绝歧义。

监控：answer reward、format reward、人工 correctness 分开；保留高 reward 样本抽查；使用 held-out verifier/测试；改变格式看能力是否保持。reward 100% 但人工错是系统事故，不是成功训练。

## 12.12 长 CoT 与 “aha moment”

RL 后 response 变长、出现反思/回溯可能与高分相关，但长度本身不是推理。可能是：更多计算提高成功率；reward/normalization 偏长；训练语料已有模板被放大；采样筛选效应。

验证：控制 token budget 比较；按长度分层正确率；给相同问题不同 max tokens；检查重复/空想；用过程验证或隐藏测试。不要仅凭一条漂亮轨迹宣称涌现。

## 12.13 关键诊断指标

- reward mean/std、每题 group 全同率；
- pass@1/pass@k、format/answer 分解；
- response length、EOS rate、截断率；
- policy entropy、KL to reference/old；
- importance ratio 分布、clip fraction、ESS；
- positive/negative advantage token 数；
- rollout/train tokens/s、policy lag；
- 多 seed learning curve 与置信区间；
- held-out domain/base capability regression。

ESS 可粗看 importance weights 是否被少数样本支配：

$$ESS=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.$$

## 12.14 论文思路

### Williams, *REINFORCE* (1992，课程背景)

用 $R\nabla\log\pi$ 得到无需可微环境的无偏 policy gradient；简单通用但方差高，奠定现代 LLM sequence RL。

### Schulman et al., *PPO* (2017)

用 clipped importance ratio 近似限制策略每次变化，兼顾实现简单与样本复用。PPO 在传统 RL 和 RLHF 成为默认，但 value 训练、超参、KL 与大模型系统复杂。

### DeepSeekMath / GRPO (2024)

用同 prompt 多样本的组相对 reward 代替 value model，结合 PPO-style objective，降低 RL 训练内存/复杂度；在数学推理展示效果。组 normalization 和 token/长度处理不是唯一选择，后续工作继续修正。

### DeepSeek-R1 (2025)

展示大规模可验证 reward RL 能增强长链推理，并结合 cold-start 数据、多阶段 RL/SFT 与蒸馏把能力迁移到小模型。不能简化为“只要 GRPO 就会推理”：base/model/data/verifier/系统和多阶段 recipe 都关键。

### Dr. GRPO / 2025 policy-gradient 分析

分析 GRPO 中 std 与长度 normalization 的问题，说明许多“工程细节”实际改变样本权重和优化目标；提出去掉某些归一或更直接 estimator。核心启示：从数学期望逐项推实现，不靠算法名。

## 12.15 RL 作业验收

1. 手工 logprob/reward 的 PG、ratio、clip、mask 单元测试。
2. 采样 policy 与 trainer tokenizer/template/weights 版本一致。
3. baseline→on-policy→variant 的多 seed 曲线。
4. reward、pass@k、length、KL、entropy、clip/ESS 同报。
5. off-policy sweep 报 policy lag 与 throughput，不只 final reward。
6. 高 reward/失败样本人工审计与 verifier robustness。
7. base/general benchmark 回归，证明不是只会 GSM8K 格式。
