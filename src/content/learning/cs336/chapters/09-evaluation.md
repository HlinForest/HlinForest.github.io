# 09 评测：从 validation loss 到可信的能力结论

对应官方 Lecture 12，并贯穿所有作业。

## 9.1 评测先问“为了什么决策”

同一个模型可被评测来：

- 监控预训练是否正常；
- 选择 architecture/data/checkpoint；
- 衡量通用知识、推理、代码、长上下文、安全；
- 预测用户偏好/产品 SLO；
- 做科学比较或回归测试。

不同目的需要不同数据与统计。一个排行榜分数不能同时回答所有问题。

## 9.2 内在与外在指标

- **validation NLL/PPL**：连续、低方差、训练早期可用，接近目标；但不直接代表指令遵循/事实/安全。
- **multiple-choice logprob**：比较候选条件概率，可 length-normalize；易受选项顺序/格式影响。
- **exact match/pass@k**：数学/代码可验证，客观；忽略部分正确与解释质量。
- **generative judge/human preference**：接近聊天质量；有 judge 偏差、位置/长度/风格偏好。
- **calibration**：置信与正确率匹配；可用 ECE/Brier，但生成任务定义复杂。
- **systems metrics**：latency、throughput、memory、cost；质量相同才可比。

## 9.3 Prompt 是评测协议的一部分

zero-shot/few-shot、system prompt、chat template、答案格式、stop tokens、temperature、max tokens 都能改结果。公平比较必须：

- 使用各模型正确 tokenizer/chat template；
- 报完整 prompt 或 hash/version；
- 控制 few-shot 示例与顺序；
- 生成评测固定 sampling seeds/参数；
- 解析失败单独报告，不能静默算错/丢弃；
- 做 prompt perturbation，判断结论是否脆弱。

基础模型常更适合 completion-scoring，指令模型适合 chat generation；强行用同一种形式可能不公平。

## 9.4 Train-test contamination

若 benchmark 或近似解答进入预训练/后训练，分数测到记忆而非泛化。污染形式：

- exact 文本；
- 去格式/大小写后的近重复；
- 题目相同但答案/顺序不同；
- GitHub 解题代码、讲解网页、合成改写；
- benchmark 训练集被用于 instruction tuning，间接影响测试集模板。

检测层级：规范化 exact hash → n-gram/MinHash → embedding/检索 → 模型记忆探针。对预训练语料做 decontamination 要在 split 前/数据版本上记录；只删测试题 exact string 远远不够，也可能误删大量常识文本。

时间切分（模型训练截止后发布的数据）有帮助，但网页时间戳、未来泄漏和数据抓取不透明仍会破坏。

## 9.5 Dataset quality

人工抽查每个 benchmark：

- 问题是否有唯一正确答案；
- 标注/解析是否错；
- 是否依赖过时事实；
- 文化/语言分布是否窄；
- 模板线索能否猜答案；
- few-shot 是否泄漏策略；
- metric 是否奖错误行为。

MMLU 覆盖 57 个学科的多选知识题，广但格式单一、会受 contamination/choice scoring 影响；GSM8K 有可验证小学数学，适合 RLVR，但覆盖推理类型窄，容易对答案格式/数据分布过拟合。

## 9.6 pass@k 与采样

代码/数学从每题采样 $n$ 个，$c$ 个正确，若从中无放回选 $k$ 个，至少一个正确的无偏估计常写：

$$\widehat{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.$$

pass@1 测单次可靠性，pass@k 还测分布中是否“藏着”正确答案。改变 temperature、n 或去重会改 metric；必须一起报告。产品一次回答不能用 pass@64 替代。

## 9.7 LLM-as-a-judge

优点：规模化比较开放回答。典型偏差：

- position bias（偏 A/B）；
- verbosity/style bias；
- self-preference/同系列偏好；
- 对事实、代码执行、安全边界判断错；
- judge prompt 注入；
- reference answer 本身错误。

缓解：交换位置两次、隐藏模型名、控制长度或报告长度、多个 judge/人类校准、可验证任务优先执行、抽样人工审计、报告 tie/invalid 和 judge 版本。judge 更新后旧分数不可直接比。

## 9.8 统计不确定性

对 $n$ 道独立二值题，准确率标准误粗略：

$$SE\approx\sqrt{\frac{p(1-p)}{n}}.$$

比较同一题集两个模型应做 paired bootstrap/每题差值，而不是把两者当独立。生成与 RL 还受 sampling seed、训练 seed 影响，要分解：题目抽样不确定性、生成随机性、训练运行方差。

报告 confidence interval 和 effect size。0.3 分提升若 CI 重叠且做了 50 次试验选最好，证据很弱。多 benchmark 选择也有 multiple comparisons/leaderboard overfitting。

## 9.9 Evaluation harness 设计

推荐分层：

```text
dataset loader/version
-> prompt renderer/chat template
-> model adapter (logprob/generate)
-> parser/normalizer
-> scorer
-> per-example record
-> aggregate + confidence interval
```

保存 per-example：prompt hash、raw output、parsed answer、score、latency、token 数、error。聚合表之外的原始记录是排查解析错误、reward hacking 和分域失败的关键。

先用 dummy model/人工答案测 harness：全对应 100%，全错 0%，格式边界行为明确。很多“模型提升”其实是 parser 改了。

## 9.10 Evaluation card

每次结论附：

```text
model/checkpoint/tokenizer/chat template
benchmark name/version/split/license
prompt/few-shot/examples order
generation/logprob settings
parser/scorer/judge version
contamination/decontamination policy
sample size and confidence interval
hardware/runtime for systems metrics
known limitations and failed examples
```

## 9.11 论文思路

### Hendrycks et al., *MMLU* (2021)

构建跨 STEM、人文、社科、专业领域的 57 科多选题，目标是测广泛知识与问题求解。它成为通用 base model 指标，但多选格式、静态题库、污染和知识时效限制了“智能”解释。

### OLMo/Dolma 的开放评测实践

公开数据、训练中间 checkpoint 和去污染规则，让研究者能把分数变化追到数据/训练阶段；Paloma 等多域 PPL 避免单一网页分布。核心启示是可审计 lineage 本身就是评测能力。

## 9.12 评测高手的判断

面对“模型 A 超过 B 2 分”，先问：同一 tokenizer/template 吗？样本多大、CI 多宽？训练/benchmark 是否污染？用了几次 prompt/seed 后挑最好？judge 是否偏长度？成本和延迟怎样？提升集中在哪些样本？没有这些，结论只是数字，不是证据。
