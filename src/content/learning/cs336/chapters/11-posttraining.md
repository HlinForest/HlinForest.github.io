# 11 后训练：SFT、偏好学习、DPO 与 RLHF

对应官方 Lecture 15 与 Assignment 5 可选 supplement。

## 11.1 Base model 为什么不会自动成为好助手

预训练目标是模仿广泛文本的续写分布，里面混有文章、对话、广告、代码、错误、攻击和多种角色。用户需要的是特定条件行为：理解指令、给合适格式、诚实表达不确定、遵守安全边界。后训练把宽泛分布集中到目标交互分布。

典型路径：

```text
base model
-> supervised fine-tuning (SFT) on demonstrations
-> preference optimization (reward model + PPO, or DPO variants)
-> safety/domain/tool-use tuning
-> online/offline evaluation and red teaming
```

后训练通常数据远少于预训练，但每条信息更针对行为；它能重排/激活已有能力，不能可靠补回 base model 根本没学过的大量知识。

## 11.2 指令数据长什么样

一条 conversation 由 role + content 组成，经 chat template 变成 token：system、user、assistant、tool、分隔/EOS。模板是模型协议，不是显示装饰。训练/推理模板不一致会造成明显退化。

数据来源：

- 人类专家示范；
- 真实用户交互（需隐私/同意治理）；
- self-instruct/模型生成再过滤；
- 强模型蒸馏；
- 工具执行、代码测试、数学验证器生成的轨迹；
- 多轮、拒答、纠错、澄清、格式数据。

质量维度：正确、相关、完整、简洁、风格、多样、难度、安全。大量模板化低质数据会让模型学口癖和虚假自信。

## 11.3 SFT 目标与 masking

对 prompt $x$、assistant response $y$：

$$\mathcal L_{SFT}=-\sum_{t\in y}m_t\log\pi_\theta(y_t\mid x,y_{<t}).$$

$m_t$ 只在要学习的 assistant tokens 为 1；system/user/padding 通常 mask 掉。多轮时可对所有 assistant spans 训练，或只最后一轮；必须明确定义。

常见静默 bug：

- tokenizer 后再用字符长度切 mask，token 边界错；
- prompt/response 拼接时 special token 重复或缺失；
- EOS 没有 loss，模型不会停；
- padding token 同 EOS，mask 逻辑把真实 EOS 忽略；
- truncation 把 response 全截掉仍算样本；
- packed samples 互相 attention；
- loss 除总序列长度而非 response token 数。

最佳测试是打印 `token, id, role, loss_mask` 表，人工检查边界。

## 11.4 SFT 的优化取舍

SFT 数据少、模型已有能力，LR 通常远低于 pretraining，epoch 过多会遗忘/过拟合风格。监控：held-out response NLL、任务正确率、base 能力回归、输出长度、拒答率、多样性。

全参数 tuning 表达力强；LoRA 在权重更新上用低秩 $\Delta W=BA$，显存/存储低，适合多 adapter，但 optimizer/activation 仍有成本，rank 限制可能不足。课程 supplement 关注全流程机制，不应把参数高效方法当成目标函数的替代。

## 11.5 偏好数据

同 prompt $x$ 有 chosen $y_w$ 与 rejected $y_l$。来源：人类 pairwise、模型 judge、规则/验证器、不同模型/采样候选。标注协议决定“更好”含义；帮助、安全、事实、风格若混成一个标签，模型难理解取舍。

控制 position bias，随机左右；记录 annotator disagreement；允许 tie/两者都坏。长度是强 confounder：如果 chosen 总更长，模型学“多说”而非更正确。

## 11.6 经典 RLHF：Reward Model + PPO

Reward model $r_\phi(x,y)$ 用 Bradley-Terry：

$$P(y_w\succ y_l)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)),$$

$$\mathcal L_{RM}=-\log\sigma(r_w-r_l).$$

然后 policy 最大化 reward 并受 reference KL 约束：

$$\max_\theta E_{y\sim\pi_\theta}[r_\phi(x,y)]-\beta D_{KL}(\pi_\theta\|\pi_{ref}).$$

PPO 用 rollout、value/advantage、importance ratio clipping 更新。优势：可直接优化 learned reward、使用在线样本；代价：policy/value/reference/reward 多模型、采样贵、稳定性和 reward overoptimization 难。

Reward model 只是人类偏好的有限代理。policy 找到 RM 漏洞时 reward 上升、人类质量下降（Goodhart）。定期做人类/外部 judge 验证、控制 KL、扩充对抗数据。

## 11.7 DPO 的推导直觉

KL-regularized reward optimization 的最优 policy 满足：

$$\pi^*(y|x)\propto\pi_{ref}(y|x)\exp(r(x,y)/\beta).$$

因此隐式 reward 可由 log policy ratio 表示：

$$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+C(x).$$

代入 Bradley-Terry，$C(x)$ 在 chosen/rejected 差中抵消，得到 DPO loss：

$$\mathcal L_{DPO}=-\log\sigma\left(
\beta[(\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x))
-(\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x))]
\right).$$

它直接让 policy 相对 reference 更偏 chosen，不单训 reward model、不在线 rollout。$\beta$ 控制偏离 reference/隐式 reward 尺度（不同实现命名和直觉可能相反，读公式）。

### 序列 logprob

$$\log\pi(y|x)=\sum_{t\in response}\log\pi(y_t|x,y_{<t}).$$

求 chosen-rejected 同模型差时 prompt logprob 可抵消，但实现通常仍只 mask response，避免 truncation/template 差异。用 sum 会使长响应有更大绝对量；若改 mean 就改变目标，不是无害数值技巧。

## 11.8 DPO 常见错误

- policy/reference 不是从同一起点或 reference 仍更新；
- chosen/rejected prompt tokenization 不同；
- 未 mask prompt/padding；
- label shift 错一位；
- reference forward 构图浪费显存；
- 把四个 logprob 先取均值/长度归一却没说明；
- $\beta$ 符号/位置错；
- 对 chosen/rejected 分开 batch 导致 dropout 不一致/效率低；
- 只看 DPO loss，没看 chosen/rejected margin、KL、长度与任务质量。

单元测试用人工 logprobs：若 policy 比 reference 更偏 chosen，loss 应低；交换 chosen/rejected 应反转。

## 11.9 Constitutional AI、RLAIF 与蒸馏

Constitutional AI 用一组原则让模型 critique/revise 自己回答，并用 AI feedback 构造 preference，降低每条都由人标的成本。它把规范显式化、可扩展，但 constitution 与 judge 模型的偏见/盲点会传递。

蒸馏强模型回答能迅速提升小模型 instruction/reasoning 风格；LIMA 的“少量高质量数据”强调 base model 已有知识时，SFT 主要学交互格式。局限：teacher 错误/口癖、许可、覆盖、学生容量；只模仿成功轨迹缺少负例与自我纠错。

## 11.10 安全不是单一拒答率

评测至少区分：明显恶意、双用途、合法敏感、高风险专业建议、无害但含触发词。过度拒答会伤 helpfulness，过少拒答增风险。red team 用人工与自动攻击；保存 attack category、模型原文、judge 与人工复核。

SFT/DPO 能塑造行为，但不能提供形式保证。部署还需 policy、moderation、权限、工具 sandbox、日志与 incident response。训练数据涉及个人/有害内容时要做访问和心理健康保护。

## 11.11 评测矩阵

课程 supplement 用 MMLU（知识）、GSM8K（推理）、AlpacaEval（对话 judge）、SimpleSafetyTests（安全）体现多目标。正确实验应比较 base→SFT→DPO：

- 每项能力变化和 CI；
- 输出长度/格式/拒答率；
- base capability regression；
- chosen/rejected log-ratio margin；
- reference KL/隐式 reward；
- 人工抽查 failure taxonomy。

单一 AlpacaEval 胜率提升可能只来自变长；单一 safety 高分可能是全拒答。

## 11.12 论文思路

### Ouyang et al., *InstructGPT* (2022)

用人类 demonstrations 做 SFT，用 pairwise preference 训 reward model，再 PPO 优化并加 KL。小的 aligned 模型在人类偏好上可超过更大 base GPT-3，说明目标/数据匹配重要。局限是标注者/任务分布有限、RM overoptimization 与复杂 pipeline。

### Bai et al., *Constitutional AI* (2022)

用显式原则驱动 self-critique/revision，再用 AI preference 做 RL；扩展安全反馈并提高透明度。原则解释和 AI judge 偏差仍需人类治理。

### Rafailov et al., *DPO* (2023)

从 KL-regularized RL 的最优 policy 推出隐式 reward，把偏好学习化成简单分类式 loss；无需 reward model rollout/PPO。它简洁稳定，但依赖离线 preference 覆盖，过强训练仍会偏离 reference/过拟合偏好。

### Zhou et al., *LIMA* (2023)

少量精心挑选 SFT 数据可让强 base model 学会高质量交互，提出“表面对齐”视角。不能误读成数据永远越少越好：前提是 base 能力强、数据高质量，广覆盖/安全仍需更多数据与评测。

## 11.13 后训练验收

1. chat template/token/mask 人工可视化测试。
2. base/SFT/preference 三阶段可复现 checkpoints。
3. 多任务、多目标、长度控制的 evaluation card。
4. DPO/RM 数学小例单元测试。
5. KL、entropy、margin、长度、拒答与 reward 曲线。
6. 人工 failure taxonomy 和 red-team 抽样。
7. 明确数据来源、annotator/judge、许可与隐私。
