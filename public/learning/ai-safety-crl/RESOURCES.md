# AI Safety × Causal Representation Learning Resources

## Knowledge

- Local book: `D:/book-oct11.pdf` — *Causal Artificial Intelligence* draft, Draft 0.27.
  本课程的因果主线来源。优先阅读：PDF pp. 13-14（全书假说）、27-30（AI 挑战）、49-50（变量学习入口）、1071-1123（第 18 章 CRL）。注意：草稿中仍有 `??` 交叉引用与空缺练习，不能把占位符当正式定理编号。
- [Yujia Zheng — official homepage](https://yjzheng.com/)
  当前研究定位与论文入口。用于校准 causal representation learning、latent concepts、world models 和 feature consistency 的关系。
- [Zhang, Xie, Ng & Zheng (ICML 2024): Causal Representation Learning from Multiple Distributions](https://proceedings.mlr.press/v235/zhang24br.html)
  非参数、多分布 CRL 的主要原始来源。用于理解“充分变化 + 稀疏结构”能识别到什么。
- [Zheng, Xie & Zhang (ICML 2025): Nonparametric Identification of Latent Concepts](https://proceedings.mlr.press/v267/zheng25p.html)
  用于进阶理解一般潜在概念的非参数可识别性。
- [Davin Choo — official homepage](https://davinchoo.com/)
  当前研究定位与论文入口。用于校准 statistical learning、causal inference、imperfect advice 与 resource-efficient algorithms 的关系。
- [Choo, Gouleakis & Bhattacharyya (ICML 2023): Active Causal Structure Learning with Advice](https://proceedings.mlr.press/v202/choo23a.html)
  “建议准确时受益、建议任意错误时仍有保证”的主要原始来源。
- [Kim, Choo, Neoh & Tambe (AAMAS 2026): Incentive-Aware AI Safety via Strategic Resource Allocation](https://arxiv.org/abs/2602.07259)
  把训练数据审计、部署前评测与多模型部署写成有限资源下的 Stackelberg security games。
- [Schölkopf et al. (2021): Toward Causal Representation Learning](https://arxiv.org/abs/2102.11107)
  CRL 的经典问题地图：从低层观测发现高层因果变量。

## Wisdom (Communities)

- [Conference on Causal Learning and Reasoning (CLeaR)](https://www.cclear.cc/)
  用于检验 CRL/因果发现问题设定是否与领域惯例一致，并跟踪 tutorials、workshops 与 open problems。
- [International Programme on AI Evaluation — Open Seminars](https://ai-evaluation.org/open-seminars)
  用于把理论假设与真实模型评测、能力/安全测量实践对照。

## Gaps

- 需要用户反馈其数学背景、主要应用场景与可投入时间，才能把后续证明深度和项目规模调准。
- “CRL 特征如何改善战略安全评测分配”目前是研究接口，不是书中或两位研究者论文已经给出的统一定理；课程会明确标为综合推论或研究假设。

