# 来源、版本与公开作业评估

核对日期：2026-08-11。正文以官方 2026 Spring 材料为准；2025 视频/公开实现只作补充。链接可能在未来更新，关键版本信息在本地 `cs336-notes-sources/` 留有浅克隆。

## 官方材料

- [CS336 2026 官方课程主页](https://cs336.stanford.edu/)：课程结构、17 个核心 lecture、五份作业。
- [2026 官方 lectures 仓库](https://github.com/stanford-cs336/lectures)：Lecture 1/2/6/7/10/12/13/14/17 的 executable Python，以及 3/4/5/8/9/11/15/16 PDF；`references.py` 是论文地图主来源。
- [2025 官方课程归档](https://cs336.stanford.edu/spring2025/)：用于对应官方公开视频的讲次。
- [Stanford Online 2025 官方播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)：在 2026 无公开视频或 executable lecture 需要口头解释时补充。
- [Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)
- [Assignment 2: Systems](https://github.com/stanford-cs336/assignment2-systems)
- [Assignment 3: Scaling](https://github.com/stanford-cs336/assignment3-scaling)
- [Assignment 4: Data](https://github.com/stanford-cs336/assignment4-data)
- [Assignment 5: Alignment](https://github.com/stanford-cs336/assignment5-alignment)

2026 与 2025 的显著变化：A2 增 activation checkpointing 与 fully sharded DP；A3 目标改为按 B200 wall-clock API 规划；A4 更新 Common Crawl snapshot；A5 主线从 2025 的 Qwen2.5-Math/MATH、SFT/Expert Iteration/GRPO，改为 OLMo-2-1B/GSM8K、prompting、GRPO estimator variants 与 off-policy；SFT/DPO 通用助手变成可选 supplement。

## 公开作业如何筛选

评价维度：任务覆盖、是否声称通过官方测试、是否有 write-up/实验而非只有代码、是否报告失败/限制、性能数字上下文、代码组织与提交历史。stars 只作弱信号，绝不视为正确性证明。

### [Louisym/Stanford-CS336-spring25](https://github.com/Louisym/Stanford-CS336-spring25)

覆盖五份作业、目录清楚、社区使用多，适合看项目分层和端到端文件组织；作者也明确 A1 未做 CPU 并行 tokenizer、A2 未完成分布式部分、A3 API 未完整验证、A4 用小数据、A5 supplement 未做。因此不把它当“全套已验证答案”。

### [donglinkang2021/cs336-assignment1-basics](https://github.com/donglinkang2021/cs336-assignment1-basics)

A1 有详细文档、LR/batch/架构消融、W&B 报告、生成与 KV cache、Muon 扩展，并报告 TinyStories/OpenWebText validation loss。最值得借鉴的是实验记录和分组件文档。数值依赖其配置/机器，不与 2026 leaderboard 直接比较。

### [zhasion/CS336](https://github.com/zhasion/CS336)

五份作业均有逐题 write-up，作者展示 A1/A2/A4/A5 测试通过并明确 A3 受 API、部分实验受 GPU 资源限制。BPE 性能分解、无预热/预热 benchmark、系统表格、数据 inspect、GRPO/DPO 指标对笔记有启发。逐题内容可能含实现错误或不完整实验，正文只吸收可由官方材料/原理交叉验证的方法。

### [Melody-Zhou/stanford-cs336-spring2025-assignments](https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments)

五份作业实现和按主题博客链接完整，README 明确由 ChatGPT 辅助、A3 因 API 未完全测试。适合定位作业模块和中文解释，不作为独立正确性证据。

### [mocibb/cs336](https://github.com/mocibb/cs336)

中文概念笔记较强，A1 有 C++ 高速 BPE，A2 涵盖 causal 负载平衡与 Triton backward，且有大量提交。适合补系统优化问题清单；高性能实现必须在目标硬件/shape 复测。

### [heng380/cs336_assignment2](https://github.com/heng380/cs336_assignment2)

专注 A2，有 profiler、FlashAttention 与并行实现/QA，适合作为系统调试案例。README 本身实验说明有限，因此只辅助交叉检查，不把代码风格或结果写成推荐唯一方案。

### [Luyuan: CS336 Assignment 1 学习笔记](https://blog.wangluyuan.cc/2025/10/25/cs336-assignment-1/)

对中文初学者容易卡住的 BPE、RoPE、模型实现有清晰问题导向说明；属于个人经验，不替代官方规格。

## 使用边界

课程官方说明在校生不应查现成实现。本文没有把公开 solution code 合入笔记，也不提供逐测试函数的完整代码。公开数字只有在硬件、版本、shape、dtype、预算明确时才作为案例；任何未能由官方讲义、原论文或独立推导确认的个人结论都不会作为定论。

## 本地研究材料

`../cs336-notes-sources/` 包含官方和上述部分公开仓库的浅克隆；`_research/` 是 PDF 文本抽取与视觉抽查中间文件，不是正文。若只需要阅读笔记，可忽略或删除这些研究缓存；删除前注意它们是公开资料的本地副本，不影响正文 Markdown。
