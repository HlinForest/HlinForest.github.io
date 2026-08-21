# 13 多模态模型：把图像变成语言模型能共同建模的序列

对应官方 Lecture 17。CS336 只做导论，本章连接前面的 tokenizer、架构、数据与后训练。

## 13.1 三种主路线

### 冻结视觉编码器 + 投影 + LLM

图像经 ViT/Conv encoder 得 patch features `[B,Nv,Dv]`，projector 映到 LLM 宽度 `[B,Nv,D]`，与文本 embeddings 拼接。训练 projector/LLM adapter 或部分解冻。便宜稳定，依赖预训练视觉空间。

### Cross-attention connector

语言 token 在若干层通过 cross-attention 读取视觉 features（Flamingo 类），不必把所有视觉 token 塞入 self-attention；架构更复杂，但可控制视觉条件注入。

### Unified autoregressive tokens

把文本、图像离散 token/连续 patch 放进统一序列，用一个 Transformer 预测/理解多模态；可同时生成图像/文本，训练目标和序列成本更复杂。

## 13.2 ViT patchification

图像 `[H,W,3]` 按 $P\times P$ patch 切成 $N_v=HW/P^2$ 个，展平线性投影。分辨率加倍时 patch 数增 4 倍，self-attention 成本可能增 16 倍。动态分辨率、tiling、token pooling/merging 是多模态系统的核心，不是预处理小事。

位置要表达二维结构：learned 2D、RoPE 2D、相对 bias，且处理裁剪/缩放/多图。只按一维 raster 顺序位置可能能学，但分辨率外推/空间关系受限。

## 13.3 Contrastive 与 generative pretraining

CLIP 对 image/text encoders 做对比学习：匹配 pair 相似度高、batch 内负例低。得到强对齐表示和 zero-shot 分类，但不直接生成详细文本。

视觉语言生成用 next-token/seq2seq 预测 caption、OCR、QA；能学细粒度条件生成，但容易语言先验“看图说错”。现代模型混合 contrastive、captioning、interleaved document、grounding 与 instruction 数据。

## 13.4 数据

网页 image-alt/caption 对噪声大、版权/隐私/人物信息敏感；OCR 与文档需要高分辨率；图表/GUI/视频又是不同分布。pipeline 除文本过滤外还需：

- image decode、尺寸/长宽比、模糊/水印/重复；
- image-text 相似度与 caption 质量；
- perceptual hash/embedding near-dedup；
- NSFW/人脸/PII 与许可；
- OCR、语言、图表/文档/自然图分类；
- benchmark 图像与改尺寸/截图近重复去污染。

合成 caption 能增强细节，却继承 teacher hallucination；保留原 alt/metadata 与生成 caption 的来源标签。

## 13.5 训练阶段

常见：

1. 预训练视觉 encoder（或直接用现成）。
2. 冻结两端训练 projector，完成表示对齐。
3. 多模态 pre/mid-training，解冻部分/全部，学习跨模态知识。
4. multimodal SFT：视觉问答、OCR、grounding、GUI、图表。
5. preference/RL：helpfulness、groundedness、tool/agent actions。

如果一开始全解冻，随机 projector 的噪声可能破坏 LLM；冻结/分阶段减少灾难性遗忘。文本-only batch 仍重要，避免语言能力回退。

## 13.6 评测与幻觉

分类/QA 分数不足。区分：

- perception：是否看到物体、文字、数量、位置；
- grounding：描述是否能对应区域；
- reasoning：基于视觉证据推理；
- knowledge：外部知识；
- robustness：裁剪、旋转、分辨率、对抗文字；
- abstention：看不清时是否承认。

语言先验可在不看图时答对 benchmark。做 image ablation（遮图/换图）、counterfactual pairs、区域标注与 OCR 执行检查。LLM judge 也可能不读图或被图中文字 prompt injection 欺骗。

## 13.7 多模态 Agent

模型读截图/视频，输出点击、键盘、代码或工具调用。训练对象从静态答案扩为 trajectory：observation → action → new observation。需要 coordinate grounding、状态跟踪、长上下文、并行工具、安全权限。

2026 讲义以多模态/agent 作为扩展：Kimi K2.5 等报告联合 text-vision pretraining、zero-vision SFT（用无图样本保持文本能力）、联合 RL，并把任务拆成并行 agent swarm。模型报告的 benchmark 与延迟收益应在相同工具/预算/环境复核；多 agent 提速来自并行，不保证 token/成本更低。

## 13.8 系统代价

视觉 token 拉长 prefill；高分辨率请求长度差异大，dynamic batching 更难。视觉 encoder 可缓存；多轮同图复用 features/KV。视频帧数再乘时间维，需采样、压缩、分层 attention。

容量规划要把 image tokens 显式换算为文本 token 等价、encoder FLOPs、KV bytes；API 只按“每张图”计费会掩盖分辨率差异。

## 13.9 论文思路

### Radford et al., *CLIP* (2021)

从 4 亿 image-text pairs 学对比表示，把类别名文本当分类器实现 zero-shot transfer。证明自然语言监督可扩展视觉概念；网页数据偏见、细粒度生成/计数不是其强项。

### Dosovitskiy et al., *Vision Transformer* (2020)

把图像 patch 当 token，用标准 Transformer encoder；足够数据规模下可与 CNN 竞争。奠定视觉 token 接入 LLM 的基础，代价是 patch attention 随分辨率平方。

### Flamingo / LLaVA 类

Flamingo 用冻结视觉/语言骨干和 gated cross-attention 处理 interleaved multimodal context；LLaVA 用视觉 encoder + projector，并用合成 visual instruction data 做 SFT。共同说明连接器与高质量指令数据可低成本获得强交互，但视觉幻觉与数据依赖明显。

## 13.10 验收

除常规 benchmark，必须做 image ablation/counterfactual、分辨率/长宽比 sweep、OCR/计数/空间分域、文本-only 回归、视觉 token 与 latency/memory 表、包含图片 prompt injection 的安全测试。
