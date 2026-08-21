# 论文地图：讲义提到的工作解决了什么

本表覆盖 2026 `references.py`、PDF 讲义中具名的核心论文/技术报告。正文已详解的论文这里压缩成“问题 → 思路 → 边界”；模型报告只提其最值得学习的设计，不把 benchmark 宣传当结论。建议先读带 ★ 的主干论文。

## A. 语言模型与 Transformer 的历史主线

### 1950-2016

- ★ [Shannon, *Prediction and Entropy of Printed English* (1950)](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)：用人类逐字符猜测估计英语条件熵，把“语言可预测性”变成量；现代 next-token NLL 是规模化版本。
- [Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)：用带门控的 cell state 和近线性梯度通路缓解普通 RNN 长期依赖/梯度消失；仍需按时间串行。
- ★ [Bengio et al., *A Neural Probabilistic Language Model* (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)：embedding + 固定窗口前馈网络 + softmax，使相似词共享统计；受窗口限制。
- [Brants et al., *Large Language Models in Machine Translation* (2007)](https://aclanthology.org/D07-1090.pdf)：在 2T tokens 上训练分布式 5-gram，展示数据规模与传统 count LM；稀疏/固定上下文促使神经方法兴起。
- [Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks* (2010)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)：分析激活/梯度方差随层传播，提出 fan-in/fan-out 初始化；现代 residual Transformer 仍需进一步深度缩放。
- [Duchi et al., *AdaGrad* (2011)](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)：累计平方梯度给稀疏参数更大、频繁参数更小的步长；累计量只增导致后期步长过小。
- [Sutskever et al., *Sequence to Sequence Learning with Neural Networks* (2014)](https://arxiv.org/abs/1409.3215)：encoder 把输入压成向量，decoder 生成输出，统一可变长映射；单向量瓶颈促成 attention。
- ★ [Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2014/2015)](https://arxiv.org/abs/1409.0473)：decoder 每步对 encoder states 软对齐加权，不再只靠一个向量；attention 最初服务跨序列条件生成。
- ★ [Kingma & Ba, *Adam* (2014)](https://arxiv.org/abs/1412.6980)：一/二阶梯度 EMA + 偏置修正，逐参数自适应更新；多两份状态且缩放行为需谨慎。
- ★ [Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016)](https://arxiv.org/abs/1508.07909)：把 BPE 用于 subword，稀有词可组合且序列不过长；固定词表仍有语言/域偏差。
- ★ [Ba et al., *Layer Normalization* (2016)](https://arxiv.org/abs/1607.06450)：每样本跨 hidden 维归一，不依赖 batch statistics，适合序列/RNN；均值/方差与 affine 带额外计算。

### 2017-2020

- ★ [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762)：用多头 self-attention 完全替代 recurrence，实现序列并行和短信息路径；dense attention 随长度平方。
- [Loshchilov & Hutter, *SGDR* (2017)](https://arxiv.org/abs/1608.03983)：cosine LR 与 warm restart；LLM 常保留 warmup+cosine 而不 restart。
- ★ [Loshchilov & Hutter, *AdamW* (2017)](https://arxiv.org/abs/1711.05101)：把 weight decay 从 Adam 梯度预条件中解耦，让参数收缩语义一致。
- ★ [Schulman et al., *PPO* (2017)](https://arxiv.org/abs/1707.06347)：clip importance ratio 限制 policy 大步变化，在易实现与稳定间折中；不是真正 KL 保证。
- ★ [Shazeer et al., *Sparsely-Gated Mixture-of-Experts* (2017)](https://arxiv.org/abs/1701.06538)：router 每 token 激活少数 FFN experts，以近固定 active compute 扩总参数；负载、all-to-all、路由稳定成为新瓶颈。
- [Peters et al., *ELMo* (2018)](https://arxiv.org/abs/1802.05365)：双向 LSTM 预训练的上下文表示迁移下游，推动通用预训练表示。
- ★ [Devlin et al., *BERT* (2018)](https://arxiv.org/abs/1810.04805)：masked LM 双向预训练 + task fine-tuning，奠定 encoder foundation model；不适合直接自回归生成。
- [Huang et al., *GPipe* (2018/2019)](https://arxiv.org/abs/1811.06965)：层分 stage、microbatch pipeline 与重计算，跨设备训练大网络；有 bubble/负载平衡问题。
- [McCandlish et al., *An Empirical Model of Large-Batch Training* (2018)](https://arxiv.org/abs/1812.06162)：用 gradient noise scale 描述 critical batch，解释 batch 增大后的收益饱和。
- [Child et al., *Sparse Transformer* (2019)](https://arxiv.org/abs/1904.10509)：固定 factorized/local 稀疏模式把 attention 从平方降到次平方，同时保持跨区路径；pattern 可能漏任务需要的连接。
- [Radford et al., *GPT-2* (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)：更大 decoder LM 在网页数据上显示 zero-shot task behavior；WebText 数据/阶段发布也成为开放与风险案例。
- [Gokaslan & Cohen, *OpenWebText* (2019)](https://skylion007.github.io/OpenWebTextCorpus/)：复现 WebText 思路，用 Reddit 高票外链构建开放网页语料；选择偏差与抓取/许可仍在。
- ★ [Raffel et al., *T5 / C4* (2019/2020)](https://arxiv.org/abs/1910.10683)：把任务统一成 text-to-text，并系统研究架构/目标/数据；C4 展示 Common Crawl 规则清洗，也继承启发式偏差。
- ★ [Shoeybi et al., *Megatron-LM* (2019)](https://arxiv.org/abs/1909.08053)：按矩阵行/列和 attention heads 做 tensor parallel，配对减少通信，适合节点内高速互连。
- ★ [Rajbhandari et al., *ZeRO* (2019/2020)](https://arxiv.org/abs/1910.02054)：逐级分片 optimizer/gradient/parameter，移除 data parallel 状态复制；以更复杂通信换容量。
- [Zhang & Sennrich, *RMSNorm* (2019)](https://arxiv.org/abs/1910.07467)：只按 RMS 缩放、不减均值，减少 LayerNorm 计算；现代 LLM 常用。
- ★ [Kaplan et al., *Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361)：跨规模观察 loss 对参数/数据/计算的平滑幂律，推动小实验外推；compute-optimal N/D 后被 Chinchilla 修正。
- ★ [Shazeer, *GLU Variants Improve Transformer* (2020)](https://arxiv.org/abs/2002.05202)：系统比较门控 FFN，SwiGLU 在匹配预算下成为强默认；主要是经验规律。
- [Xiong et al., *On Layer Normalization in the Transformer Architecture* (2020)](https://arxiv.org/abs/2002.04745)：用初始化梯度分析解释 post-norm 需 warmup、pre-norm 更稳；最终质量仍受深度/recipe 影响。
- [Katharopoulos et al., *Transformers are RNNs / Linear Attention* (2020)](https://arxiv.org/abs/2006.16236)：用 feature map 分解相似度，先聚合 $K^TV$，得到线性训练/递归推理；softmax 表达与实际 kernel 是边界。
- [Beltagy et al., *Longformer* (2020)](https://arxiv.org/abs/2004.05150)：滑窗局部 attention + 少量全局 token，处理长文档；远程信息依赖层级传播/全局选择。
- ★ [Brown et al., *GPT-3* (2020)](https://arxiv.org/abs/2005.14165)：175B decoder LM 展示 in-context few-shot，规模与多样网页带来任务泛化；训练数据/成本、偏差和欠训练问题突出。
- [Gao et al., *The Pile* (2020/2021)](https://arxiv.org/abs/2101.00027)：22 个领域的开放混合语料，为开源大模型提供数据基础；来源异质性要求许可/质量/去重治理。

## B. 2021-2022：规模、开放模型与现代默认组件

- ★ [Hendrycks et al., *MMLU* (2021)](https://arxiv.org/abs/2009.03300)：57 学科多选题测知识/问题求解；覆盖广但格式、静态题库与污染限制解释。
- ★ [Su et al., *RoFormer / RoPE* (2021)](https://arxiv.org/abs/2104.09864)：按位置旋转 Q/K，让点积依赖相对位移；长度外推仍需频率处理。
- [Narayanan et al., *Efficient Large-Scale Language Model Training* (2021)](https://arxiv.org/abs/2104.04473)：组合 data/tensor/pipeline parallel，并按拓扑/成本模型选 3D 配置；说明多维并行是大规模必需。
- [Xue et al., *ByT5* (2021)](https://arxiv.org/abs/2105.13626)：T5 直接处理 UTF-8 bytes，消除 tokenizer/OOV 并增强噪声鲁棒；序列更长使算力上升。
- [Wang & Komatsuzaki, *GPT-J* (2021)](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/)：6B 级开放 GPT 复现，采用并行 attention/FFN 等工程选择，证明社区可训练可用大模型。
- [Rae et al., *Gopher* (2021)](https://arxiv.org/abs/2112.11446)：280B 模型与 MassiveText，系统分析规模、数据与多任务；后来被更多 token 的较小 Chinchilla 超越。
- ★ [Fedus et al., *Switch Transformers* (2021/2022)](https://arxiv.org/abs/2101.03961)：把 MoE 简化成 top-1 routing，研究 capacity、load loss、低精度稳定；容量扩大但 expert parallel 系统复杂。
- ★ [Ouyang et al., *InstructGPT* (2022)](https://arxiv.org/abs/2203.02155)：demonstration SFT→preference reward model→PPO+KL，较小 aligned 模型赢人类偏好；RM 代理与标注分布是边界。
- ★ [Hoffmann et al., *Chinchilla* (2022)](https://arxiv.org/abs/2203.15556)：fixed-N、IsoFLOP、联合 loss 三路拟合，说明同计算下应更均衡扩 N 与 tokens；70B 多训数据胜更大欠训练模型。
- [Chowdhery et al., *PaLM* (2022)](https://arxiv.org/abs/2204.02311)：540B 模型在 6144 TPUv4 上展示大规模 Pathways、SwiGLU、MQA/并行块与 MFU 报告；也是“参数巨大但数据相对少”的时代案例。
- [Black et al., *GPT-NeoX-20B* (2022)](https://arxiv.org/abs/2204.06745)：公开模型、数据（Pile）和训练系统 recipe，含 RoPE、并行 attention/FFN；重在开放复现。
- [Zhang et al., *OPT-175B* (2022)](https://arxiv.org/abs/2205.01068)：公开 GPT-3 规模复现及训练日志，详细记录硬件故障、FSDP/Megatron 与 fp16 稳定问题；工程失败信息极有价值。
- [Le Scao et al., *BLOOM* (2022)](https://arxiv.org/abs/2211.05100)：BigScience 多机构训练 176B 多语言模型，强调 ROOTS 数据治理与开放协作；展示超算/ZeRO-1 大训练。
- [Bahdanau, *The FLOPs Calculus of Language Model Training* (2022)](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)：从矩阵乘推 Transformer 参数/FLOPs，帮助理解 $6ND$ 近似；属教学推导，需按实际架构修正。
- ★ [Yang et al., *Tensor Programs V / μTransfer* (2022)](https://arxiv.org/abs/2203.03466)：μP 通过宽度相关初始化/LR 参数化，使小模型超参迁移大宽度；实现约定敏感。
- ★ [Dao et al., *FlashAttention* (2022)](https://arxiv.org/abs/2205.14135)：把瓶颈定位为 HBM I/O，用 tiling+online softmax 算 exact attention，不保存 $T^2$ 中间；数学 FLOPs 仍平方。
- [Lee et al., *Deduplicating Training Data Makes Language Models Better* (2021/2022)](https://arxiv.org/abs/2107.06499)：exact/near dedup 降低记忆、污染和 token 浪费；阈值/代表样本选择决定误删风险。

## C. 2023：Llama 时代、推理效率与直接偏好优化

- ★ [Touvron et al., *LLaMA* (2023)](https://arxiv.org/abs/2302.13971)：只用公开数据训练 7B-65B，采用 pre-norm/RMSNorm/SwiGLU/RoPE，强调推理预算下较小模型多训 tokens；形成现代 dense 开源 recipe。
- [OpenAI, *GPT-4 Technical Report* (2023)](https://arxiv.org/abs/2303.08774)：展示多模态/考试与可预测 scaling，但不披露架构、数据和算力细节；正是 CS336 强调开放底层的动机。
- [Taori et al., *Alpaca* (2023)](https://crfm.stanford.edu/2023/03/13/alpaca.html)：用 text-davinci 生成 52K instruction 数据微调 LLaMA，低成本复现 instruction following；teacher、许可和评测偏差限制明显。
- [EleutherAI, *Transformer Math 101* (2023)](https://blog.eleuther.ai/transformer-math/)：系统推导参数、FLOPs、激活/KV/训练显存，是资源核算实践参考；具体常数取决于架构/并行。
- ★ [Ainslie et al., *GQA* (2023)](https://arxiv.org/abs/2305.13245)：在 MHA 与单 KV 头 MQA 之间用若干 KV groups，通过 uptraining 获得近 MHA 质量和近 MQA decode 成本。
- [Zhou et al., *LIMA* (2023)](https://arxiv.org/abs/2305.11206)：强 base model 用约 1K 高质量示范就能学交互风格，提出“表面对齐”；不代表覆盖/安全只需少数据。
- [Yu et al., *MEGABYTE* (2023)](https://arxiv.org/abs/2305.07185)：byte 序列分 patch，用全局模型跨 patch、局部模型内生成，降低 tokenizer-free 长序列成本；层级边界和实现复杂。
- ★ [Rafailov et al., *DPO* (2023)](https://arxiv.org/abs/2305.18290)：把 KL-regularized RL 最优 policy 代入偏好模型，直接优化 policy/reference log-ratio；省 RM/PPO，但受离线 preference 覆盖限制。
- [Touvron et al., *Llama 2* (2023)](https://arxiv.org/abs/2307.09288)：2T tokens、GQA（70B）与 SFT/RLHF 安全 recipe，推动开放 chat 模型；训练数据细节仍不完全开放。
- ★ [Dao, *FlashAttention-2* (2023)](https://arxiv.org/abs/2307.08691)：减少非 matmul FLOPs、改 sequence/warp 并行，使 exact attention 更接近 GEMM 峰值；收益依硬件/shape。
- [Jiang et al., *Mistral 7B* (2023)](https://arxiv.org/abs/2310.06825)：GQA + sliding-window attention，在较小模型上强调推理效率；远程精确依赖受窗口限制。
- [Dehghani et al., *Scaling Vision Transformers* / 视觉架构经验 (2023)](https://arxiv.org/abs/2302.05442)：Lecture 3 用 QK normalization 相关工作说明对 Q/K 规范可控制 attention logits、提高大模型稳定；不同论文实现为 per-head norm/scale，需读具体公式。
- [Gu & Dao, *Mamba* (2023)](https://arxiv.org/abs/2312.00752)：输入依赖的 selective SSM 加硬件友好 scan，兼顾并行训练和 constant-state decode；精确检索/状态容量是核心边界。

## D. 2024：开放全栈、MoE、数据与线性序列模型

### 模型与架构

- [Llama 3 Herd of Models (2024)](https://arxiv.org/abs/2407.21783)：把公开 recipe 扩到 405B/15T+ tokens，详述 data filtering、scaling、并行和多阶段 post-training；模型强度来自数据/规模/后训练整体，不是架构突变。
- [DeepSeek LLM (2024)](https://arxiv.org/abs/2401.02954)：在中英 2T tokens 上研究 model/data、LR、batch scaling，再训练 7B/67B；强调不同数据分布 scaling law 不同。
- [Jiang et al., *Mixtral of Experts* (2024)](https://arxiv.org/abs/2401.04088)：稀疏 top-2 MoE，把每层多个 FFN experts 中只激活两个，以较低 active compute 获大容量；权重/KV/路由服务成本仍高。
- ★ [Groeneveld et al., *OLMo* (2024)](https://arxiv.org/abs/2402.00838)：公开数据、代码、训练日志、checkpoints 和评测，使用 Dolma/FSDP；最大贡献是全栈可审计科学。
- ★ [Shao et al., *DeepSeekMath / GRPO* (2024)](https://arxiv.org/abs/2402.03300)：数学语料与组相对 policy optimization，省 value model；组归一、长度与 verifier 细节后来被继续审视。
- [NVIDIA, *Nemotron-4 15B* (2024)](https://arxiv.org/abs/2402.16819)：8T multilingual/code 数据、GQA、squared ReLU 与后期高质量 mix，展示长 token training 与硬件 recipe。
- [Young et al., *Yi* (2024)](https://arxiv.org/abs/2403.04652)：开放 6B/34B bilingual 模型并讨论数据/训练/长上下文扩展；作为讲义架构比较样本。
- [Gemma Team, *Gemma* (2024)](https://arxiv.org/abs/2403.08295)：2B/7B 开放权重，采用 MQA/RoPE/GeGLU/RMSNorm，在 TPU 上训练；展示不同规模的 KV/激活选择。
- ★ [DeepSeek-V2 / MLA (2024)](https://arxiv.org/abs/2405.04434)：低维 latent KV cache + MoE，显著降 decode cache/成本；RoPE 解耦、矩阵吸收和 kernel/并行复杂。
- ★ [Dao & Gu, *Transformers are SSMs / Mamba-2* (2024)](https://arxiv.org/abs/2405.21060)：用 state-space duality 把结构化状态模型与注意力矩阵联系，提出更适合矩阵硬件的 SSD 层；固定状态仍限制某些任务。
- [Tencent, *Hunyuan-Large* (2024)](https://arxiv.org/abs/2411.02265)：大规模 MoE bilingual 技术报告，覆盖路由、训练/长上下文/后训练；用于比较工业 MoE recipe。
- [Qwen Team, *Qwen2.5* (2024)](https://arxiv.org/abs/2412.15115)：多尺寸 dense/MoE、18T tokens、代码数学与长上下文系列，体现完整 base→instruction 家族；各子模型 recipe 需区分。
- [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437)：MLA、无辅助损失 MoE、multi-token prediction、FP8 与大规模系统协同；质量/成本来自整套共同设计，不能归因单一组件。
- [Gated Delta Networks (2024)](https://arxiv.org/abs/2412.06464)：结合 gating 的快速遗忘与 delta rule 的定向 memory 更新，配并行算法，改善线性模型的检索/状态跟踪；混合 local attention 常更强。

### 数据、tokenization 与训练目标

- ★ [Soldaini et al., *Dolma* (2024)](https://arxiv.org/abs/2402.00159)：3T token 开放语料与过滤/mixing 工具，做语言、PII、毒性、去污染和 data ablation；强调 lineage 与偏差审计。
- [Maini et al., *WRAP* (2024)](https://arxiv.org/abs/2401.16380)：让 instruction LM 把噪声网页重写成 Wikipedia/QA 风格，与原文共同预训练，提高数据效用；teacher 风格、事实改写和合成偏差是风险。
- ★ [Li et al., *DCLM* (2024)](https://arxiv.org/abs/2406.11794)：固定模型/计算/评测，把 Common Crawl data curation 变可比较 benchmark；model-based quality filter 是强基线。
- [T-FREE (2024)](https://arxiv.org/abs/2406.19223)：用字符 trigram 稀疏激活直接表示词，免训练 subword tokenizer、压 embedding/head 并改善跨语言；生成/硬件生态仍需验证。
- [Pagnoni et al., *Byte Latent Transformer* (2024)](https://arxiv.org/abs/2412.09871)：按 byte 局部熵动态 patch，高熵区细、低熵区粗，把算力按信息密度分配；patcher/全局-局部交互复杂。
- [NVIDIA, *Nemotron-CC* (2024)](https://arxiv.org/abs/2412.02595)：从 Common Crawl 做分类、去重和高质量合成重述以增高质量 tokens；合成/筛选模型偏差需审计。
- [Gloeckle et al., *Multi-token Prediction* (2024)](https://arxiv.org/abs/2404.19737)：共享 trunk 上多个 heads 同时预测多个未来 token，改善样本效率/代码，并可作 speculative decoding；额外 heads 与目标权重需设计。
- [Maini et al., *RegMix* (2024/2025)](https://arxiv.org/abs/2407.01492)：随机 mixtures 训练小 proxy，以回归预测候选 mix，减少大模型 mixture search；迁移受模型规模/域集合变化限制。

### 优化、scaling 与系统

- ★ [Gemstones / *Overtrained LMs* (2024)](https://arxiv.org/abs/2403.08540)：把推理需求纳入 compute-optimal，说明较小模型多训远超 Chinchilla tokens 可能降低生命周期成本；目标从单次 train loss 变总成本。
- ★ [Hu et al., *MiniCPM / WSD* (2024)](https://arxiv.org/abs/2404.06395)：μP + scaling + warmup-stable-decay，稳定轨迹复用不同预算 decay，降低超参/scale 探索成本。
- [Jiang et al., *MegaScale* (2024)](https://arxiv.org/abs/2402.15627)：在万卡规模组合 data/tensor/pipeline/sequence parallel 与网络优化，报告高 MFU；强调 topology、故障与调度的共同作用。
- [SOAP (2024)](https://arxiv.org/abs/2409.11321)：把 Adam 放到 Shampoo 预条件器 eigenbasis，兼顾二阶矩阵几何与持续二阶矩更新；更少 step 但有 eigendecomposition/额外超参成本。
- [Keller Jordan, *Muon* (2024)](https://kellerjordan.github.io/posts/muon/)：对 hidden matrix momentum 做 Newton-Schulz 正交化，改善矩阵方向更新；embedding/head 另用 AdamW，墙钟/规模迁移需实测。
- [Wang et al., *Auxiliary-Loss-Free Load Balancing* (2024)](https://arxiv.org/abs/2408.15664)：按近期 expert load 动态调 routing score bias，不用 auxiliary gradient 干扰主目标；bias 更新稳定与分布变化需调。

## E. 2025-2026：reasoning、动态 tokenization、混合模型与 Agent

- [OLMo 2 (2025)](https://arxiv.org/abs/2501.00656) 与 [OLMo 2 32B](https://allenai.org/blog/olmo2-32B)：改训练稳定、数据与 staged curriculum 的开放模型系列，并把 recipe 扩到 32B；2026 A5 用其 1B base 展示可审计 RL。
- ★ [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)：cold-start、多阶段 RL/SFT、可验证 reward 和蒸馏，展示长 CoT reasoning；不是单一 GRPO 算法成果。
- [Kimi k1.5 (2025)](https://arxiv.org/abs/2501.12599)：用长上下文 RL、课程/采样与 policy optimization 提升多模态 reasoning；训练细节/预算共同决定结果。
- [SmolLM2 (2025)](https://arxiv.org/abs/2502.02737)：1.7B 小模型通过 FineMath/Stack-Edu 等高质量数据和 staged training 提升，说明小模型尤其受数据 mix 影响。
- [Meta, *Llama 4* (2025)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)：原生多模态 MoE 家族与长上下文/蒸馏，作为视觉-文本联合训练案例；公开程度低于 OLMo/Marin。
- [Qwen3 (2025)](https://arxiv.org/abs/2505.09388)：dense/MoE、多语言，统一 thinking/non-thinking 与预算控制；后训练和推理模式是主线。
- [H-Net (2025)](https://arxiv.org/abs/2507.07955)：端到端学习内容/上下文依赖的动态 byte chunks，层级模型替代 tokenize-LM-detokenize；在中文、代码、DNA 显示潜力。
- [GLM-4.5 (2025)](https://arxiv.org/abs/2508.06471)：MoE 与 reasoning/agent/coding 多阶段训练报告，用作现代 agentic 模型 recipe 案例。
- [LongCat-Flash (2025)](https://arxiv.org/abs/2509.01322)：围绕高效 MoE/长上下文和 serving 的技术报告；讲义用来展示现代架构越来越由推理效率驱动。
- [Ling 2.0 (2025)](https://arxiv.org/abs/2510.22115)：混合线性 attention/MoE 类开放报告，体现 attention/recurrence 混合趋势；具体优势需按硬件/任务复核。
- [Marin 8B](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/) / [32B retrospectives (2025)](https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/)：开放开发过程、失败和 scaling/数据/系统复盘；价值在过程透明，不只最终 checkpoint。
- [OLMo 3 (2025)](https://arxiv.org/abs/2512.13961)：延续全栈开放，把数据、预/中/后训练与推理模型 artifacts 公开；适合因果追踪 recipe。
- [Nemotron 3 (2025)](https://arxiv.org/abs/2512.20856)：NVIDIA 开放/混合 Mamba-Transformer-MoE 系列，优化 agentic reasoning 的效率。
- [DeepSeek-V3.2 (2025)](https://arxiv.org/abs/2512.02556)：进一步结合稀疏/长上下文、reasoning/agent 后训练；Lecture 2 用来说明低精度/现代系统设计共同演进。
- [Kimi K2.5 (2026)](https://arxiv.org/abs/2602.02276)：联合 text-vision pretraining、zero-vision SFT、联合 RL，并用 Agent Swarm 并行分解任务；并行降延迟不等于降总成本。
- [GLM-5 (2026)](https://arxiv.org/abs/2602.15763)：现代 MoE/agentic reasoning 报告，讲义作为开放权重接近闭源模型的例子；应区分技术报告声明与独立复现。
- [Arcee Trinity (2026)](https://arxiv.org/abs/2602.17004)：开放 MoE 模型/数据或训练 recipe 的新案例，体现小团队通过数据与稀疏架构进入大模型训练。
- [Qwen3.5 (2026)](https://qwen.ai/blog?id=qwen3.5)、[MiniMax M2.5](https://www.minimax.io/news/minimax-m25)、[Xiaomi MiMo-V2](https://mimo.xiaomi.com/mimo-v2-pro)：Lecture 1 的当前开放模型版图例子；共同趋势是 reasoning、agent、MoE/高效推理，博客指标要以独立评测核验。
- ★ [Mamba-3 (2026)](https://arxiv.org/abs/2603.15569)：更有表达力的 SSM 离散化、complex state 与 MIMO，在 constant-memory decode 下改善状态跟踪/质量；仍需与强 attention hybrid 比。
- ★ [Olmix (2026)](https://arxiv.org/abs/2602.12237)：系统研究 data mixing 配置，并在域集合增删时复用旧 mixture，只重算受影响域；把数据开发的动态现实纳入算法。
- [Nemotron 3 Super (2026)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)：hybrid Mamba-Transformer MoE 面向 agentic reasoning，在质量、active compute、长上下文间折中。
- [DeepSeek-V4 (2026)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)：Lecture 10 用其百万 token inference 设计讨论 local/global/latent attention 与 cache；属于新报告，结论应按具体公开版本与硬件验证。

## F. PDF 讲义中的补充思想

- [Tay et al., *Do Transformer Modifications Transfer Across Implementations and Applications?* / architecture studies](https://arxiv.org/abs/2102.11972)：大规模架构改动常不能跨代码/任务稳定迁移，支持“强 baseline + matched budget 消融”。
- [Narang et al., *Do Transformer Modifications Transfer Across Implementations and Applications?* (2021)](https://arxiv.org/abs/2102.11972)：系统复测多种 Transformer 改进，许多收益不稳；不要 cargo-cult 小论文组件。
- [Zoph et al., *ST-MoE* (2022)](https://arxiv.org/abs/2202.08906)：研究稀疏 MoE 稳定、迁移和 fine-tuning，router z-loss/训练技巧缓解 logits/路由问题。
- [Clark et al., *Scaling Laws for Autoregressive Generative Modeling* / MoE routing analyses](https://arxiv.org/abs/2202.01169)：Lecture 4 用于说明 router baseline/load 与专家专门化的实证分析；路由质量不能只看负载均匀。
- [Hestness et al., *Deep Learning Scaling is Predictable, Empirically* (2017)](https://arxiv.org/abs/1712.00409)：多个任务的 error 随数据呈 power law，是 LM scaling 前史；可预测区间有限。
- [Bahri et al., *Explaining Neural Scaling Laws* (2021)](https://arxiv.org/abs/2102.06701)：从数据流形/方差等理论解释幂律为何常见；理论假设不能替代实际外推验证。
- [Besiroglu et al., *Chinchilla Scaling: A Replication Attempt* (2024)](https://arxiv.org/abs/2404.10102)：重查 Chinchilla 拟合数据/不确定性，说明 scaling 系数并非精确常数，应报告区间与敏感性。
- [Lee-Thorp et al./Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022)](https://arxiv.org/abs/2205.05198)：sequence parallel 与 selective recomputation 降激活显存/重算，把 tensor parallel 冗余激活再切分。
- [Bai et al., *Constitutional AI* (2022)](https://arxiv.org/abs/2212.08073)：原则驱动 self-critique/revision，再用 AI preference 对齐，扩展安全监督；原则和 judge 偏差仍需人审。
- [Bubeck et al., *Sparks of AGI* (2023)](https://arxiv.org/abs/2303.12712)：用大量定性任务描述 GPT-4 能力，Lecture 15 用来展示 instruction control；不是透明训练/严格 benchmark 证据。
- [Gekhman et al., *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* (2023/2024)](https://arxiv.org/abs/2405.05904)：区分 SFT 学格式/行为与注入新事实，发现低熟悉知识易诱发幻觉；支持知识更多放预/检索、SFT 聚焦行为。
- [Stiennon et al., *Learning to Summarize with Human Feedback* (2020)](https://arxiv.org/abs/2009.01325)：早期完整 RLHF pipeline，公开偏好标注和 reward/PPO 细节；展示代理过优化与人评必要。
- [Cobbe et al., *GSM8K* (2021)](https://arxiv.org/abs/2110.14168)：8.5K 小学数学 word problems 与自然语言解答，答案可验证、适合 CoT/RLVR；题型窄且易污染/格式拟合。
- [Li et al., *AlpacaEval* (2023)](https://arxiv.org/abs/2305.14387)：用强 LLM judge 自动比较 instruction responses，低成本但有长度、位置、judge 偏差。
- [Vidgen et al., *SimpleSafetyTests* (2024)](https://arxiv.org/abs/2311.08370)：小而清晰的安全行为检查，适合 regression；覆盖远非完整威胁模型。

## G. 怎样用这张地图

若时间有限，按顺序精读：Bengio 2003 → Transformer → GPT-3 → Chinchilla → LLaMA → FlashAttention → ZeRO/Megatron → Dolma/DCLM → InstructGPT/DPO → DeepSeekMath/R1。其余先把“改了栈的哪一层、换来什么、代价什么”说清，再在项目遇到对应瓶颈时深读。
