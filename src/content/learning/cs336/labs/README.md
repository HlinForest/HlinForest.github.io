# 五个综合实验：把知识变成能力

这些实验与官方五份作业同构，但不提供可直接提交的答案。若你在修课，应遵守当期 honor code。自学者也应先独立实现，再把公开实现当作 code review 对象。

## 通用工作流

每个实验建独立环境和 README，按红-绿-优化：

1. 从接口与不变量写测试，确认最初失败。
2. 写清楚但慢的 reference。
3. 小数据过 correctness；逐层比较数值/梯度。
4. profile 后只优化 top bottleneck。
5. 优化版与 reference 做随机/边界 property tests。
6. 固定预算跑实验，生成表/图与 failure log。
7. 写“结论适用范围”，不把单机小实验外推成普遍定律。

公开作业筛选标准与具体来源见 [SOURCES](../SOURCES.md)。综合吸收的是方法：增量 BPE 计数、预热计时、数值 reference、DDP bucket overlap、受控消融、过滤后人工抽查、GRPO 指标拆解；不复制实现。

## Lab 1：从零训练一个小型 Transformer LM

### 目标

实现 byte-level BPE、tokenizer、Linear/Embedding/RMSNorm/RoPE/SwiGLU/MHA/Transformer、cross-entropy、AdamW、schedule、clipping、dataloader、checkpoint 和生成。

### 里程碑

1. Unicode/bytes round-trip；special token atomic。
2. 小语料 BPE merges 与手算一致，tie-break 确定。
3. 模块前向/梯度与 PyTorch reference 对齐。
4. causal invariant：改变未来 token 不影响过去 logits。
5. 过拟合一个 batch。
6. TinyStories 训练出可读文本；报告 val PPL、tokens/s、memory。
7. 受控消融：norm、position、SwiGLU/非门控、LR/batch。

### BPE 性能实验

版本 A 每轮全量重计；版本 B 维护 pair count 与倒排索引；版本 C 并行预分词/计数。三者必须生成完全相同 vocab/merges。报告 CPU、RAM、语料规模、时间分解，不只给总时间。

### Transformer 资源表

手算并由程序核对：参数量（embedding、每层 QKV/O、FFN、norm、head）、每 token forward/training FLOPs、AdamW model-state bytes、主要 activations。误差来源单列。

### 训练实验模板

| run | changed factor | fixed budget | train/val loss | tok/s | peak GB | notes |
|---|---|---|---:|---:|---:|---|

每次只改变一个因素。至少三个 LR；最佳附近再细化。生成使用同 checkpoint、多 seed，禁止只挑最好样本。

### 低资源版

vocab 2k-5k、$D=128$、$L=4$、$T=128$，TinyStories 子集；CPU/MPS/CUDA 都可。目标是正确和趋势，不追 leaderboard。

### 常见失败

BPE 跨 special 合并；RoPE 维/位置错；softmax mask 后整行 NaN；targets 未 shift；accumulation loss 未除；resume 丢 optimizer/RNG；PPL 跨 tokenizer 比。

## Lab 2：从 profiler 到 FlashAttention 与分布式训练

### 目标

建立 benchmark/profiler；activation checkpoint；Triton fused RMSNorm/FlashAttention2；实现 naive→flat→overlapped→bucketed DDP；optimizer state sharding/FSDP 概念实验。

### 先写性能协议

固定 GPU、dtype、shape、软件；warmup；CUDA event；median/p10/p90；forward/backward/step；输出/梯度 tolerance；峰值 memory。cold compile 单独报。

### FlashAttention 里程碑

1. NumPy/PyTorch 小矩阵验证 online softmax 可合并。
2. 分块 PyTorch reference，不 materialize 全 score。
3. Triton forward：非 causal→causal；任意长度 mask。
4. backward 推导与 autograd reference。
5. sweep `T, Dh, causal, dtype`，画 speedup/memory，不只挑优势 shape。
6. end-to-end Transformer step；解释 Amdahl 限制。

### DDP 里程碑

1. 两 rank 固定 batch 与单卡 global reference 一步对齐。
2. naive all-reduce；记录消息数/bytes。
3. flatten 降 latency。
4. autograd hook + async，画 timeline。
5. bucket size sweep，找 overlap/latency 折中。
6. ZeRO-1：state ownership、reduce-scatter/all-gather；核对每 rank bytes。

### 低资源版

无 NVIDIA/Triton 时，完成理论 online softmax、PyTorch 分块 reference、CPU/Gloo 2-process DDP correctness 和通信 accounting；性能结论明确限定。

### 常见失败

异步计时；无预热；只测 kernel 不测模型；causal 三角负载不均；低精度累加；async handle 未 wait；每 microbatch 都同步；rank loss 有效 token 不同却直接平均。

## Lab 3：Scaling law 与大训练决策

### 目标

从一组小模型运行拟合 IsoFLOP，预测更大预算的 $N_{opt},D_{opt},L$，给不确定性和系统可行配置。

### 里程碑

1. 用官方 synthetic JSON 复现 IsoFLOP 流程。
2. 每 compute group 拟合 `loss vs log N`，极小值被两侧点包围。
3. 拟合 $N=aC^\alpha,D=bC^\beta$。
4. bootstrap 与 leave-one-budget-out。
5. 自己跑小 proxy：至少 4 compute groups、每组 5 sizes（低资源可再缩）。
6. 将连续预测离散成层/宽/heads，重新核算参数、FLOPs、显存和墙钟。

### 实验设计得分比拟合代码更重要

先 pilot 找 U 型区间；不同规模 LR sweep；失败/OOM 也入表；相同 tokenizer/data/val；用 final loss；报告外推倍数。比较理论 FLOPs law 与实际 GPU-hours law。

### 低资源版

在合成数据上完成全部统计；真实实验用 1M-100M 参数和较小 token budget，只验证方法。不得把系数用于前沿训练预算。

### 常见失败

每组最优在边界；不调 LR；把 total/embed 参数混用；final/min loss 混用；忽略 attention 平方项/MFU；只报点估计；在 log loss 上错误处理不可约项。

## Lab 4：可审计语料管线

### 目标

把一个 Common Crawl WARC 小样本变成带 lineage 的训练 shards：HTML extraction、language、PII、安全/质量过滤、exact/MinHash dedup、decontamination、mix、tokenize，并用小模型做 data ablation。

### 里程碑

1. WARC streaming，保留 record ID/URL/snapshot/raw hash。
2. 两种 HTML extractor 人工 precision/recall 样本。
3. 语言阈值、PII regex/NER 的正负例集。
4. heuristic/model score 分布与边界人工抽查。
5. exact hash dedup。
6. shingles→MinHash→LSH candidates→真实 Jaccard→clusters；短文档单独处理。
7. benchmark exact/near contamination report。
8. raw/filter/dedup 三份等 token 小模型训练对照。

### 必交流量漏斗

| stage | docs | bytes | tokens | keep % | top removal reasons | domain/lang shift |
|---|---:|---:|---:|---:|---|---|

同时提交 filter/dedup cluster 人工样本，尤其 false positive；不能只展示干净正例。

### 低资源版

取数百到数千网页或官方样本，不下载完整 Common Crawl。数据消融训练极小 LM 或用轻量 quality proxy，重点是 lineage 和统计正确。

### 常见失败

日志泄露原 PII；quality=像 Wikipedia；规范化破坏代码；去重 representative 随并行顺序随机；train/val 先切后去重；MinHash 把短文档误合；只量保留率不训 ablation。

## Lab 5：后训练与推理 RL

### 路线 A：SFT + DPO（通用助手）

1. base zero-shot：知识、推理、对话、安全。
2. chat template 与 role/loss mask 可视化。
3. SFT；比较 base 回归、长度和格式。
4. pairwise preference 数据审计（位置/长度/tie）。
5. DPO 数学小例测试；reference 冻结。
6. beta/LR sweep；监控 log-ratio margin、KL、长度。
7. 多 judge + 人工抽查。

### 路线 B：GRPO/RLVR（数学推理）

1. zero/few-shot/CoT baseline，分 format/answer reward。
2. 同 prompt group sampling，group mean/std/全同率。
3. naive PG→clip objective；手工 ratio 测试。
4. on-policy GRPO，多 seed。
5. 去 std、不同长度 normalization、RFT/Expert Iteration 对照。
6. off-policy sample reuse：policy age、ratio/ESS、clip sweep。
7. 人工审计高 reward，held-out verifier 和 base benchmark 回归。

### 系统报告

把 wall time 分成 rollout、verifier、train、weight sync、idle；报告 rollout/train tok/s、平均/尾 response 长度和 GPU 利用。任何算法提升要说明是否用了更多生成 tokens/墙钟。

### 低资源版

用小模型/少量 prompt 验证 loss 与 mask，SFT/DPO 可 LoRA；GRPO 只做小规模趋势或离线记录 replay。不能从几十题单 seed 得出算法优劣。

### 常见失败

prompt token 进 loss；old policy/logprob 版本错；vLLM 和 trainer template 不同；std=0；长度归一改变目标却未说明；reward parser 可 hack；只看 reward 不看正确率/KL/长度；off-policy 提速但样本 age 无限。

## 最终综合项目：给 1B 模型写训练设计书

假设固定 GPU-hours 与数据预算，提交：

- 目标用户/能力/风险和评测协议；
- 数据来源、清洗、mix、tokenizer；
- 架构尺寸与参数/FLOPs/KV cache；
- scaling 实验与不确定性；
- 并行 mesh、显存、通信、MFU 预测；
- optimizer/schedule/batch/checkpoint/故障策略；
- SFT/RL 数据和目标；
- serving SLO、量化/batching；
- 预算、里程碑、stop/go criteria；
- 最可能失败的五件事和验证实验。

这份设计书能被另一位工程师复现并质疑，才算真正完成 CS336 主线。
