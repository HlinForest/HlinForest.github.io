# 02 语言建模与 Tokenization：文字怎样变成模型能算的整数

对应官方 Lecture 1 与 Assignment 1 前半。

## 2.1 Tokenizer 不是无损压缩之外的“小工具”

模型只处理固定词表中的整数。tokenizer 定义：

$$\text{encode}: \text{string}\to [0,V)^*,\qquad
\text{decode}: [0,V)^*\to \text{string}.$$

它同时决定：

- 同一上下文窗口能覆盖多少原始文本；
- embedding/output head 的参数和计算量；
- 不同语言、代码、数字、空格与稀有字符的效率；
- 模型一次决策的粒度和序列长度；
- loss/PPL 的单位以及可比性；
- 文档边界、聊天控制 token 和工具协议能否被可靠表达。

理想 tokenizer 要无损、覆盖任意输入、压缩率高、训练/编码快、特殊 token 可控、不同语言不过度失衡。不存在同时最优的方案。

## 2.2 Unicode、code point、UTF-8 byte

Python `str` 是 Unicode code point 序列；文件通常用 UTF-8 bytes 存储。一个“看起来的字符”可能由多个 code point 组成，例如字母加组合音标、emoji 加肤色/连接符。`len(text)` 不是人眼字形数，也不是 UTF-8 byte 数。

UTF-8 的关键性质：

- ASCII 占 1 byte，其他字符占 2-4 bytes；
- 任意 Unicode 字符串可确定地编码为 bytes；
- 256 种 byte 是有限且完整的基本字母表。

byte-level tokenizer 从 256 个 byte token 开始，因此不会出现真正的 OOV。decode 时先拼 bytes 再 UTF-8 解码；单个 token 的 bytes 可能不是合法独立 UTF-8，只有完整序列才合法。

应测试：

```python
assert decode(encode(s)) == s
```

测试集要包含空串、中文、emoji、组合字符、换行、NUL、不同 Unicode 规范化形式和随机 bytes 可解码样本。

## 2.3 为什么不用“词”或“字符”

- **词级**：词表巨大、形态变化和新词造成 OOV，多语言分词困难。
- **字符/code point 级**：覆盖词表仍大，常用片段不能压缩，序列长。
- **纯 byte 级**：词表只有 256 且最稳健，但英文单词/中文字符会变成多步，attention 和生成成本高。
- **subword**：从 byte/字符出发，把频繁相邻片段合并；在覆盖率、词表和序列长度间折中。

当模型/硬件固定时，更好的压缩率能把更多语义放进同一 $T$，但更大词表会增加 embedding 和 output projection（约 $VD$ 参数），低频大 token 也更难学。

## 2.4 Byte Pair Encoding 训练

现代 byte-level BPE 的概念算法：

1. 把语料按 special token 边界切开；special token 永远视为原子。
2. 用 regex 预分词，防止合并跨越不希望跨越的边界。
3. 每个预词转成 UTF-8 bytes；初始符号是单 byte。
4. 统计所有相邻 symbol pair 的频数。
5. 选最高频 pair $(a,b)$；固定 tie-break 规则。
6. 新建 token $a\Vert b$，在所有预词中把不重叠的该 pair 合并。
7. 重复到词表达到目标大小。

若初始 256 bytes、$S$ 个 special token、做 $M$ 次 merge，则 $V=256+S+M$。

### 一个手工例子

语料预词计数：`low:5, lower:2, newest:6, widest:3`。初始是 bytes/字符。若 `(e,s)` 频率最高，先把每处 `e s` 合并成 `es`；下一轮统计的 pair 已改变。BPE 学到的是**有顺序的 merges**，编码必须按训练时优先级应用，不能仅用最终词表做最长匹配来假装等价。

### 预分词 regex 在做什么

类似 GPT 系列的 regex 会把：单词、带前导空格的片段、数字、标点、空白分开。它限制 merge 不跨预词，从而：

- 保留常见的“空格+词”模式；
- 避免整段文本被奇怪跨词片段吞并；
- 让训练可按预词类型/计数聚合。

regex 是 tokenizer 规格的一部分，改它等于换 tokenizer。不同 Unicode regex 引擎、换行处理也可能改变结果。

### special token 必须优先处理

`<|endoftext|>` 既可能是控制 token，也可能作为普通字符序列出现在文本中。编码 API 应明确 `allowed_special`/`disallowed_special` 策略。不能先普通 regex/BPE 再“寻找”special，因为它可能早已被拆分；也不能让普通 merge 跨过文档边界。

## 2.5 从正确版本到可用版本

朴素实现每轮扫描整个语料重计 pair，复杂度很差。公开作业 write-up 中最有价值的经验不是某段代码，而是数据结构演化：

1. 先把相同预词聚合成 `(symbol_tuple -> count)`，避免对重复字符串重复工作。
2. 建 `pair -> 包含该 pair 的预词集合/位置` 倒排索引。
3. merge 后只更新受影响位置的左/右邻 pair 计数。
4. 最大堆可做 lazy deletion：堆顶频数若与当前表不一致就丢弃。
5. tie-break 必须确定，不能让 set/dict 遍历顺序决定 merges。
6. 并行化优先放在独立文档/预分词计数；全局 merge 顺序仍需一致。

一份公开实验记录显示，优化相邻 pair 更新、再加多进程和内存优化后，OpenWebText 子集上的 BPE 训练从数百秒/数 GB 降到约一分钟/几十 MB。这个数字依赖机器和实现，不能当基准；真正可迁移的结论是：profile 后会发现反复全量扫描和复制 tuple 才是瓶颈，而不是 regex 本身。

### 并行边界的陷阱

若简单按 byte offset 切大文件，可能：

- 从 UTF-8 多字节字符中间切开；
- 从 special token 中间切开；
- 把一个预词拆成两份，改变 pair 计数。

安全方案通常在文档分隔 special token 处切 chunk，或让相邻 chunk 有重叠并明确去重边界。先证明合并后的计数等价于单进程版本。

## 2.6 BPE 编码与解码

编码每个预词时，初始 symbols 是 bytes，然后按 merges rank 反复合并当前仍相邻的最低 rank pair，最后查 token ID。可用 linked list + heap 加速；学习实现也可先用 list 扫描，确保正确。

几个不变量：

- `decode(encode(x)) == x`；
- encode 不跨 special/document boundary；
- 同输入、同配置结果确定；
- streaming encode 与整文件 encode 在约定边界上等价；
- 所有输出 ID 都在 `[0,V)`。

decode 是 `id -> bytes` 拼接后再解码。对 API 输入非法 token ID、截断在半个字符处的 token 流，要规定报错还是 replacement character，不能默默随机行为。

## 2.7 训练语料怎样影响 tokenizer

tokenizer 也是数据模型：

- 在英文网页上训练，中文可能接近 UTF-8 byte 粒度，token/字符比很差；
- 代码 tokenizer 会学缩进、运算符和常见标识片段；
- 数字若合成不规则长 token，算术泛化可能受影响；
- 训练语料里的脏 HTML、重复模板会浪费 merge 配额；
- 词表越大并非越好，要同时量序列压缩和 embedding/head 成本。

报告至少包含：每语言 bytes/token、characters/token、未知/回退行为、长尾 token 频率、最长 token、特殊 token 测试、训练与编码吞吐。

## 2.8 无 tokenizer 或动态 tokenization

讲义提到的路线可理解为“固定 subword 是否必须”：

- **ByT5 (2021)**：T5 直接处理 UTF-8 bytes，免 OOV 且对噪声稳健；代价是序列更长，必须让模型更高效地处理 byte 流。
- **MEGABYTE (2023)**：用局部模型处理 byte patch、全局模型处理 patch 表示，分层降低长 byte 序列成本。
- **BLT (2024)**：按字节流的局部熵动态决定 patch 边界；可预测区域用大 patch，困难区域细分，算力随信息密度分配。
- **T-FREE/H-Net 等**：探索 hashing、层级/动态表示，目标是减轻固定词表的语言不公平和部署僵化。

这些方法的共同问题不是“能否表示文本”，而是能否在当代加速器上同时达到质量、吞吐、长上下文和易训练性。固定 BPE 仍流行，因为它把大量局部压缩预先做掉，工程生态成熟。

## 2.9 论文思路

### Shannon, *Prediction and Entropy of Printed English* (1950)

用人类逐步猜下一个字符来估计英语熵，核心观念是语言的可预测性可以量化。现代 LM 的 next-token loss 正是这种思想的可扩展统计实现；局限是早期实验小且依赖人类猜测分布。

### Bengio et al., *A Neural Probabilistic Language Model* (2003)

把离散词映射为连续向量，用前馈网络根据固定窗口预测下一词；相似词共享统计强度，缓解 n-gram 的组合稀疏。它仍受固定窗口限制，但奠定“embedding + 神经网络 + softmax 最大似然”的模板。

### Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016)

把 BPE 从压缩算法用于 subword，让稀有词可由常见片段组合，兼顾开放词表与较短序列。论文验证翻译中的稀有词处理；今天 byte-level BPE 加入了 byte 基础字母表和更复杂预分词，但核心贪心合并思想相同。

## 2.10 实践检查点

在进入模型前，应交付一份 tokenizer report：

1. 训练配置、语料 hash、regex、tie-break 和 special token 规格。
2. 单元测试与 property-based round-trip 测试。
3. TinyStories 与另一种语言/代码上的压缩率。
4. 最频繁/最长/最怪 token 的人工检查。
5. 朴素与增量实现的 profile；优化前后必须生成相同 merges。

如果 tokenizer 不确定或会跨文档合并，后面的模型实验已经不可复现。
