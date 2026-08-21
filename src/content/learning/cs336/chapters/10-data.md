# 10 数据工程：从 Common Crawl 到可审计的训练语料

对应官方 Lecture 13-14 与 Assignment 4。

## 10.1 数据不是模型前的准备，而是模型的一部分

预训练得到的是数据分布、tokenizer、目标与优化过程的联合产物。同参数/FLOPs，数据选择常比小架构改动影响更大。一个可用 pipeline：

```text
source acquisition
-> parse/extract
-> normalize
-> language/domain detection
-> PII/safety/legal policy
-> heuristic/model quality scoring
-> exact + near dedup
-> benchmark decontamination
-> domain mixing/sampling
-> tokenize/pack/shard
-> train small ablations
-> publish lineage/data card
```

顺序会改变结果：先去重再过滤和先过滤再去重，cluster representative 可能不同；先 tokenize 再删除文本代价大；decontamination 必须作用于最终候选语料版本。

## 10.2 Common Crawl：WARC、WAT、WET

- **WARC**：原始 HTTP 响应、URL、headers、HTML/其他内容，最完整也最脏。
- **WAT**：解析出的 JSON 元数据、链接、标题等。
- **WET**：预抽取纯文本，方便但你接受其 extractor 的选择/错误。

要研究自己的 HTML 提取策略，用 WARC。stream `.warc.gz`，按 record 处理，避免解压整个文件；保留 crawl snapshot、WARC record ID、URL、时间与 content hash 作为 lineage。

## 10.3 HTML 到文本

不能只 `BeautifulSoup.get_text()` 就结束。要处理：

- script/style/nav/footer/cookie banner；
- DOM block 边界与换行；
- boilerplate、菜单、评论、广告；
- encoding/header 冲突；
- 非 HTML、压缩/损坏响应；
- 标题、列表、代码、表格是否保结构；
- 极长 DOM/压缩炸弹的资源限制。

比较 extractor 时人工标注一小组页面，量 precision（保留内容中多少是真正文）与 recall（正文丢了多少），再训练小 LM 做最终 data ablation。只看文本长度会奖励垃圾。

## 10.4 规范化要克制

可统一 Unicode 规范、换行、控制字符、重复空白；但过度规范化会破坏代码缩进、诗歌、表格、数学和多语言。保存 raw hash 和每步 transformation version；最好保留 document-level metadata 与 score，不要只输出无法追溯的 `.txt`。

## 10.5 语言识别

文档可能代码混文本、短标题、多语言切换。fastText 类 classifier 输出语言概率，可按 document/paragraph 过滤。阈值提高 precision、降低少数语言/短文本 recall。应人工抽查 threshold 附近、按语言/长度报告 confusion，不把 classifier score 当真值。

多语言模型要按目标分布混合，不一定删除非主语言；temperature sampling 可降低高资源语言垄断：若原比例 $p_i$，用 $q_i\propto p_i^\alpha,\alpha<1$ 提升低资源域。

## 10.6 PII：检测、mask 还是删除

常见 email、phone、IPv4/IPv6、地址、证件/密钥。regex 对 email/IP 可覆盖常见形式，但 phone/地址跨国家极难；误报会破坏普通数字/代码，漏报有隐私风险。

策略：

- mask span 为类型 token，保留上下文结构；
- 删除段落/文档，风险低但丢数据；
- 高风险源整域排除；
- classifier/NER + regex ensemble；
- 保存数量统计，不保存原 PII 日志。

验证用合成正例 + 真实难负例，报告 precision/recall；安全/法律策略应由合适专业人员审查，技术过滤不能保证合规。

## 10.7 “质量”不是客观单一量

heuristics 常见：字符/词数、平均词长、字母比例、标点/符号、重复行、stopword、句末标点、HTML 残留、极端 perplexity。Gopher/C4 类规则便宜可解释，但会偏向规范英语/百科文风，误删方言、代码、聊天和创意文本。

model-based filter：用 Wikipedia/高质量集合做 positive，随机网页做 negative，训练 n-gram/linear 或 LM classifier。风险是把“像 positive 域”当质量，缩窄多样性，并继承标注偏见。DCLM 的重要经验是把 data curation 作为可控竞赛：固定模型与训练计算，只优化数据，并用下游结果选择 filter。

不要只保留最高分：可按 score 分桶抽样，保持多样性；训练小模型做 filter threshold/domain ablation。

## 10.8 Harmful/toxic content

“有害”依任务、语境、身份与政策而变。单词表会误伤讨论/自我保护内容并对少数群体有偏。区分：暴力/色情描述、仇恨、违法指导、医疗错误、骚扰等类别；文档 vs span；训练允许理解的内容 vs 希望模型生成的行为。

pretraining 过滤、post-training 行为对齐、deployment moderation 是不同层。把所有风险交给一个 toxicity classifier 会同时漏报和损害知识覆盖。

## 10.9 Exact dedup

对规范化全文 hash 分组，只留一个。需定义 canonicalization：大小写/空白/Unicode/boilerplate 是否忽略。过强规范化会把不同代码/文本误合；过弱会漏模板变化。

选择 representative 可按来源质量、时间、长度和 license，不要由分布式处理顺序随机决定。split 前全局去重，避免同一文档进入 train/val；但保留来源 cluster metadata 便于审计。

## 10.10 Near dedup：Jaccard、MinHash、LSH

把文档变成 n-gram/shingle 集合 $A,B$：

$$J(A,B)=\frac{|A\cap B|}{|A\cup B|}.$$

全对全比较 $O(n^2)$ 不可行。MinHash 对每个随机 hash/permutation 记录集合最小 hash：

$$P[h_{min}(A)=h_{min}(B)]=J(A,B).$$

用 $k$ 个 minhash 的相等比例估 Jaccard。LSH 把 signature 分成 $b$ bands、每 band $r$ rows；任一 band 完全相同则成为 candidate：

$$P(\text{candidate})=1-(1-s^r)^b.$$

它产生 S-shaped 阈值。对 candidates 计算真实 Jaccard，再用 union-find 连 cluster。

### 工程选择

- shingle 单位：word/byte/character n-gram；
- n 太小误合常见短语，太大漏轻微改写；
- set 忽略重复，multiset/containment 对短长文档可能更合理；
- 短文档 shingles 少，MinHash 方差/误合严重，应单独策略；
- hash seed、位宽、band 参数必须版本化；
- 大 cluster 常是模板/镜像，也可能是合法引用，抽查。

公开作业实现最值得吸收的是把 exact 与 MinHash 分开、对 filtered data 人工 inspect、再训练小模型验证，而不是把“去重率越高”当目标。

## 10.11 为什么去重提升 LM

重复让训练分布对某些文本过度加权，浪费 token budget，增加记忆与 benchmark 泄漏，train/val 也可能假性接近。Lee et al. 2021 展示文本去重能减少记忆、改善或维持下游效果，并发现 C4 中有句子重复数万次。

但重复并非总是垃圾：基础语法、代码 idiom、权威定义自然重复；完全去掉频次信息可能损学习。目标是移除网页模板、镜像、抓取重复和不期望的近副本，而不是让所有 n-gram 只出现一次。

## 10.12 Benchmark decontamination

把 test items 规范化成 token/character n-grams，检索训练文档 exact/near matches。对题目、答案、解释分别检查；代码题需考虑函数名/测试/solution。记录删除率和人工验证。

污染检测本身会 false positive，尤其常识短句。可在训练数据保留但将污染 benchmark 从结论中移除，或报告 clean/contaminated subsets，取决于目标。

## 10.13 Data mixing

不同域 $i$ 设 sampling weight $w_i$。目标不是复制原网页频率，而是优化模型能力/风险/覆盖。方法：

- 手工基于规模/质量设权重；
- temperature 平滑；
- 小 proxy model 做 mixture ablation；
- RegMix：随机采样许多 mixtures 训练小模型，用回归预测大候选 mix；
- Olmix：系统研究配置，并在域集合增删时复用既有 mixture，减少重算。

混合效果与训练阶段有关：pretraining 广覆盖，mid-training 可提高 math/code/instruction/高质量；同一固定 mix 未必全程最优。重复上采样小域会增加 overfitting/记忆，记录 effective epochs。

## 10.14 合法性、许可与治理

公开可访问不等于自由复制/训练。记录来源、抓取条款、license、opt-out、个人数据、地域法律和下游发布限制。不同司法辖区/时间的版权与隐私规则会变；本章是工程框架，不是法律意见。

建立 removal pipeline：根据 URL/content hash/source ID 定位并重建受影响 shards/model data record。若连来源都没保存，无法响应审计或删除请求。

## 10.15 小模型 data ablation

固定 tokenizer、model、训练 tokens/FLOPs、seed/schedule，只换一个 data intervention：raw、extracted、heuristic-filtered、model-filtered、dedup、different mix。评测：多域 val PPL、下游、memorization、toxicity/PII、吞吐与保留率。

小模型能筛明显坏策略，但 filter 与规模交互；最终大训练前在更接近规模做确认。数据 experiment table 要保留 bytes/docs/tokens 流量漏斗。

## 10.16 论文思路

### Raffel et al., *T5 / C4* (2019/2020)

把 NLP 任务统一成 text-to-text，并构建 Colossal Clean Crawled Corpus；用语言、标点、坏词、重复等规则清洗 Common Crawl。C4 成为重要基线，也暴露 heuristic 偏见与可复现数据治理问题。

### Gao et al., *The Pile* (2020/2021)

把 22 个多样来源混成 825GB 开放语料，强调领域覆盖而非单一网页；为 EleutherAI 开放模型提供基础。不同来源许可/质量/重复不一致，是混合语料治理的案例。

### Lee et al., *Deduplicating Training Data Makes Language Models Better* (2021)

构建 exact/near dedup 工具，展示重复语料导致生成记忆、train-test overlap 与计算浪费；去重可降低记忆并改善效率。阈值/单位影响误删，是部署关键。

### Soldaini et al., *Dolma* (2024)

公开 3T token 语料、工具和数据消融，统一 filtering/mixing，并记录 PII、语言、质量和去污染。其贡献不仅是规模，更是可审计过程与受控 ablation。

### Li et al., *DCLM* (2024)

固定训练/evaluation recipe，把“怎样从公共 crawl 选数据”变成 data curation benchmark；强 baseline 用 model-based quality filtering。说明数据算法可在固定模型预算上系统优化。

## 10.17 数据管线验收

1. 每个最终文档能追到 source snapshot/record 和 transformation version。
2. 流量漏斗：每步 docs/bytes/tokens/域分布、删除原因。
3. PII/语言/质量 filter 的带人工样本 precision/recall。
4. exact/near dedup cluster 抽查、阈值曲线和 split leakage 检查。
5. benchmark contamination report。
6. 至少一个固定预算小模型 ablation。
7. data card：来源、许可、偏差、风险、opt-out/removal、已知缺口。
