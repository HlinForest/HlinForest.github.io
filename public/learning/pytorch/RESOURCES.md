# PyTorch 系统基础资源

## Knowledge

- [ARENA Chapter 0：Fundamentals](https://learn.arena.education/chapter0_fundamentals/)
  本阶段的官方课程入口，当前包含 0.0 Prerequisites、0.1 Ray Tracing、0.2 CNNs & ResNets、0.3 Optimization、0.4 Backpropagation、0.5 VAEs & GANs。
- [ARENA 官方仓库](https://github.com/callummcdougall/ARENA_3.0)
  页面正文、练习、测试与辅助代码的源版本。2026-08-13 的阅读审计以仓库 `main` 和在线课程交叉核对。
- [ARENA Fundamentals 全景地图](reference/0006-arena-fundamentals-map.html)
  本工作区的中文总览：六章的依赖链、关键技能、bonus 与毕业标准。
- [ARENA 正文章节阅读摘要](reference/0007-arena-reading-digests.html)
  Reading 区块及正文明确推荐材料的逐项中文摘要。
- [ARENA 先修与延伸阅读摘要](reference/0008-arena-prerequisite-reading-digests.html)
  0.0 先修清单和 Optional Reading 的逐项摘要；按薄弱项选择，不要求开课前全部通读。
- [ARENA 正文散列与 Bonus 阅读摘要](reference/0009-arena-inline-and-bonus-reading-digests.html)
  补齐正式 Reading 区块之外的概念证明、Bonus 背景、调试方法、规模化训练与数学查阅材料。

- [课程：00. PyTorch Fundamentals — Learn PyTorch](https://www.learnpytorch.io/00_pytorch_fundamentals/)
  本套材料的第一主线来源。覆盖 tensor 创建、dtype/device/shape、运算、索引、NumPy 互操作、随机复现和 GPU。用于课程 1–3。
- [课程：01. PyTorch Workflow Fundamentals — Learn PyTorch](https://www.learnpytorch.io/01_pytorch_workflow/)
  本套材料的第二主线来源。用线性回归串起数据、`nn.Module`、loss、optimizer、训练/测试、推理及保存加载。用于课程 4–7。
- [课程：02. PyTorch Neural Network Classification — Learn PyTorch](https://www.learnpytorch.io/02_pytorch_classification/)
  本套材料的第三主线来源。覆盖二/多分类、logits、Sigmoid/Softmax、非线性、分类训练与指标。用于课程 8–10。
- [原始 Notebook 仓库：mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)
  三章可运行 notebook、练习和源代码的权威版本。需要完整复现原课程或对照更新时使用。
- [PyTorch 官方：Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/)
  官方端到端入门路径，覆盖 tensors、DataLoader、模型、autograd、优化和保存加载。用于核对当前 API 与补充主课程。
- [PyTorch 官方：Tensor](https://docs.pytorch.org/docs/stable/tensors.html)
  `torch.Tensor` 的权威定义、构造、索引、dtype/device 及 autograd 行为。遇到张量语义问题时优先查阅。
- [PyTorch 官方：Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
  计算图、`requires_grad`、`grad_fn`、`backward()` 和禁用梯度的官方解释。用于课程 4–5。
- [PyTorch 官方：BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
  二分类稳定损失的权威说明，包括 logits 输入、数值稳定性、target shape 与 `pos_weight`。用于课程 8。
- [PyTorch 官方：CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
  多分类交叉熵的权威说明，包括 logits、类别索引标签与 shape 约定。用于课程 10。
- [PyTorch 官方：安装选择器](https://pytorch.org/get-started/locally/)
  根据操作系统、包管理器与 CUDA/CPU 组合生成当前安装命令。不要依赖博客中的旧命令。

## Wisdom (Communities)

- [PyTorch Forums](https://discuss.pytorch.org/)
  PyTorch 官方社区，适合带最小复现代码询问 shape、autograd、数据加载与运行时错误。
- [课程 GitHub Discussions](https://github.com/mrdbourke/pytorch-deep-learning/discussions)
  与三章课程直接对应，适合查询 notebook 版本差异、练习疑问和他人踩坑记录。

## Gaps

- 完成本阶段并通过综合挑战后，再补充专门面向 Transformer 的 PyTorch 张量与训练资料；当前先避免超出近侧发展区。
