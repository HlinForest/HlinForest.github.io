# ARENA Fundamentals 阅读材料审计

审计日期：2026-08-13。课程版本：ARENA 官方仓库 `main` 与在线 Chapter 0。六个正文文件约 79 万字符，正文共包含约 173 个去重后的外部链接；其中不少是 API、数据、图片、安装、Colab、W&B 面板或故障排查入口，并不属于要求通读的教学材料。

本工作区采用可复核口径：所有正式 `Reading`、`Optional Reading`，以及正文明确使用 read / watch / recommend 指向的文章、视频、论文、教程，都进入摘要；纯 API、数据集、图片与操作入口列为“按需查阅”，不冒充已经通读的文章。归一化后，三份摘要覆盖 111 个不同的教学来源，合并为 92 张逐项或同主题摘要卡。

## 摘要入口

- [章节 Reading 与正文推荐：0007](reference/0007-arena-reading-digests.html)
- [0.0 先修与 Optional Reading：0008](reference/0008-arena-prerequisite-reading-digests.html)
- [正文散列、证明与 Bonus 阅读：0009](reference/0009-arena-inline-and-bonus-reading-digests.html)
- [0.0–0.5 内容全景：0006](reference/0006-arena-fundamentals-map.html)

## 逐章检查

### 0.0 Prerequisites / Tensor manipulation

- [x] 3Blue1Brown：Neural Networks、Gradient Descent、Backpropagation
- [x] 3Blue1Brown：Essence of Linear Algebra、matrix multiplication 视频；ML Wiki matrix multiplication
- [x] SVD 直觉、Transformer weight matrices 的 SVD
- [x] Linear Algebra Done Right；Neel Nanda 两段线代课（作为长课/参考书归纳重点）
- [x] expectation / variance / covariance；Essence of Calculus
- [x] Elements of Information Theory（按本章所需主题摘要）；KL divergence 直觉
- [x] Intermediate Python、Python tutorial、typing
- [x] NumPy 100、PyTorch Learn the Basics
- [x] Einops benefits、Einops basics、Einsum is all you need
- [x] Plotly、Streamlit（工具定位与使用边界）
- [x] Project Euler；VS Code/Jupyter/debugger；Git 教程；Conda；Unix 教程
- [x] Optional Reading 逐项：NumPy 100、What is torch.nn really、NLP Demystified、Visualising Representations、Spinning Up、David Silver RL、Matrix Cookbook、Zoom In、Why Momentum Really Works、Matrix Calculus、Transformer Circuits、Induction Heads、Michael Nielsen

### 0.1 Ray Tracing

- [x] Real Python `pathlib`（课程 setup 的查阅材料）
- [x] Using GPT-4 to Understand Code（正文明确推荐）
- [x] Play in Hard Mode / Easy Mode；debugger step controls；Colab debugging
- [x] rotation matrices、Lambertian reflection、mypy
- [x] 其余核心理论已直接纳入全景梳理：ray/segment 参数方程、批量线性系统、barycentric coordinates、view/copy/storage、GPU 与 Lambert lighting

### 0.2 CNNs & ResNets

- [x] ARENA 本节导入讲座（按章节结构与重点归纳）
- [x] 3Blue1Brown：But what is a convolution?
- [x] A Comprehensive Guide to CNNs
- [x] Zoom In: An Introduction to Circuits
- [x] Batch Normalization in CNNs
- [x] Deep Residual Learning for Image Recognition
- [x] Bonus 三项：NumPy strides 视频、as_strided and sum、25 illustrated stride exercises
- [x] OrderedDict、skip connections、cross entropy / KL 直觉视频

### 0.3 Optimization

- [x] ARENA 本节导入讲座（按章节结构与重点归纳）
- [x] Andrew Ng：Momentum、RMSProp、Adam
- [x] A Visual Explanation of Gradient Descent Methods
- [x] Why Momentum Really Works
- [x] On Large-Batch Training
- [x] Decoupled Weight Decay Regularization / AdamW
- [x] Random Search for Hyper-Parameter Optimization
- [x] Ring All-Reduce
- [x] Chinchilla / compute-optimal training
- [x] The Optimizer's Curse
- [x] directional derivatives、learning-rate selection、validation overfitting、Hyperband
- [x] Shampoo、mixed-precision large-scale training
- [x] Muon 设计/推导/实现、NanoGPT speedrun 记录、scaling-laws curriculum
- [x] 人工分类 CIFAR-10 与 human/reference baseline
- [x] 分布式通信原语以 ARENA 正文、算法伪代码和官方接口为主梳理；短 API 页保留为查阅项

### 0.4 Backpropagation

- [x] ARENA 本节导入讲座（按章节结构与重点归纳）
- [x] Calculus on Computational Graphs: Backpropagation
- [x] 0.0 的 3Blue1Brown Backprop、Matrix Calculus 与 Michael Nielsen 作为交叉材料
- [x] 从零 autograd 的全部正文结构已梳理：recipe、registry、topological sort、unbroadcast、gradient accumulation、Tensor/Parameter/Module/NoGrad/SGD
- [x] finite differences 与 topological sorting / Kahn’s algorithm

### 0.5 VAEs & GANs

- [x] ARENA 本节导入讲座（按章节结构与重点归纳）
- [x] Understanding Variational Autoencoders
- [x] Six and a half intuitions for KL divergence
- [x] From Autoencoder to β-VAE
- [x] Transposed Convolutions with Excel；convolution arithmetic animations
- [x] β-VAE paper
- [x] Understanding Diffusion Models: A Unified Perspective（hierarchical VAE 延伸）
- [x] Google Machine Learning GAN course
- [x] DCGAN paper
- [x] ganhacks
- [x] Gaussian KL 闭式推导；MNIST/GAN 输入 normalization 讨论

## 按需查阅，不计为“通读文章”

- PyTorch / NumPy / einops 函数与模块 API，例如 `Conv2d`、`BatchNorm2d`、`AdamW`、`ConvTranspose2d`、`gather`、distributed primitives。
- MNIST、CIFAR-10、CelebA 等数据或数据集主页；预训练权重枚举；下载与 Colab 入口。
- Plotly 图片、GIF、课程内嵌图、代码文件、测试文件和练习答案链接。
- VS Code 快捷键、W&B 单个功能面板、GPU/驱动/网络排错页面。

这些链接仍保留在 ARENA 原文和本地官方源码副本中，实际编码遇到接口或环境问题时再查。这个边界避免把“点开一个 API 页面”虚报为阅读研究材料。

## 受限来源说明

少数 Medium、Distill、旧站点或反爬页面不能稳定直接读取。对应摘要使用作者可访问页面、论文正式版本、可靠镜像和 ARENA 正文论述交叉核验；摘要不依赖无法验证的长篇逐字复述。所有摘要都是中文改写，不是原文替代品。
