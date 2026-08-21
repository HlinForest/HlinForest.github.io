# Mission: 掌握 ARENA Fundamentals，并独立训练与诊断模型

## Why
把已有的 Python 编程与机器学习理论知识转化为真正的深度学习实践能力：完整学完 ARENA Chapter 0 Fundamentals，能够独立实现、训练、评估和调试模型，并为 Transformer、机制可解释性与 AI Safety 学习打下可迁移的工程基础。

## Success looks like
- 能只看任务描述，独立写出张量处理、`nn.Module`、训练/验证循环、推理及检查点保存代码
- 能根据任务正确推导输入/输出 shape，并为回归、二分类和多分类选择匹配的输出层、标签格式与损失函数
- 能解释 autograd、梯度累积、优化器更新、训练/评估模式，而不是只会套模板
- 能训练并诊断一个小型分类模型，依据 loss、准确率和决策边界提出有根据的改进
- 能用批量线性代数实现光线追踪，并解释 broadcasting、view/copy 与向量化为何重要
- 能从基础模块组装 CNN 与 ResNet，解释卷积、BatchNorm、残差连接和迁移学习
- 能实现 SGD、RMSprop、Adam/AdamW，设计可复现的实验与超参数搜索
- 能从零实现小型 autograd，并说明反向传播的拓扑顺序、梯度累积和 unbroadcast
- 能训练 Autoencoder、VAE 与 GAN，解释潜变量、重参数化、ELBO、对抗损失与训练不稳定性
- 能把这些概念映射到大模型中的 token 张量、embedding、logits、交叉熵与训练循环

## Constraints
- Python 基础扎实，能够完成 LeetCode 题目
- 机器学习理论可通过笔试并能推导 loss，但尚无实际训练模型经验
- 课程必须详尽，同时拆成短课，优先动手、主动回忆和可观察结果
- 当前本地 Python 环境未安装 PyTorch；材料必须支持 CPU 与 Google Colab
- ARENA 原课按高强度全日制节奏设计；本工作区改造成适合自学的分层路线，必修内容先于奖励性底层实现

## Out of scope
- 本阶段不进入 ARENA Chapter 1 之后的 Transformer、RL、LLM Evals 与 Alignment Science 正文
- 分布式训练先掌握概念和通信原语；没有多 GPU 环境时不要求完成昂贵实跑
- 不追求完整 MLOps、部署、性能 profiling 或 CUDA 内核开发
- 不把调库跑通当作掌握；核心目标是能解释并独立重建训练流程
