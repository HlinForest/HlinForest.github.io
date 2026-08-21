# ARENA Fundamentals 自学路线

这不是把 ARENA 的“五天冲刺”机械拉长，而是按依赖关系重排成适合零训练经验学习者的路线。单次建议 60–90 分钟：10 分钟闭卷回忆，20–30 分钟讲义，30–45 分钟编码，10 分钟记录错误。

## 阶段 A：先跑通训练闭环（已有课程 01–10）

先完成 `lessons/0001` 到 `0010`，目标是能写出 Tensor 操作、`nn.Module`、训练/验证循环和分类损失契约。若已经能闭卷训练 MNIST，可用诊断题跳过部分内容。

## 阶段 B：ARENA 0.0 与 0.1（约 6–9 次）

1. 先修诊断：神经网络、线性代数、概率、微积分、信息论。
2. `einops` / `einsum`：用轴名而不是位置猜测张量语义。
3. 张量操作：broadcasting、indexing、gather、logsumexp、softmax、cross entropy。
4. 光线与线段：把几何相交写成二元线性方程。
5. 批量光追：构造 `(ray, segment, 2, 2)` 批量并处理奇异矩阵。
6. 三角形与 mesh：一次求解所有 ray–triangle 对，理解 views/copies。
7. 奖励：GPU、旋转、Lambert 光照、pytest。

## 阶段 C：ARENA 0.2（约 7–10 次）

1. 从零实现 `ReLU`、`Linear`、`Flatten`、MLP。
2. MNIST 训练与验证，明确 train/eval 状态机。
3. 卷积、池化、输出 shape 与归纳偏置。
4. BatchNorm 的参数、buffer 和训练/推理差异。
5. 残差块、BlockGroup、ResNet34 与权重复制。
6. 特征提取：冻结骨干，只训练新分类头。
7. 奖励：`as_strided` 与从零卷积。

## 阶段 D：ARENA 0.3（约 6–9 次）

1. SGD、momentum、weight decay 与病态曲率。
2. RMSprop、Adam、AdamW：状态变量与逐坐标缩放。
3. 参数组与实验公平性。
4. W&B 日志、可复现运行、随机搜索和 sweep。
5. 分布式概念：data / tensor / pipeline parallelism。
6. `send/recv → broadcast/reduce → all_reduce → DDP`。
7. 奖励：Muon、ring all-reduce、scaling laws、optimizer's curse。

## 阶段 E：ARENA 0.4（约 6–8 次）

1. 计算图、局部 backward function 与链式法则。
2. broadcasting 的反向是沿复制轴求和（unbroadcast）。
3. `Tensor + Recipe + registry` 构建动态图。
4. 拓扑排序与梯度累积。
5. 为 reshape、sum、index、max、matmul 实现反向。
6. 构建 Parameter / Module / Linear / CrossEntropy / NoGrad / SGD。
7. 用自己的 autograd 训练 MNIST MLP。

## 阶段 F：ARENA 0.5（约 7–10 次）

1. Autoencoder 与潜空间为何不可直接采样。
2. 转置卷积：卷积的转置算子，不是卷积的逆。
3. VAE：概率编码器、重参数化与 `reconstruction + KL`。
4. ELBO 与 β-VAE 的“重建—规则化”权衡。
5. GAN：generator/discriminator、minimax 与 non-saturating loss。
6. DCGAN 结构、初始化、双优化器与梯度隔离。
7. 诊断 mode collapse、判别器过强、NaN 和训练不稳定。

## 三道毕业检查

1. 不看模板，写出一个含验证、checkpoint 和复现种子的分类训练脚本。
2. 画出任意 PyTorch 表达式的计算图，写出每个节点的局部反向并解释梯度为何相加。
3. 比较 CNN、ResNet、VAE 与 GAN：各自的输入输出、训练信号、归纳偏置和最常见失败模式。
