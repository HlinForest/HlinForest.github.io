# AI Infra 前置基础 Resources

## Knowledge

- [AIInfraGuide：第 1 章——编程语言基础](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC1%E7%AB%A0-%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80%E5%9F%BA%E7%A1%80/)
  Python、C/C++、Linux、环境与跨语言扩展的总览。用于建立跨层调试主线。
- [AIInfraGuide：第 2 章——数学基础](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC2%E7%AB%A0-%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/)
  从 shape、代价与数值风险三个工程问题组织线代、概率、反传和混合精度。
- [AIInfraGuide：第 3 章——AI Infra 工程师学 Transformer](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC3%E7%AB%A0-transformer%E6%9E%B6%E6%9E%84%E8%AF%A6%E8%A7%A3/)
  建立 Attention、FFN、归一化、位置编码与 Infra 优化之间的映射。
- [AIInfraGuide：第 4 章——PyTorch 框架](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC4%E7%AB%A0-pytorch%E6%A1%86%E6%9E%B6/)
  Tensor、autograd、训练循环、显存调试与 profiler 的路线索引。
- [AIInfraGuide：第 5 章——GPU 硬件概论](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC5%E7%AB%A0-gpu%E7%A1%AC%E4%BB%B6%E6%A6%82%E8%AE%BA/)
  SM、Warp、存储层次、Roofline 与互联拓扑的总览。
- [AIInfraGuide：第 6 章——集合通信基础](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/%E7%AC%AC6%E7%AB%A0-%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E5%9F%BA%E7%A1%80/)
  Send/Recv、AllReduce、AllGather、ReduceScatter、Ring/Tree 与 NCCL 的路线索引。
- [Python Language Reference：Data model](https://docs.python.org/3/reference/datamodel.html)
  Python 官方语言参考。用于核对对象身份、类型、值、可变性和语言协议。
- [GCC：Overall Options](https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html)
  GCC 官方手册。用于核对预处理、编译、汇编与链接各阶段。
- [CMake 官方教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
  用于从最小项目开始理解 target、库与构建配置。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  Transformer 原始论文。用于核对架构、缩放点积注意力与并行化动机。
- [PyTorch：Automatic Differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
  PyTorch 官方教程。用于理解动态图、梯度累积和 `backward()`。
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  CUDA 官方编程指南。用于核对执行模型、内存层次与同步语义。
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  CUDA 官方性能指南。用于建立先测量、再并行、再优化并持续验证的 APOD 循环。
- [NVIDIA NCCL：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
  NCCL 官方文档。用于核对集合通信原语的输入、输出和排序要求。
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
  ISO C++ 社区维护的现代 C++ 工程准则。用于核对 RAII、所有权和资源安全实践。
- [Linux signal(7)](https://man7.org/linux/man-pages/man7/signal.7.html)
  Linux man-pages 项目参考。用于核对进程信号、默认动作与可靠退出语义。

## Wisdom (Communities)

- [AIInfraGuide GitHub 仓库](https://github.com/caomaolufei/AIInfraGuide)
  用于提交文档问题、对照源码与观察其他学习者的真实困惑。
