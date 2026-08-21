# 单元答案与自检提示

请先独立完成实验再阅读。

## 单元 02

- 默认可变参数在函数定义时创建一次，所以不同调用共享对象。
- 纯 Python CPU 循环优先多进程或原生扩展；I/O 等待优先线程/asyncio。
- deepcopy 会复制大量状态，并不理解 Tensor device、共享存储或外部资源语义。

## 单元 03

- `span` 不拥有数据，释放责任属于原拥有者。
- `undefined reference` 是链接错误；`cannot open shared object` 是运行时装载错误。
- 非连续 Tensor 边界至少检查 dtype、device、shape、stride、contiguity、所有权、生命周期和 GIL。

## 单元 04

- 优雅停止优先 SIGTERM；SIGKILL 是无法清理的最后手段。
- `nvidia-smi` 成功但无 `nvcc` 常表示 Toolkit 缺失或 PATH 不对，不代表预编译框架必然不能运行。

## 单元 05

- `(B,H,S,D)@(B,H,D,T)→(B,H,S,T)`。
- 低算术强度通常首先受带宽限制。
- 理想 bytes 是下界，真实实现还包含重复加载、cache miss、workspace 和 C 读写。

## 单元 06

- 稳定 Softmax 先减最大值。
- BF16 的优势是指数范围接近 FP32；FP16 尾数更细但范围更窄。
- Softmax + one-hot CE 的梯度为 `p-y`。

## 单元 07

- 缩放 `√Dh` 控制点积分数方差，避免 Softmax 过饱和。
- GQA/MQA 主要减少 KV Cache 与 Decode 读取带宽。

## 单元 08

- RMSNorm 输入输出 shape 不变。
- Norm/残差/激活融合主要减少 Kernel 启动与 HBM 往返。

## 单元 09

- Prefill 更像大 GEMM/计算密集；Decode 更容易受权重与 KV 读取限制。
- KV Cache 对序列长度线性增长。

## 单元 10

- PyTorch 默认累加叶节点梯度，所以标准 step 需要清零。
- `.item()` 可能让 CPU 等待 GPU 结果，形成同步点。

## 单元 11

- 合并访存减少全局内存事务。
- Occupancy 是隐藏延迟的手段，不是越高越好的最终目标。

## 单元 12

- 数据并行梯度同步常用 AllReduce。
- Ring 每 rank 字节量接近 `2M`，但轮数是 `2(N-1)`，带宽友好、延迟未必友好。
