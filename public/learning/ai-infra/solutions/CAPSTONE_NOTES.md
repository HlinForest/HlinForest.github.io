# Capstone instructor notes

这不是唯一正确答案，只列出证据中应被发现的主线。

## 高优先级正确性问题

1. 手工 `softmax → log` 在 FP16 下数值不稳定；应使用 fused cross entropy / log-softmax 稳定路径。
2. 没有 GradScaler 的 FP16 训练可能发生梯度下溢；也应比较 BF16/FP32 参考。
3. `zero_grad` 的位置与 gradient accumulation 语义错误：当前代码每一步都 `optimizer.step()`，却隔若干步才清梯度。
4. 应定位首个非有限 logits、probabilities、loss 或 gradient，而不是等待最终 loss。

## 性能问题

1. 每一步 `loss.item()` 是同步点。
2. DataLoader 周期性等待与 pageable H2D 表明数据管线/固定内存仍需验证。
3. profiler 显示大量 pointwise kernel，存在 fusion 候选。
4. 12 个 AllReduce bucket 可检查 bucket 粒度与 overlap；当前记录显示 overlap 很少。
5. AdamW foreach 关闭可能增加 Kernel 数。

## 显存差异

首版表格若只算参数/梯度/优化器，会漏掉保存激活、通信 bucket、workspace、allocator reserved/fragmentation 与框架上下文。

## 通信与 hang

1. 所有 ranks 使用 mlx5_0，但拓扑显示 GPU4-7 更接近 NUMA1 / mlx5_1，NIC 亲和性可疑。
2. rank6 因本地 non-finite loss 跳过 backward，而其他 ranks 进入 collective，调用顺序不一致足以造成 hang。
3. 应先修正确性，再从单机两卡逐级复现，结合 nccl-tests 建立链路基线。

## ABI

环境 B 直接复制了 Python 3.12、PyTorch 2.8 构建的扩展到 Python 3.11、PyTorch 2.7 环境。文件名 ABI tag 与 undefined C10 symbol 都支持二进制不兼容，应在目标环境按匹配工具链重新构建，而不是调整 PYTHONPATH 掩盖。
