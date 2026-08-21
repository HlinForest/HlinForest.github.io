# AI Infra 前置基础实验

课程材料的静态完整性可以在项目根目录运行：

```powershell
python aiinfra-foundations/tools/validate_course.py
```

建议从项目根目录运行：

```powershell
python aiinfra-foundations/labs/environment_audit.py
python aiinfra-foundations/labs/python_semantics_lab.py
python aiinfra-foundations/labs/tensor_math_lab.py
python aiinfra-foundations/labs/numerics_lab.py
python aiinfra-foundations/labs/attention_lab.py
python aiinfra-foundations/labs/decoder_block_ledger.py
python aiinfra-foundations/labs/kv_cache_calculator.py
python aiinfra-foundations/labs/roofline_lab.py
python aiinfra-foundations/labs/collectives_lab.py
```

`pytorch_training_lab.py` 需要 PyTorch；CUDA/BF16 对比需要 NVIDIA GPU。C++ 实验需要 CMake 和支持 C++20 的编译器。

实验方法：

1. 先阅读对应单元并预测输出。
2. 不改代码运行一次，保存基线。
3. 按文件末尾的 EXERCISES 修改。
4. 让断言重新全部通过。
5. 用自己的话解释结果和性能含义。

完成实验不等于只看到 `PASS`；你必须能解释为什么通过。
