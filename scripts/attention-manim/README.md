# Attention 分步矩阵图

使用 **3b1b/ManimGL 1.7.2**，不是 Manim Community。每个场景描述一个计算动作；
samples.py、core_attention.py、memory_variants.py、advanced.py
保存公式、真实数值矩阵和对应说明。storyboard.py 负责渲染。

## 复现

安装 Python 3.11、TeX Live（包含 latex 与 dvisvgm）和支持 OpenGL 的驱动。
将 TeX Live 的可执行目录加入 PATH。中文字体使用 Microsoft YaHei；
其他系统请将渲染器及配置中的字体换成本地可用 CJK 字体。

~~~powershell
python -m venv .venv-attention
.venv-attention\Scripts\python -m pip install -r scripts/attention-manim/requirements.txt
.venv-attention\Scripts\python scripts/attention-manim/check_scenes.py
.venv-attention\Scripts\python scripts/attention-manim/render.py
~~~

render.py 使用当前 Python 环境启动 Manim，无固定磁盘或个人目录路径。
可添加 --groups heads,linear-normalizer 或 --layout mobile 做局部渲染。
网页接入前必须完整渲染两种布局；局部渲染产生的清单不能作为整篇验收结果。

## 阅读约定

- 数学向量默认是列向量；行向量标记转置。MLA、GQA 的行向量公式
  在各图中直接标明。矩阵高对应行数、宽对应列数。
- 所有数值矩阵共享单元格边长；转置实际交换高宽。已知零和不可见格子不填色。
- Q 蓝、K 橙、V 紫、概率绿、状态及归一化和赭色、输出粉红、其他权重灰。
- 各组小例子用于解释局部运算，不代表同一套端到端模型参数；
  特别是示意概率、latent 和简化门系数，其假设写在对应图注中。
- 手机按完整运算对象纵向排版。仅裁掉画布外围空白，不截取或拼接矩阵内部。
- check_scenes.py 独立复算图中可数值化的矩阵乘法、加减和逐元素乘法。
  这不能替代对图形布局、索引语义和正文定义的人工检查。

参考教学顺序来自 Jia-Bin Huang《How Attention Got So Efficient》
及用户提供的截图；图中数值、场景和图片为本笔记重新制作。
