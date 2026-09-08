# Attention 图解修订验证记录

日期：2026-09-08。文章路径与线上 URL 保持不变。

## 内容与来源

- 面向只了解 Attention 的读者，展开加权和、head 拼接与 WO、mask/padding/dropout、矩阵乘法 FLOPs。
- 新增 GQA、MLA 权重吸收、解耦 RoPE、DSA 索引/gather/量化/训练；按用户提供的视频章节安排组织，未声称取得视频字幕或逐字还原。
- Linear 从新核、结合律、分子/分母、外积数值例子到因果与分块展开。
- Flash 从 3-pass safe softmax、2-pass online normalizer、融合 value，到 tile 和 FA1/2/3 展开。
- 38 张原创 SVG，含后续 Delta/KDA/chunk/反向传播图；保留初版系统示意图和标明日期的 CPU 性能记录。
- 核对记录见 [一手来源审读](research/attention-revision-sources.md)。修正了 batch 投影梯度说明、online 空行例外、DSA 行向量记法及 KDA scale 边界。

## 已执行的检查

| 检查 | 结果 |
|---|---|
| 完整参考程序，Python 3.11.7 / NumPy 1.26.4 / Matplotlib 3.8.0 | 129 项必需 CPU 检查通过 |
| 新增逐步 softmax / GQA / MLA / DSA 程序 | 18 项检查通过，含非等宽维度、负 indexer 权重、因果扰动 |
| PyTorch SDPA CPU 对照 | 通过；不计入上述 147 项 |
| 官方 FA2/FA3 GPU | 无可用 CUDA，跳过；不声称验证实际 GPU 性能 |
| `python scripts/attention-check.py` | 文中/下载代码完全一致、语法合法、38 图引用完整、等形与方阵几何通过 |
| `pnpm build` | Astro 21 文件类型检查 0 error / 0 warning，静态构建成功；另有已有配置项弃用提示 |
| `node scripts/check-site.mjs` | 5 个 HTML 页面、135 个生成文件，站内引用无缺失 |
| 浏览器 SVG 文字边界检查 | 38 图无文字越界或文字重叠；另检查全部缩略图与主要图原尺寸 |
| Edge headless：1440px / 390px | 均显示 38 张新图，无缺失图片、KaTeX 错误、整页水平溢出；图内可单独横向滚动 |

SVG 源码就是网页中的可编辑矢量文件；配方与形状记录可用于复核与再生成。浏览器检查的临时截图留在工作区 `tmp/attention-qa/`，未作为网站内容发布。

## 复现

在仓库根目录执行：

```bash
python public/code/attention-reference.py
python public/code/attention-variants.py
python scripts/attention-figures.py
python scripts/attention-check.py
pnpm build
node scripts/check-site.mjs
```

无图形界面的环境可设置 `MPLBACKEND=Agg` 运行参考程序；`plt.show()` 的非交互提示不影响数值断言。文章中的 benchmark 图片来自初版日期，重新运行会因 CPU/BLAS/线程配置而得到不同耗时。
