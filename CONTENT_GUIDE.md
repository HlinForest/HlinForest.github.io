# Spike 个人网站维护与发布手册

这份手册对应当前仓库：[`HlinForest/HlinForest.github.io`](https://github.com/HlinForest/HlinForest.github.io)。

网站地址：<https://hlinforest.github.io/>

## 1. 先认识三个位置

| 要修改的内容 | 文件或目录 | 网站中的路径 |
| --- | --- | --- |
| 首页 / Misc 照片 | `src/assets/` / `src/assets/misc/` | 由 Astro 自动生成多尺寸 AVIF / WebP |
| 文章 SVG、Manim 图等静态文件 | `public/images/notes/` | `/images/notes/文件名` |
| Misc 照片、音乐与署名数据 | `src/data/misc.ts` | `/misc/` |
| Misc 页面布局 | `src/pages/misc/index.astro` | `/misc/` |
| Notes 笔记 | `src/content/blog/` 中的 `.md` 或 `.mdx` | `/notes/文件名/` |

重要规则：图片放在 `public/images/example.jpg` 后，网页里写的是 `/images/example.jpg`，不要写 `/public/images/example.jpg`。

## 2. 两种维护方式

### 方式 A：直接用 GitHub 网页（最简单）

1. 打开[仓库首页](https://github.com/HlinForest/HlinForest.github.io)。
2. 上传图片时，进入目标目录，选择 **Add file → Upload files**。
3. 新建笔记时，进入 `src/content/blog/`，选择 **Add file → Create new file**。
4. 修改已有文件时，打开文件，点击铅笔图标 **Edit this file**。
5. 页面下方填写简短说明，点击 **Commit changes**，提交到 `main`。
6. 打开仓库的 [Actions](https://github.com/HlinForest/HlinForest.github.io/actions)，等待 `Deploy to GitHub Pages` 变成绿色。
7. 通常一至两分钟后刷新网站。若仍是旧页面，使用 `Ctrl + F5` 强制刷新。

只要提交到 `main`，GitHub 会自动构建和发布，不需要手动上传 `dist`。

### 方式 B：在电脑上预览后再发布

项目使用 Node.js 24 和 pnpm 11.19.0。

```powershell
git clone https://github.com/HlinForest/HlinForest.github.io.git
cd HlinForest.github.io
corepack.cmd pnpm install
corepack.cmd pnpm dev
```

浏览器打开终端显示的地址，通常是 <http://127.0.0.1:4321/>。

修改完成后检查并发布：

```powershell
corepack.cmd pnpm run build
node scripts/check-site.mjs
git status
git add .
git commit -m "Add a new note"
git push origin main
```

构建先运行 Astro 类型检查，再生成站点；链接检查同时检查图片的响应式 srcset。PowerShell 使用 `corepack.cmd` 可避开 `.ps1` 执行策略限制。

## 3. 如何上传照片

### 3.1 文件位置

```text
src/assets/spike-profile-space.webp  # 首页 Hero 原图
src/assets/misc/                     # Misc 照片和音乐人原图
src/data/misc.ts                      # 图片列表、分组、图注和出处
public/images/notes/                 # 文章 SVG、Manim 图等，保持原路径
```

文件名使用小写英文、数字与短横线。保留高质量原图即可，Astro 的 `<Picture>` 在构建时生成 AVIF、WebP 和多种尺寸；不要手动提交 `dist/`。`src/assets` 文件必须通过 import 引用，不能把它写成网页 URL。

### 3.2 给相册增加照片

把照片放到 `src/assets/misc/my-photo.jpg`，然后在 `src/data/misc.ts` 增加导入：

```ts
import myPhoto from '../assets/misc/my-photo.jpg';
```

在 `photos` 数组里增加一项：

```ts
{ src: myPhoto, alt: '雨后街道上的树影', label: '雨后散步', group: '我的照片', credit: { text: 'Photography · Spike' } },
```

`alt` 描述画面，`label` 是图注，`group` 是分组按钮文字。分组、计数、键盘操作和触屏切换都自动生成，不需要添加按钮或修改轮播脚本。原来的 `Tokyo Blue Hour` 和 `言叶之庭` 已各自保留出处；新照片填写自己的署名或真实来源，不要复用不相关的作者信息。

有来源或许可时可以这样填写：

```ts
credit: {
  text: 'Photography · 作者名',
  href: 'https://example.com/original',
  linkLabel: 'Source · 原始作品',
  license: { label: '许可名称', href: 'https://example.com/license' },
}
```

相册一次显示一张，进入可视区域后只预加载下一张；其余图片保存在模板中，切换时才载入。没有 JavaScript 时仍显示第一张照片和出处。

### 3.3 首页与音乐人

首页 Hero 文件在 `src/assets/spike-profile-space.webp`。更换图片时保留文件名，或同时修改 `src/pages/index.astro` 与 `src/layouts/BaseLayout.astro` 的导入；默认社交分享图由 Hero 自动生成 1200px WebP。

音乐人原图同样放入 `src/assets/misc/`，在 `musicians` 数组中增加一项；`quote` 是可选的文字数组，每项一行。保留原作作者、AI 转换说明和许可证信息。

## 4. 如何修改文字与样式

### 4.1 图注与出处

修改 `src/data/misc.ts` 中对应记录的 `label`、`credit`。图注与出处跟随每张照片，不再依赖“第几张属于哪个作者”的硬编码。

### 4.2 东京文字与全站字体

东京的蓝调时刻文字在 `src/pages/misc/index.astro` 的 `<details class="tokyo-story">` 中。文字默认收起，只在 `Tokyo Blue Hour` 分组显示。

中文正文使用霞鹜文楷屏幕阅读版，通过锁定版本的 jsDelivr 字符分片加载；不可用时回退到苹方 / 微软雅黑。英文界面用自托管 Inter，英文标题用 Source Serif 4，日期和代码用 JetBrains Mono。字体入口在 `BaseLayout.astro`，颜色与字体变量在 `src/styles/tokens.css`。

全站样式按 `base / layout / home / notes / article / misc` 拆分，`global.css` 只负责导入。默认主题跟随系统，手动选择后会记住偏好。清除浏览器存储中的 `theme` 可恢复跟随系统。

### 4.3 在笔记中插入带说明的照片

最简单的 Markdown 写法：

```md
![注意力矩阵的可视化](/images/notes/attention-map.png)
```

需要图注时使用 HTML：

```html
<figure>
  <img
    src="/images/notes/attention-map.png"
    alt="不同注意力头的热力图"
    loading="lazy"
  />
  <figcaption>图 1：不同注意力头关注的位置。</figcaption>
</figure>
```

## 5. 如何新增一篇笔记

### 5.1 新建文件

在 `src/content/blog/` 中新建文件，例如：

```text
src/content/blog/transformer-notes.md
```

文件名就是网址的一部分：

```text
https://hlinforest.github.io/notes/transformer-notes/
```

普通内容使用 `.md`；需要 Callout、流程图或可切换面板时使用 `.mdx`。

### 5.2 可直接复制的 Markdown 模板

````md
---
title: "Transformer 学习笔记"
description: "从注意力机制到训练细节的整理。"
publishedAt: "2026-08-22"
updatedAt: "2026-08-22"
tags: ["Transformer", "Deep Learning"]
series: "Machine Learning"
seriesOrder: 1
featured: false
draft: false
lang: "zh"
---

这段是文章导语。页面标题会根据上方的 `title` 自动生成，正文不需要再写一级标题。

## 1. 问题背景

正文内容。

### 1.1 一个小节

行内公式：$y = Wx + b$。

块级公式：

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

```python title="attention.py" {3}
def attention(q, k, v):
    scores = q @ k.transpose(-1, -2)
    return scores.softmax(dim=-1) @ v
```

![注意力示意图](/images/notes/attention.png)

## 2. 总结

- 第一条结论
- 第二条结论
````

### 5.3 各字段的作用

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `title` | 是 | 页面标题 |
| `description` | 是 | Notes 列表和页面摘要 |
| `publishedAt` | 是 | 首次发布日期，格式为 `YYYY-MM-DD` |
| `updatedAt` | 否 | 最近更新日期 |
| `tags` | 否 | 标签筛选，可写多个 |
| `series` | 否 | Notes 页的“分区” |
| `seriesOrder` | 否 | 系列内部预留顺序字段 |
| `featured` | 否 | 预留的精选标记 |
| `draft` | 否 | `true` 不发布，`false` 发布 |
| `lang` | 否 | `zh` 或 `en` |

`series` 用于大分区，例如 `Machine Learning`、`Systems`、`Mathematics`；`tags` 用于更细的主题，例如 `Transformer`、`CUDA`、`Optimization`。

### 5.4 MDX 交互组件

将文件扩展名改为 `.mdx`，在正文最前面导入组件：

```mdx
import Callout from '../../components/Callout.astro';
import FlowDiagram from '../../components/FlowDiagram.astro';
import ComparisonTabs from '../../components/ComparisonTabs.astro';
```

提示框：

```mdx
<Callout type="idea" title="关键直觉">
  注意力机制让每个位置按内容动态聚合其他位置的信息。
</Callout>
```

`type` 可以是 `note`、`warning`、`idea` 或 `danger`。

流程图：

```mdx
<FlowDiagram
  nodes={['输入', 'Embedding', 'Attention', 'MLP', '输出']}
  note="Transformer 单层数据流"
/>
```

可切换对比面板：

```mdx
<ComparisonTabs
  label="训练方式对比"
  tabs={[
    { label: 'Full FT', title: '全量微调', body: '更新全部参数。' },
    { label: 'LoRA', title: '低秩适配', body: '只训练低秩增量。' },
  ]}
/>
```

二级和三级标题会自动进入文章右侧目录；代码块自带语法高亮与复制按钮；表格、任务列表和 KaTeX 公式均可直接使用。

## 6. 发布前检查清单

- [ ] Misc 图片通过 import 引用；文章静态图片路径以 `/images/` 开头。
- [ ] 文件名大小写与引用完全一致。
- [ ] 每张图片都有准确的 `alt`。
- [ ] 他人照片保留作者、来源、许可和水印。
- [ ] 笔记 frontmatter 的 `title`、`description`、`publishedAt` 已填写。
- [ ] 准备发布时 `draft: false`。
- [ ] 本地运行过 `pnpm test`，或 GitHub Actions 已变绿。
- [ ] 手机和电脑上各检查一次页面。

## 7. 常见问题

### 图片显示 404

检查路径、扩展名和大小写。Misc 图片检查 `src/assets/misc/` 与 `src/data/misc.ts` 的 import；文章图片 `public/images/notes/a.svg` 对应 `/images/notes/a.svg`。

### 新笔记没有出现在 Notes

检查文件是否位于 `src/content/blog/`、扩展名是否为 `.md` 或 `.mdx`，以及 `draft` 是否仍为 `true`。

### GitHub Actions 失败

打开失败的构建，展开红色步骤。最常见原因是 YAML 引号或缩进错误、必填字段缺失、MDX 标签没有闭合、图片路径拼错。

### 已发布但仍看到旧页面

先确认 Actions 已完成，再等待约一分钟并使用 `Ctrl + F5`。手机端可使用无痕窗口检查。
