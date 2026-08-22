# Spike 个人网站维护与发布手册

这份手册对应当前仓库：[`HlinForest/HlinForest.github.io`](https://github.com/HlinForest/HlinForest.github.io)。

网站地址：<https://hlinforest.github.io/>

## 1. 先认识三个位置

| 要修改的内容 | 文件或目录 | 网站中的路径 |
| --- | --- | --- |
| 照片、插图等静态文件 | `public/`，新文件建议放进 `public/images/` | `/images/文件名.jpg` |
| Misc 摄影与音乐页面 | `src/pages/misc/index.astro` | `/misc/` |
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
npm install -g pnpm@11.19.0
pnpm install
pnpm dev
```

浏览器打开终端显示的地址，通常是 <http://127.0.0.1:4321/>。

修改完成后检查并发布：

```powershell
pnpm test
git status
git add .
git commit -m "Add a new note"
git push origin main
```

`pnpm test` 会构建网站，并检查生成页面中的站内链接与本地资源。

## 3. 如何上传照片

### 3.1 文件整理建议

新照片建议按用途放置：

```text
public/
└─ images/
   ├─ misc/       # Misc 页面照片
   └─ notes/      # 笔记插图
```

文件名建议使用小写英文、数字与短横线：

```text
tokyo-blue-hour.jpg
garden-rain-01.webp
attention-diagram.png
```

不要使用空格，尽量避免中文文件名。GitHub Pages 区分大小写，`Photo.jpg` 与 `photo.jpg` 是两个不同文件。

照片优先使用 `.jpg` 或 `.webp`；透明图和界面截图使用 `.png`。普通照片建议宽度约 1600–2400 像素、单张尽量控制在 1.5 MB 以内。

只上传自己拍摄、获得授权或许可明确的图片。引用他人作品时保留水印、作者、原始链接与许可信息。

### 3.2 把一张照片放进 Misc

先把照片上传为：

```text
public/images/misc/my-photo.jpg
```

然后打开 `src/pages/misc/index.astro`，在 Photography 区域加入：

```astro
<figure class="photo-feature">
  <div class="photo-feature-frame">
    <img
      src="/images/misc/my-photo.jpg"
      alt="准确描述画面内容，用于无障碍访问"
      width="1800"
      height="1200"
      loading="lazy"
    />
  </div>
  <figcaption>
    <span>照片标题</span>
    <span class="photo-meta">地点 · 年份 · Photographer: Spike</span>
  </figcaption>
</figure>
```

`width` 和 `height` 填照片原始尺寸。`alt` 描述画面，不要写“图片”或文件名。

### 3.3 给当前东京／言叶之庭相册增加照片

当前合并相册的顺序是：东京蓝调时刻、湖面、草地、绣球、百合。若要继续增加同一组、同一来源的雨景：

1. 在 `garden-album-stage` 中、相册控制栏之前复制一个 `data-garden-slide` 块并修改图片路径。
2. 在 `garden-album-dots` 中增加下一个编号按钮。
3. 在页面下方脚本的 `gardenLabels` 数组末尾增加对应标题。
4. `data-garden-dot` 从 `0` 开始，必须与照片顺序一致。

第 6 张照片的示例：

```astro
<button type="button" class="garden-album-slide" data-garden-slide hidden>
  <img
    src="/images/misc/garden-rain-05.jpg"
    alt="雨中的庭院石阶"
    width="1800"
    height="1200"
    loading="lazy"
  />
</button>
```

控制栏增加：

```astro
<button
  type="button"
  data-garden-dot="5"
  aria-label="查看雨中石阶"
  aria-pressed="false"
>06</button>
```

脚本数组增加：

```ts
const gardenLabels = [
  'Tokyo · Blue hour',
  '言叶之庭 · 雨天湖面',
  '言叶之庭 · 雨幕草地',
  '言叶之庭 · 雨中绣球',
  '言叶之庭 · 雨中百合',
  '言叶之庭 · 雨中石阶',
];
```

注意：当前第 2 张以后统一显示 Alan D. Haller 的署名与来源。自己的照片或其他作者的照片不要直接加进这组相册，应使用上一节的独立照片块并填写正确署名。

## 4. 如何给照片添加文字

### 4.1 照片下方的标题与说明

在 `figcaption` 中修改：

```astro
<figcaption>
  <span>Tokyo · Blue hour</span>
  <span class="photo-meta">
    摄于东京 · 2026<br />
    <a href="原始来源链接">Source · 作者或出处 ↗</a>
  </span>
</figcaption>
```

没有外部来源时不要虚构链接，写自己的署名即可。

### 4.2 修改东京蓝调照片的悬停文字

在 `src/pages/misc/index.astro` 中搜索 `data-photo-overlay`，编辑其中的段落：

```astro
<div class="photo-reveal" data-photo-overlay>
  <blockquote>
    <p>第一段文字。</p>
    <p>第二段文字。</p>
    <p>第三段文字。</p>
  </blockquote>
</div>
```

每个 `<p>...</p>` 是一段。不要删除外层的 `photo-reveal`、`blockquote` 或 `data-photo-overlay`，否则悬停效果会失效。

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

- [ ] 图片路径以 `/` 开头，且没有写 `public`。
- [ ] 文件名大小写与引用完全一致。
- [ ] 每张图片都有准确的 `alt`。
- [ ] 他人照片保留作者、来源、许可和水印。
- [ ] 笔记 frontmatter 的 `title`、`description`、`publishedAt` 已填写。
- [ ] 准备发布时 `draft: false`。
- [ ] 本地运行过 `pnpm test`，或 GitHub Actions 已变绿。
- [ ] 手机和电脑上各检查一次页面。

## 7. 常见问题

### 图片显示 404

检查路径、扩展名和大小写。`public/images/misc/a.jpg` 的引用必须是 `/images/misc/a.jpg`。

### 新笔记没有出现在 Notes

检查文件是否位于 `src/content/blog/`、扩展名是否为 `.md` 或 `.mdx`，以及 `draft` 是否仍为 `true`。

### GitHub Actions 失败

打开失败的构建，展开红色步骤。最常见原因是 YAML 引号或缩进错误、必填字段缺失、MDX 标签没有闭合、图片路径拼错。

### 已发布但仍看到旧页面

先确认 Actions 已完成，再等待约一分钟并使用 `Ctrl + F5`。手机端可使用无痕窗口检查。
