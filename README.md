# HlinForest

Spike 的个人学术主页、摄影收藏与 Notes，发布于 <https://hlinforest.github.io/>。

## 维护手册

上传照片、添加照片文字、新增笔记、本地预览与 GitHub 发布的完整步骤，请阅读 **[CONTENT_GUIDE.md](./CONTENT_GUIDE.md)**。

## 内容结构

- `src/content/blog/`：Notes，按 `series` 分区、按 `tags` 分类。
- `src/pages/misc/index.astro`：摄影收藏与音乐人页面。
- `public/`：照片、插图和其他静态文件。
- `src/pages/`：首页、Misc、Notes、RSS 与其他页面。

## 新增 Blog

在 `src/content/blog/` 新建 `.md` 或 `.mdx` 文件：

```yaml
---
title: "文章标题"
description: "用于首页、搜索与社交分享的摘要"
publishedAt: "2026-08-21"
updatedAt: "2026-08-22" # 可选
tags: ["Agents", "AI Infra"]
series: "Agent Systems"
featured: false
draft: false
lang: "zh"
---
```

Markdown 原生支持表格、任务列表、目录标题、KaTeX 公式和带语法高亮的代码块。MDX 还可以导入：

- `Callout.astro`：提示、警告、观点框。
- `FlowDiagram.astro`：响应式流程图。
- `ComparisonTabs.astro`：可切换的对比内容。

代码块使用 Expressive Code，可加文件名、行高亮并自带复制按钮：

````md
```python title="train.py" {3,7-10}
def train():
    ...
```
````

## 本地开发

```bash
pnpm install
pnpm dev
pnpm build
node scripts/check-site.mjs
```

`pnpm build` 会先执行 Astro 类型检查，再生成静态站点；链接检查脚本会验证所有生成页面中的本地资源与站内链接。

## 发布

推送到 `main` 后，`.github/workflows/deploy.yml` 使用 Astro 官方 GitHub Action 构建并发布 GitHub Pages。仓库名必须保持为 `HlinForest.github.io`。
