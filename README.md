# HlinForest

个人技术博客与学习文档库，发布到 `https://hlinforest.github.io`。

## 内容结构

- `src/content/blog/`：可独立阅读的 Blog，按 `series` 分区、按 `tags` 分类。
- `src/content/learning/cs336/`：原生接入本站阅读器的 CS336 长篇学习文档。
- `public/learning/`：保留现有交互门户的 AI Infra、PyTorch/ARENA、AI Safety × CRL 课程。
- `src/pages/`：首页、文章、归档、学习文档、关于、RSS 与动态内容路由。

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

Markdown 原生支持表格、任务列表、脚注、目录标题、KaTeX 公式和带语法高亮的代码块。MDX 还可以导入：

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
