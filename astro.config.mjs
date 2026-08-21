import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import expressiveCode from 'astro-expressive-code';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import path from 'node:path';

function rewriteLearningLinks() {
  return (tree, file) => {
    const source = String(file.history?.[0] || '').replaceAll('\\', '/');
    const marker = '/src/content/learning/';
    if (!source.includes(marker)) return;
    const current = source.split(marker)[1];
    const base = path.posix.dirname(current);

    const walk = (node) => {
      if (node?.tagName === 'a' && typeof node.properties?.href === 'string') {
        const href = node.properties.href;
        const [filePart, hash] = href.split('#');
        if (filePart && !filePart.match(/^[a-z]+:/i) && filePart.toLowerCase().endsWith('.md')) {
          let target = path.posix.normalize(path.posix.join(base, decodeURI(filePart))).replace(/\.md$/i, '');
          target = target.replace(/\/(README|index)$/i, '');
          target = target.toLowerCase();
          node.properties.href = `/learning/${target}/${hash ? `#${hash}` : ''}`;
        }
      }
      if (Array.isArray(node?.children)) node.children.forEach(walk);
    };
    walk(tree);
  };
}

export default defineConfig({
  site: 'https://hlinforest.github.io',
  output: 'static',
  integrations: [
    expressiveCode({
      themes: ['github-light', 'github-dark'],
      styleOverrides: {
        borderRadius: '14px',
        frames: { frameBoxShadowCssValue: 'none' },
      },
    }),
    mdx(),
    sitemap(),
  ],
  markdown: {
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [
      rehypeKatex,
      rehypeSlug,
      rewriteLearningLinks,
      [rehypeAutolinkHeadings, {
        behavior: 'append',
        properties: { className: ['heading-anchor'], ariaLabel: '复制本节链接' },
        content: { type: 'text', value: '#' },
      }],
    ],
  },
});
