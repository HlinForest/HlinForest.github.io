import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import expressiveCode from 'astro-expressive-code';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

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
      [rehypeAutolinkHeadings, {
        behavior: 'append',
        properties: { className: ['heading-anchor'], ariaLabel: '复制本节链接' },
        content: { type: 'text', value: '#' },
      }],
    ],
  },
});
