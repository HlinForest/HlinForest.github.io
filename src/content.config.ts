import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const blog = defineCollection({
  loader: glob({ base: './src/content/blog', pattern: '**/*.{md,mdx}' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishedAt: z.coerce.date(),
    updatedAt: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false),
    lang: z.enum(['zh', 'en']).default('zh'),
    series: z.string().optional(),
    seriesOrder: z.number().optional(),
  }),
});

const learning = defineCollection({
  loader: glob({
    base: './src/content/learning',
    pattern: '**/*.md',
    generateId: ({ entry }) => entry.replace(/\.md$/i, '').toLowerCase(),
  }),
});

export const collections = { blog, learning };
