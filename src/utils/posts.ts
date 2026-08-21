import type { CollectionEntry } from 'astro:content';

export type BlogPost = CollectionEntry<'blog'>;

export const byNewest = (a: BlogPost, b: BlogPost) =>
  b.data.publishedAt.valueOf() - a.data.publishedAt.valueOf();

export const publishedOnly = (post: BlogPost) => !post.data.draft;

export const postUrl = (post: BlogPost) => `/notes/${post.id}/`;

export function readingMinutes(body = '') {
  const chinese = (body.match(/[\u3400-\u9fff]/g) || []).length;
  const words = (body.replace(/[\u3400-\u9fff]/g, ' ').match(/[A-Za-z0-9_]+/g) || []).length;
  return Math.max(1, Math.ceil(chinese / 420 + words / 220));
}

export const formatDate = (date: Date, lang = 'zh-CN') =>
  new Intl.DateTimeFormat(lang, { year: 'numeric', month: 'short', day: 'numeric' }).format(date);
