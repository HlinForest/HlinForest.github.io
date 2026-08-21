import type { CollectionEntry } from 'astro:content';

export type LearningDoc = CollectionEntry<'learning'>;

export function learningUrl(entry: LearningDoc) {
  const clean = entry.id.replace(/\/(README|index)$/i, '');
  return `/learning/${clean}/`;
}

export function learningTitle(entry: LearningDoc) {
  const data = entry.data as { title?: string };
  if (data.title) return data.title;
  const heading = entry.body?.match(/^#\s+(.+)$/m)?.[1];
  return heading?.replace(/[`*_]/g, '') || entry.id.split('/').at(-1)?.replace(/^\d+-?/, '') || 'Untitled';
}

export const learningTracks = [
  {
    slug: 'cs336',
    area: 'LLM Systems',
    title: 'CS336 中文深度笔记',
    description: '从 tokenizer、Transformer 与训练循环，到 GPU、分布式、Scaling Law、后训练和推理服务。',
    href: '/learning/cs336/',
    count: '18 个章节与实验',
    native: true,
  },
  {
    slug: 'ai-infra',
    area: 'AI Infra',
    title: 'AI Infra 前置基础',
    description: '把 Python、C++、张量、Transformer、GPU 架构与集合通信串成一条可诊断的技术链。',
    href: '/learning/ai-infra/index.html',
    count: '12 单元 + 综合项目',
    native: false,
  },
  {
    slug: 'pytorch',
    area: 'Foundations',
    title: 'PyTorch / ARENA 基础',
    description: '以张量、自动微分、模块、训练循环与实验诊断为主线的渐进式课程。',
    href: '/learning/pytorch/index.html',
    count: '课程门户与 10 讲',
    native: false,
  },
  {
    slug: 'ai-safety-crl',
    area: 'AI Safety',
    title: 'AI Safety × 因果表征学习',
    description: '从因果基础、表征学习与结构发现，进入安全研究问题、实验设计与论文阅读。',
    href: '/learning/ai-safety-crl/index.html',
    count: '28 讲专题课程',
    native: false,
  },
] as const;
