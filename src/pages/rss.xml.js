import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { byNewest, postUrl, publishedOnly } from '../utils/posts';
import { site } from '../data/site';

export async function GET(context) {
  const posts = (await getCollection('blog')).filter(publishedOnly).sort(byNewest);
  return rss({
    title: site.title,
    description: site.description,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.publishedAt,
      link: postUrl(post),
      categories: post.data.tags,
    })),
  });
}
