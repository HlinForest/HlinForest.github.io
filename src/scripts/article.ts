const progress = document.querySelector<HTMLElement>('[data-reading-progress]');
const article = document.querySelector<HTMLElement>('[data-article-body]');
const topButton = document.querySelector<HTMLElement>('[data-back-top]');
const sentinel = document.querySelector('[data-back-top-sentinel]');
if (sentinel && 'IntersectionObserver' in window) {
  new IntersectionObserver(([entry]) => { if (topButton) topButton.hidden = entry.isIntersecting; }).observe(sentinel);
}
const updateProgress = () => {
  if (!progress || !article) return;
  const top = article.getBoundingClientRect().top + scrollY;
  const distance = Math.max(1, article.scrollHeight - innerHeight * 0.45);
  progress.style.transform = `scaleX(${Math.max(0, Math.min(1, (scrollY - top + 90) / distance))})`;
};
// Batch the reading indicator with animation frames; no framework state or continuous loop.
let scheduled = false;
const scheduleProgress = () => {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => { updateProgress(); scheduled = false; });
};
document.addEventListener('scroll', scheduleProgress, { passive: true });
addEventListener('resize', scheduleProgress);
if (article) new ResizeObserver(scheduleProgress).observe(article);
updateProgress();
document.querySelector<HTMLButtonElement>('[data-copy-link]')?.addEventListener('click', async event => {
  const button = event.currentTarget as HTMLButtonElement;
  const status = document.querySelector('[data-article-copy-status]');
  try {
    await navigator.clipboard.writeText(location.href);
    button.dataset.copied = 'true';
    if (status) status.textContent = '链接已复制';
    setTimeout(() => { delete button.dataset.copied; if (status) status.textContent = ''; }, 1800);
  } catch { if (status) status.textContent = '请从地址栏复制链接'; }
});
const details = document.querySelector<HTMLDetailsElement>('[data-toc-details]');
const desktop = matchMedia('(min-width: 1024px)');
const setTocLayout = () => { if (details) details.open = desktop.matches; };
desktop.addEventListener('change', setTocLayout);
setTocLayout();
const links = [...document.querySelectorAll<HTMLAnchorElement>('[data-toc-link]')];
const sections = links.map(link => document.getElementById(link.dataset.tocLink || '')).filter((section): section is HTMLElement => Boolean(section));
links.forEach(link => link.addEventListener('click', () => {
  // Collapse first so native anchor scrolling uses the final mobile layout.
  if (details && !desktop.matches) details.open = false;
}));
if (sections.length && 'IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    const current = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (!current) return;
    links.forEach(link => {
      const active = link.dataset.tocLink === current.target.id;
      link.dataset.active = String(active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-12% 0px -65% 0px' });
  sections.forEach(section => observer.observe(section));
}
