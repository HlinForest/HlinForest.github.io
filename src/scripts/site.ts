const root = document.documentElement;
const preference = matchMedia('(prefers-color-scheme: dark)');
const toggle = document.querySelector<HTMLButtonElement>('[data-theme-toggle]');
let manual = false;
try { manual = ['light', 'dark'].includes(localStorage.getItem('theme') || ''); } catch {}
const applyTheme = (theme: 'light' | 'dark') => {
  root.dataset.theme = theme;
  const label = theme === 'dark' ? '切换为浅色主题' : '切换为深色主题';
  toggle?.setAttribute('aria-label', label);
  toggle?.setAttribute('title', label);
  document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]').forEach(meta => meta.content = theme === 'dark' ? '#0a0f1e' : '#f4f5f5');
};
applyTheme(root.dataset.theme === 'dark' ? 'dark' : 'light');
toggle?.addEventListener('click', () => {
  const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
  manual = true;
  applyTheme(next);
  try { localStorage.setItem('theme', next); } catch {}
});
preference.addEventListener('change', event => { if (!manual) applyTheme(event.matches ? 'dark' : 'light'); });
if ('IntersectionObserver' in window) {
  const sentinel = document.querySelector('[data-header-sentinel]');
  if (sentinel) new IntersectionObserver(([entry]) => document.querySelector('.site-header')?.classList.toggle('is-scrolled', !entry.isIntersecting)).observe(sentinel);
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.remove('reveal-pending'); observer.unobserve(entry.target); }
    }), { threshold: 0.08 });
    document.querySelectorAll('.reveal').forEach(element => { element.classList.add('reveal-pending'); observer.observe(element); });
  }
}
document.querySelectorAll<HTMLAnchorElement>('.heading-anchor').forEach(anchor => {
  anchor.addEventListener('click', () => {
    // Preserve native anchor navigation when the clipboard is unavailable.
    navigator.clipboard?.writeText(anchor.href).then(() => {
      const status = document.querySelector('[data-copy-status]');
      if (status) status.textContent = '本节链接已复制';
    }).catch(() => {});
  });
});
