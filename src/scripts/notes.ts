const tagButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-filter]')];
const sectionButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-section]')];
const rows = [...document.querySelectorAll<HTMLElement>('[data-tags]')];
let activeTag = 'all';
let activeSection = 'all';
const applyFilters = (syncUrl = true) => {
  tagButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.filter === activeTag)));
  sectionButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.section === activeSection)));
  let count = 0;
  rows.forEach(row => {
    const tags: string[] = JSON.parse(row.dataset.tags || '[]');
    const visible = (activeTag === 'all' || tags.includes(activeTag)) && (activeSection === 'all' || row.dataset.series === activeSection);
    row.hidden = !visible;
    if (visible) count++;
  });
  document.querySelector('[data-empty-state]')?.toggleAttribute('hidden', count > 0);
  document.querySelector('[data-active-filters]')?.toggleAttribute('hidden', activeTag === 'all' && activeSection === 'all');
  const description = document.querySelector('[data-filter-description]');
  if (description) description.textContent = [activeSection !== 'all' ? '分区：' + activeSection : '', activeTag !== 'all' ? '标签：' + activeTag : ''].filter(Boolean).join(' / ');
  const status = document.querySelector('[data-filter-status]');
  if (status) status.textContent = `显示 ${count} 篇笔记`;
  if (syncUrl) {
    const url = new URL(location.href);
    for (const [key, value] of [['tag', activeTag], ['section', activeSection]]) {
      if (value === 'all') url.searchParams.delete(key); else url.searchParams.set(key, value);
    }
    history.replaceState(null, '', url);
  }
};
const readUrl = () => {
  const params = new URLSearchParams(location.search);
  activeTag = params.get('tag') || 'all';
  activeSection = params.get('section') || 'all';
  applyFilters(false);
};
tagButtons.forEach(button => button.addEventListener('click', () => { activeTag = button.dataset.filter || 'all'; applyFilters(); }));
sectionButtons.forEach(button => button.addEventListener('click', () => { activeSection = button.dataset.section || 'all'; applyFilters(); }));
document.querySelector('[data-clear-filters]')?.addEventListener('click', () => { activeTag = activeSection = 'all'; applyFilters(); });
addEventListener('popstate', readUrl);
readUrl();
