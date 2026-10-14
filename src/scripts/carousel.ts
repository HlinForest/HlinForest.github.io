document.querySelectorAll<HTMLElement>('[data-carousel]').forEach(root => {
  const stage = root.querySelector<HTMLElement>('[data-carousel-stage]')!;
  const templates = [...stage.querySelectorAll<HTMLTemplateElement>('[data-slide-template]')];
  const first = stage.querySelector<HTMLElement>('[data-slide]')!;
  const slides = new Map<number, HTMLElement>([[0, first]]);
  const records = [first, ...templates].sort((a, b) => Number(a.dataset.index) - Number(b.dataset.index));
  const count = records.length;
  const groups = [...root.querySelectorAll<HTMLButtonElement>('[data-carousel-group]')];
  let index = 0;
  let inView = false;
  const materialize = (i: number) => {
    let slide = slides.get(i);
    if (!slide) {
      const template = templates.find(item => Number(item.dataset.index) === i)!;
      slide = (template.content.firstElementChild as HTMLElement).cloneNode(true) as HTMLElement;
      slide.inert = true;
      slide.setAttribute('aria-hidden', 'true');
      stage.append(slide);
      slides.set(i, slide);
    }
    return slide;
  };
  const preload = (i: number) => {
    const img = materialize(i).querySelector<HTMLImageElement>('img');
    if (img) img.loading = 'eager';
  };
  const show = (next: number, announce = true) => {
    index = (next + count) % count;
    const current = materialize(index);
    if (announce) preload(index);
    slides.forEach((slide, i) => {
      const active = i === index;
      slide.dataset.active = String(active);
      slide.inert = !active;
      slide.setAttribute('aria-hidden', String(!active));
    });
    groups.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.carouselGroup === current.dataset.group)));
    root.querySelectorAll<HTMLElement>('[data-group-content]').forEach(content => {
      content.hidden = content.dataset.groupContent !== current.dataset.group;
      if (content.hidden && content instanceof HTMLDetailsElement) content.open = false;
    });
    const counter = root.querySelector('[data-carousel-count]');
    if (counter) counter.textContent = `${String(index + 1).padStart(2, '0')} / ${String(count).padStart(2, '0')}`;
    const status = root.querySelector('[data-carousel-status]');
    if (announce && status) status.textContent = `第 ${index + 1} 张，共 ${count} 张：${current.dataset.label}`;
    if (inView && count > 1) preload((index + 1) % count);
  };
  root.querySelector('[data-carousel-prev]')?.addEventListener('click', () => show(index - 1));
  root.querySelector('[data-carousel-next]')?.addEventListener('click', () => show(index + 1));
  groups.forEach(button => button.addEventListener('click', () => {
    const target = records.findIndex(record => record.dataset.group === button.dataset.carouselGroup);
    if (target >= 0) show(target);
  }));
  root.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); show(index + (event.key === 'ArrowRight' ? 1 : -1)); }
  });
  let pointer: { id: number; x: number; y: number } | undefined;
  stage.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.pointerType === 'mouse') return;
    // Capture gestures on images only, preserving taps on credit links below them.
    if (!(event.target instanceof Element) || !event.target.closest('.carousel-image')) return;
    pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
    stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener('pointerup', event => {
    if (!pointer || pointer.id !== event.pointerId) return;
    const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
    pointer = undefined;
    if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy) * 1.3) show(index + (dx < 0 ? 1 : -1));
  });
  stage.addEventListener('pointercancel', () => { pointer = undefined; });
  root.querySelector('[data-carousel-controls]')?.removeAttribute('hidden');
  root.dataset.enhanced = 'true';
  show(0, false);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      if (inView && count > 1) preload((index + 1) % count);
    }).observe(stage);
  }
});
