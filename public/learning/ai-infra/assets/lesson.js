const courseKey = 'aiinfra-foundations-progress-v1';
const saved = JSON.parse(localStorage.getItem(courseKey) || '{}');

document.querySelectorAll('[data-quiz]').forEach((quiz) => {
  const feedback = quiz.querySelector('.feedback');
  quiz.querySelectorAll('button').forEach((button) => {
    button.addEventListener('click', () => {
      const correct = button.dataset.correct === 'true';
      quiz.querySelectorAll('button').forEach((item) => {
        item.setAttribute('aria-pressed', String(item === button));
      });
      feedback.textContent = correct
        ? `正确。${quiz.dataset.why}`
        : `再想一步：${quiz.dataset.hint}`;
      feedback.className = `feedback ${correct ? 'good' : 'bad'}`;
    });
  });
});

document.querySelectorAll('[data-reveal]').forEach((button) => {
  button.addEventListener('click', () => {
    const target = document.getElementById(button.dataset.reveal);
    const hidden = target.hasAttribute('hidden');
    target.toggleAttribute('hidden', !hidden);
    button.setAttribute('aria-expanded', String(hidden));
    button.textContent = hidden ? '收起解析' : '查看解析';
  });
});

const completionButton = document.querySelector('[data-complete]');
if (completionButton) {
  const lessonId = completionButton.dataset.complete;
  function renderCompletion() {
    const done = Boolean(saved[lessonId]);
    completionButton.setAttribute('aria-pressed', String(done));
    completionButton.textContent = done ? '已通过本单元' : '标记为已通过';
  }
  completionButton.addEventListener('click', () => {
    saved[lessonId] = !saved[lessonId];
    localStorage.setItem(courseKey, JSON.stringify(saved));
    renderCompletion();
  });
  renderCompletion();
}

const progressRoot = document.querySelector('[data-course-progress]');
if (progressRoot) {
  const items = [...document.querySelectorAll('[data-course-item]')];
  let completed = 0;
  items.forEach((item) => {
    const done = Boolean(saved[item.dataset.courseItem]);
    item.classList.toggle('is-complete', done);
    if (done) completed += 1;
  });
  const value = progressRoot.querySelector('progress');
  const label = progressRoot.querySelector('[data-progress-label]');
  value.max = items.length;
  value.value = completed;
  label.textContent = `${completed} / ${items.length} 单元已通过`;
}
