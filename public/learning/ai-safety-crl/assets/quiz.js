document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".quiz").forEach((quiz) => {
    const button = quiz.querySelector("button");
    const feedback = quiz.querySelector(".feedback");
    if (!button || !feedback) return;

    button.addEventListener("click", () => {
      const selected = quiz.querySelector("input[type='radio']:checked");
      if (!selected) {
        feedback.className = "feedback bad";
        feedback.textContent = "先选一个答案；主动回忆比直接看解析更能留下长期记忆。";
        return;
      }
      const correct = selected.value === quiz.dataset.answer;
      feedback.className = `feedback ${correct ? "good" : "bad"}`;
      feedback.textContent = correct ? quiz.dataset.correct : quiz.dataset.wrong;
    });
  });

  document.querySelectorAll("[data-advice-lab]").forEach((lab) => {
    const slider = lab.querySelector("input[type='range']");
    const value = lab.querySelector("[data-advice-value]");
    const output = lab.querySelector("[data-advice-output]");
    const update = () => {
      const k = Number(slider.value);
      const baseline = 8;
      const trust = 3 + 2 * k;
      const robust = Math.min(baseline, 3 + Math.ceil(1.25 * k));
      value.textContent = k;
      output.innerHTML = `无建议基线：<b>${baseline}</b> 次干预；盲信建议：<b>${trust}</b> 次；测试后行动：<b>${robust}</b> 次。`;
    };
    slider.addEventListener("input", update);
    update();
  });

  document.querySelectorAll("[data-budget-lab]").forEach((lab) => {
    const sliders = [...lab.querySelectorAll("input[type='range']")];
    const totalNode = lab.querySelector("[data-budget-total]");
    const output = lab.querySelector("[data-budget-output]");
    const update = () => {
      const allocations = sliders.map((s) => Number(s.value));
      sliders.forEach((s, i) => {
        const n = lab.querySelector(`[data-budget-value='${i}']`);
        if (n) n.textContent = allocations[i];
      });
      const total = allocations.reduce((a, b) => a + b, 0);
      totalNode.textContent = total;
      if (total !== 10) {
        output.innerHTML = `当前用了 <b>${total}</b>/10 份资源。请把总数调到 10。`;
        return;
      }
      const impacts = [90, 60, 80];
      const detect = [0.55, 0.8, 0.35];
      const risks = allocations.map((a, i) => impacts[i] * (1 - detect[i] * (1 - Math.exp(-a / 2))));
      const labels = ["训练数据投毒", "越狱评测", "供应链失效"];
      const worst = risks.indexOf(Math.max(...risks));
      output.innerHTML = `剩余风险约为 ${risks.map((r, i) => `${labels[i]} <b>${r.toFixed(1)}</b>`).join(" · ")}。<br>若对手观察到你的固定策略，它会优先攻击：<b>${labels[worst]}</b>。`;
    };
    sliders.forEach((s) => s.addEventListener("input", update));
    update();
  });
});
