(() => {
  const normalize = value => (value || "").trim().toLowerCase().replace(/\s+/g, " ");

  document.querySelectorAll(".quiz").forEach(quiz => {
    const button = quiz.querySelector("button[data-check]");
    const feedback = quiz.querySelector(".feedback");
    if (!button || !feedback) return;
    button.addEventListener("click", () => {
      let correct = 0;
      let total = 0;
      quiz.querySelectorAll("fieldset[data-answer]").forEach(fieldset => {
        total += 1;
        const chosen = fieldset.querySelector("input:checked");
        const answer = fieldset.dataset.answer;
        fieldset.querySelectorAll("label").forEach(label => {
          label.style.borderColor = "";
          label.style.background = "";
        });
        if (chosen && normalize(chosen.value) === normalize(answer)) {
          correct += 1;
          chosen.closest("label").style.borderColor = "var(--good)";
          chosen.closest("label").style.background = "#eef8f1";
        } else if (chosen) {
          chosen.closest("label").style.borderColor = "var(--bad)";
          chosen.closest("label").style.background = "#fff1ef";
        }
      });
      feedback.textContent = `${correct}/${total}。${correct === total ? "全部正确，可以继续。" : "先解释错因，再展开答案。"}`;
      feedback.className = `feedback ${correct === total ? "ok" : "no"}`;
    });
  });

  document.querySelectorAll("[data-reveal]").forEach(button => {
    button.addEventListener("click", () => {
      const target = document.querySelector(button.dataset.reveal);
      if (!target) return;
      target.hidden = !target.hidden;
      button.textContent = target.hidden ? "显示提示" : "隐藏提示";
    });
  });

  const key = `pytorch-course:${location.pathname}`;
  const checkbox = document.querySelector("[data-complete]");
  if (checkbox) {
    checkbox.checked = localStorage.getItem(key) === "done";
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) localStorage.setItem(key, "done");
      else localStorage.removeItem(key);
    });
  }

  const setNodeState = (root, activeIndex, completedThrough = activeIndex - 1) => {
    root.querySelectorAll(".simulator-node").forEach((node, index) => {
      node.classList.toggle("is-active", index === activeIndex);
      node.classList.toggle("is-done", index <= completedThrough);
    });
  };

  const bindSteppedSimulator = (root, steps, renderStep) => {
    const previous = root.querySelector("[data-sim-prev]");
    const next = root.querySelector("[data-sim-next]");
    const reset = root.querySelector("[data-sim-reset]");
    let index = 0;

    const render = () => {
      renderStep(steps[index], index);
      const counter = root.querySelector("[data-sim-counter]");
      if (counter) counter.textContent = `${index + 1} / ${steps.length}`;
      if (previous) previous.disabled = index === 0;
      if (next) {
        next.disabled = index === steps.length - 1;
        next.textContent = steps[index].next || "下一步";
      }
    };

    if (previous) previous.addEventListener("click", () => { index = Math.max(0, index - 1); render(); });
    if (next) next.addEventListener("click", () => { index = Math.min(steps.length - 1, index + 1); render(); });
    if (reset) reset.addEventListener("click", () => { index = 0; render(); });
    render();
  };

  document.querySelectorAll('[data-simulator="autograd"]').forEach(root => {
    const steps = [
      { active: 0, values: ["x=2, y=5", "尚未计算", "尚未计算", "w.grad=None, b.grad=None"], status: "叶 tensor 已就绪。前向执行时，PyTorch 会同时计算数值并记录运算关系。", next: "执行 forward" },
      { active: 1, values: ["x=2, y=5", "ŷ = 2", "尚未计算", "w.grad=None, b.grad=None"], status: "ŷ = wx+b = 1×2+0 = 2。这个节点记住了它由乘法和加法得到。", next: "计算 loss" },
      { active: 2, values: ["x=2, y=5", "ŷ = 2", "L = 9", "w.grad=None, b.grad=None"], status: "L=(ŷ−y)²=9。现在得到一个标量，可以从它启动反向传播。", next: "启动 backward" },
      { active: 3, values: ["x=2, y=5", "ŷ = 2", "∂L/∂ŷ = −6", "w.grad=None, b.grad=None"], status: "从 loss 向后：∂L/∂ŷ = 2(ŷ−y) = −6。上游梯度已经到达预测节点。", next: "传播到参数" },
      { active: 4, values: ["x=2, y=5", "∂ŷ/∂w=2", "∂L/∂ŷ = −6", "w.grad=−12, b.grad=−6"], status: "链式法则完成：∂L/∂w=(−6)×2=−12，∂L/∂b=(−6)×1=−6；梯度累积到叶参数。", next: "反向完成" }
    ];
    const fields = Array.from(root.querySelectorAll("[data-sim-value]"));
    bindSteppedSimulator(root, steps, (step, index) => {
      setNodeState(root, step.active, index - 1);
      fields.forEach((field, fieldIndex) => { field.textContent = step.values[fieldIndex]; });
      root.querySelector("[data-sim-status]").textContent = step.status;
    });
  });

  document.querySelectorAll('[data-simulator="optimizer"]').forEach(root => {
    const steps = [
      { active: 0, state: ["1.00", "None", "—", "—"], status: "初始状态：参数 w=1，梯度槽为空。", next: "1 · forward" },
      { active: 0, state: ["1.00", "None", "2.00", "—"], status: "forward 读取输入与当前参数，产生预测 ŷ=2，并建立计算图。", next: "2 · loss" },
      { active: 1, state: ["1.00", "None", "2.00", "9.00"], status: "loss 把预测与目标压成标量 L=9，参数仍未改变。", next: "3 · zero_grad" },
      { active: 2, state: ["1.00", "None", "2.00", "9.00"], status: "zero_grad 清理上一轮的梯度；本例原本就是 None。", next: "4 · backward" },
      { active: 3, state: ["1.00", "−12.00", "2.00", "9.00"], status: "backward 计算梯度并写入 w.grad=−12，但不会更新 w。", next: "5 · step" },
      { active: 4, state: ["2.20", "−12.00", "2.00", "9.00"], status: "step 读取 w.grad，以 lr=0.1 更新 w：1−0.1×(−12)=2.2。注意梯度仍未自动清除。", next: "本轮完成" }
    ];
    const fields = Array.from(root.querySelectorAll("[data-sim-value]"));
    bindSteppedSimulator(root, steps, step => {
      setNodeState(root, step.active, step.active - 1);
      fields.forEach((field, fieldIndex) => { field.textContent = step.state[fieldIndex]; });
      root.querySelector("[data-sim-status]").textContent = step.status;
    });
  });

  document.querySelectorAll('[data-simulator="train-eval"]').forEach(root => {
    const trainLoss = [0.82, 0.52, 0.31, 0.19, 0.12, 0.09];
    const validationLoss = [0.86, 0.58, 0.38, 0.26, 0.21, 0.20];
    const phases = [
      { active: 0, button: "执行训练前向", status: "训练模式：读取一个训练 batch，计算预测与 train loss。" },
      { active: 1, button: "执行 backward + step", status: "梯度开启：反向传播并更新参数；只有这个阶段允许改变模型。" },
      { active: 2, button: "执行验证", status: "评估模式：在 inference_mode 下读取验证集，不建立梯度图、不更新参数。" },
      { active: 3, button: "记录本轮指标", status: "把 train/validation loss 写入 history，形成可诊断的跨 epoch 证据。" }
    ];
    const next = root.querySelector("[data-sim-next]");
    const reset = root.querySelector("[data-sim-reset]");
    const trainPath = root.querySelector("[data-train-line]");
    const validationPath = root.querySelector("[data-validation-line]");
    const pointsLayer = root.querySelector("[data-chart-points]");
    let epoch = 0;
    let phase = 0;
    let recorded = 0;

    const chart = root.querySelector(".loss-chart");
    const chartFrame = root.querySelector("[data-chart-frame]");
    const chartGrid = root.querySelector("[data-chart-grid]");
    const xAxisLabel = root.querySelector("[data-x-axis-label]");
    const coordinates = (values, count, width) => {
      const left = 64;
      const right = width - 28;
      const spacing = (right - left) / (values.length - 1);
      return values.slice(0, count).map((value, index) => ({
        x: left + index * spacing,
        y: 180 - value * 140
      }));
    };
    const pathFrom = points => points.map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`).join(" ");
    const circle = (point, className) => {
      const node = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      node.setAttribute("cx", point.x);
      node.setAttribute("cy", point.y);
      node.setAttribute("r", "5");
      node.setAttribute("class", className);
      return node;
    };

    const drawChart = () => {
      const width = Math.max(280, Math.round(chart.clientWidth || 600));
      const right = width - 28;
      chart.setAttribute("viewBox", `0 0 ${width} 220`);
      chartFrame.setAttribute("width", String(width - 80));
      chartGrid.setAttribute("x2", String(right));
      xAxisLabel.setAttribute("x", String(width / 2));
      const trainPoints = coordinates(trainLoss, recorded, width);
      const validationPoints = coordinates(validationLoss, recorded, width);
      trainPath.setAttribute("d", pathFrom(trainPoints));
      validationPath.setAttribute("d", pathFrom(validationPoints));
      pointsLayer.replaceChildren();
      trainPoints.forEach(point => pointsLayer.appendChild(circle(point, "train-point")));
      validationPoints.forEach(point => pointsLayer.appendChild(circle(point, "validation-point")));
    };
    const render = () => {
      const complete = recorded === trainLoss.length;
      setNodeState(root, complete ? -1 : phases[phase].active, phase - 1);
      root.querySelector("[data-sim-counter]").textContent = complete ? "6 / 6 epochs" : `Epoch ${epoch + 1} / 6`;
      root.querySelector("[data-sim-status]").textContent = complete
        ? "六轮完成：两条损失曲线同时下降，当前没有出现训练/验证分叉。"
        : phases[phase].status;
      next.disabled = complete;
      next.textContent = complete ? "模拟完成" : phases[phase].button;
      drawChart();
    };
    next.addEventListener("click", () => {
      if (recorded === trainLoss.length) return;
      if (phase < phases.length - 1) phase += 1;
      else {
        recorded += 1;
        epoch += 1;
        phase = 0;
      }
      render();
    });
    reset.addEventListener("click", () => { epoch = 0; phase = 0; recorded = 0; render(); });
    if ("ResizeObserver" in window) new ResizeObserver(drawChart).observe(chart);
    render();
  });
})();
