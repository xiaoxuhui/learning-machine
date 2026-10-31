(function () {
  'use strict';
  const { NeuralNetwork, parseNumber, TARGET_LOSS } = window.LearningModel;
  const { format, renderNetwork, renderChart } = window.LearningVisuals;
  const model = new NeuralNetwork();
  const $ = id => document.getElementById(id);
  let prediction = null, preview = null, traceX = 0, toastTimer, animationTimer;

  function renderMappings() {
    const trace = model.forwardTrace(traceX), equations = model.equations();
    $('layer-mappings').replaceChildren();
    const groups = [
      { title: '输入归一化', equations: equations.filter(e => e.layer === 0), values: [trace.normalizedX] },
      ...trace.layers.map((layer, l) => ({ title: l === trace.layers.length - 1 ? '输出层 · 线性变换' : `隐藏层 ${l + 1} · tanh`, equations: equations.filter(e => e.layer === l + 1), values: layer.outputs })),
      { title: '还原输出尺度', equations: equations.filter(e => e.layer === model.layers.length + 1), values: [trace.output] }
    ];
    for (const [index, group] of groups.entries()) {
      const block = document.createElement('section'); block.className = 'layer-block'; block.dataset.mappingLayer = index;
      const heading = document.createElement('h3'); heading.textContent = `${String(index).padStart(2, '0')} / ${group.title}`; block.appendChild(heading);
      group.equations.forEach((equation, i) => {
        const row = document.createElement('div'); row.className = 'equation-row';
        const formula = document.createElement('code'); formula.textContent = `${equation.name} = ${equation.expression}`;
        const output = document.createElement('output'); output.textContent = `= ${format(group.values[i])}`;
        row.append(formula, output); block.appendChild(row);
      });
      $('layer-mappings').appendChild(block);
    }
    $('trace-x').value = format(traceX);
    $('trace-x').removeAttribute('aria-invalid');
    $('trace-message').textContent = `代入 x = ${format(traceX)}；每层结果使用完整参数计算，方程显示 6 位有效数字。${model.samples.length ? '' : '当前显示尚未训练的初始参数。'}`;
  }

  function renderFunction() {
    const out = model.equations().find(e => e.name === 'v');
    $('function-short').textContent = `f(x) = ${format(model.sy)} × (${out.expression})`;
    $('function-status').textContent = model.samples.length ? `${model.architecture.length} 个隐藏层 · 当前参数` : '未训练 · 初始函数';
    if ($('full-function').open) $('function-full').textContent = `f(x) = ${model.functionExpression()}`;
    else $('function-full').textContent = '';
  }

  function message(id, text, type = '') {
    $(id).textContent = text; $(id).className = `message ${type}`;
  }
  function validate(id, messageId) {
    const field = $(id), value = parseNumber(field.value);
    field.setAttribute('aria-invalid', String(value === null));
    if (value === null) {
      message(messageId, '请输入 -1000 到 1000 之间的有效数字。', 'error');
      field.focus();
    }
    return value;
  }
  function setMode(mode, focus = false) {
    const isTrain = mode === 'train';
    $('train-pane').hidden = !isTrain; $('predict-pane').hidden = isTrain;
    $('train-mode').setAttribute('aria-pressed', String(isTrain));
    $('predict-mode').setAttribute('aria-pressed', String(!isTrain));
    if (focus) $(isTrain ? 'train-x' : 'predict-x').focus();
  }
  function render() {
    const count = model.samples.length;
    $('sample-count').textContent = count;
    $('sample-badge').textContent = `${count} 组`;
    $('epoch-count').textContent = model.epochs.toLocaleString('en-US');
    $('loss-value').textContent = model.loss === null ? '—' : format(model.loss, 2);
    $('loss-unit').textContent = model.loss === null ? '等待训练' : '';
    $('machine-status').textContent = count ? `已学习 ${count} 组样本` : '等待第一组样本';
    $('structure').value = model.structureMode;
    $('architecture-label').textContent = [1, ...model.architecture, 1].join(' → ');
    $('architecture-caption').textContent = `${model.architecture.length} 个隐藏层 · ${model.architecture.reduce((sum, n) => sum + n, 0)} 个隐藏节点`;
    $('structure-info').textContent = !count
      ? model.structureMode === 'auto' ? '自动模式从小网络开始；误差较大时会尝试更宽或更深的网络。也可以手动选择多层。' : '已选择真实多层结构，训练后将使用这套网络的实际函数。'
      : model.structureMode === 'auto'
        ? `本次比较 ${model.trials.length} 种结构，选用 ${[1, ...model.architecture, 1].join(' → ')}。${model.loss / model.sy ** 2 <= TARGET_LOSS ? '已达到拟合目标。' : '当前保留误差最低的结构，尚未达到拟合目标。'}`
        : `手动结构 ${[1, ...model.architecture, 1].join(' → ')}，每次重新学习全部样本。`;
    $('sample-empty').hidden = count > 0;
    $('sample-table-wrap').hidden = count === 0;
    $('sample-rows').replaceChildren();
    model.samples.forEach(([x, y], i) => {
      const row = document.createElement('tr');
      for (const value of [String(i + 1).padStart(2, '0'), format(x), format(y), format(model.predict(x))]) {
        const cell = document.createElement('td'); cell.textContent = value; row.appendChild(cell);
      }
      $('sample-rows').appendChild(row);
    });
    $('prediction').textContent = prediction ? format(prediction.y) : '?';
    $('prediction-note').textContent = prediction
      ? `输入 ${format(prediction.x)} 的近似输出`
      : count ? '准备好了，输入一个数字试试吧。' : '先训练，再见证它的第一次预测。';
    renderNetwork($('network'), model, preview);
    renderChart($('chart'), model, prediction);
    renderFunction(); renderMappings();
  }
  function toast(text) {
    clearTimeout(toastTimer); $('toast').textContent = text; $('toast').hidden = false;
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 2800);
  }
  function fillExample(x, y) {
    setMode('train'); $('train-x').value = x; $('train-y').value = y;
    $('train-x').setAttribute('aria-invalid', 'false'); $('train-y').setAttribute('aria-invalid', 'false');
    message('train-message', '示例已填好，点击“开始学习”教给它。');
    $('train-x').focus();
  }
  $('train-mode').addEventListener('click', () => setMode('train'));
  $('predict-mode').addEventListener('click', () => setMode('predict'));
  document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => fillExample(...button.dataset.example.split(','))));
  $('try-example').addEventListener('click', () => { fillExample('0', '0'); $('train-mode').scrollIntoView({ block: 'nearest' }); });
  $('train-form').addEventListener('submit', event => {
    event.preventDefault();
    const x = validate('train-x', 'train-message'); if (x === null) return;
    const y = validate('train-y', 'train-message'); if (y === null) return;
    try {
      const existed = model.samples.some(s => s[0] === x);
      model.train(x, y); prediction = null; preview = { x, y: model.predict(x) }; traceX = x;
      $('train-form').reset();
      message('train-message', `${existed ? '已更新' : '已学习'} ${format(x)} → ${format(y)}，执行 ${model.lastRounds} 轮；当前 ${model.architecture.length} 个隐藏层。`, 'success');
      message('predict-message', '模型已更新，试着计算一个新数字吧。');
      render();
      const panel = document.querySelector('.machine-panel');
      clearTimeout(animationTimer); panel.classList.remove('learning'); void panel.offsetWidth;
      panel.classList.add('learning'); animationTimer = setTimeout(() => panel.classList.remove('learning'), 800);
      $('train-x').focus();
    } catch (error) { message('train-message', error.message, 'error'); }
  });
  $('predict-form').addEventListener('submit', event => {
    event.preventDefault();
    const x = validate('predict-x', 'predict-message'); if (x === null) return;
    try {
      prediction = { x, y: model.predict(x) }; preview = prediction; traceX = x;
      const xs = model.samples.map(s => s[0]);
      const outside = x < Math.min(...xs) || x > Math.max(...xs);
      message('predict-message', outside ? '这是训练范围之外的预测，可能不准确。' : '预测完成。参数和样本保持不变。', 'success');
      render();
    } catch (error) { message('predict-message', error.message, 'error'); }
  });
  $('reset').addEventListener('click', () => {
    model.reset(); prediction = null; preview = null; traceX = 0;
    clearTimeout(animationTimer); document.querySelector('.machine-panel').classList.remove('learning');
    $('train-form').reset(); $('predict-form').reset();
    document.querySelectorAll('input').forEach(field => field.removeAttribute('aria-invalid'));
    message('train-message', '每学一组，都会一起复习已有样本。');
    message('predict-message', '计算只使用当前模型，不会改变学习结果。');
    setMode('train'); render(); toast('已完全重置，可以开始新一轮学习。'); $('train-x').focus();
  });
  $('structure').addEventListener('change', () => {
    model.setStructure($('structure').value); prediction = null;
    preview = model.samples.length ? { x: traceX, y: model.predict(traceX) } : null;
    render();
    message('predict-message', '网络结构已改变，请重新计算。');
    toast(model.samples.length ? '已更换真实结构，并重新训练已有样本。' : '网络结构已改变，图中每一层都参与实际计算。');
  });
  $('full-function').addEventListener('toggle', renderFunction);
  $('trace-form').addEventListener('submit', event => {
    event.preventDefault(); const value = parseNumber($('trace-x').value);
    $('trace-x').setAttribute('aria-invalid', String(value === null));
    if (value === null) { $('trace-message').textContent = '请输入 -1000 到 1000 之间的有效数字。'; return; }
    traceX = value;
    preview = { x: value, y: model.forwardTrace(value).output };
    renderMappings(); renderNetwork($('network'), model, preview);
    $('trace-message').textContent = `代入 x = ${format(value)}；每层结果使用完整参数计算，方程显示 6 位有效数字。${model.samples.length ? '' : '当前显示尚未训练的初始参数。'}`;
  });
  document.querySelectorAll('input').forEach(field => field.addEventListener('input', () => field.removeAttribute('aria-invalid')));
  render();
})();
