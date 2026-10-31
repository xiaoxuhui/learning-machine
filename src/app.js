(function () {
  'use strict';
  const { NeuralNetwork, parseNumber } = window.LearningModel;
  const { format, renderNetwork, renderChart } = window.LearningVisuals;
  const model = new NeuralNetwork();
  const $ = id => document.getElementById(id);
  let prediction = null, preview = null, toastTimer, animationTimer;

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
      model.train(x, y); prediction = null; preview = { x, y: model.predict(x) };
      $('train-form').reset();
      message('train-message', `${existed ? '已更新' : '已学会'} ${format(x)} → ${format(y)}，并复习全部样本 600 轮。`, 'success');
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
      prediction = { x, y: model.predict(x) }; preview = prediction;
      const xs = model.samples.map(s => s[0]);
      const outside = x < Math.min(...xs) || x > Math.max(...xs);
      message('predict-message', outside ? '这是训练范围之外的预测，可能不准确。' : '预测完成。参数和样本保持不变。', 'success');
      render();
    } catch (error) { message('predict-message', error.message, 'error'); }
  });
  $('reset').addEventListener('click', () => {
    model.reset(); prediction = null; preview = null;
    clearTimeout(animationTimer); document.querySelector('.machine-panel').classList.remove('learning');
    $('train-form').reset(); $('predict-form').reset();
    document.querySelectorAll('input').forEach(field => field.removeAttribute('aria-invalid'));
    message('train-message', '每学一组，都会一起复习已有样本。');
    message('predict-message', '计算只使用当前模型，不会改变学习结果。');
    setMode('train'); render(); toast('已完全重置，可以开始新一轮学习。'); $('train-x').focus();
  });
  document.querySelectorAll('input').forEach(field => field.addEventListener('input', () => field.removeAttribute('aria-invalid')));
  render();
})();
