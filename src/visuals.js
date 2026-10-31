(function () {
  'use strict';
  const format = (value, digits = 6) => {
    if (!Number.isFinite(value)) return '—';
    if (Object.is(value, -0)) return '0';
    return String(Number(value.toPrecision(digits)));
  };
  const svgText = (x, y, text, attrs = '') => `<text x="${x}" y="${y}" ${attrs}>${text}</text>`;

  function renderNetwork(container, model, preview) {
    const x = preview ? preview.x : 0;
    const h = model.activation(x / model.sx);
    const ys = [44, 84, 124, 164, 204, 244];
    const point = { input: [88, 144], hiddenX: 310, output: [532, 144] };
    let content = '<title>小机器的神经网络</title><desc>一个输入、六个隐藏节点、一个输出。连线表示实际权重，绿色为正、橙色为负；粗细表示绝对值。</desc>';
    content += '<defs><radialGradient id="nodeGlow"><stop stop-color="#a9d881" stop-opacity=".15"/><stop offset="1" stop-color="#a9d881" stop-opacity="0"/></radialGradient></defs>';
    for (let j = 0; j < 6; j++) {
      for (const [weight, ax, ay, bx, by] of [
        [model.w1[j], point.input[0] + 27, point.input[1], point.hiddenX - 14, ys[j]],
        [model.w2[j], point.hiddenX + 14, ys[j], point.output[0] - 27, point.output[1]]
      ]) {
        const size = Math.min(3.6, 0.5 + Math.abs(weight) * 1.9);
        const opacity = Math.min(0.8, 0.2 + Math.abs(weight) * 0.5);
        content += `<path d="M ${ax} ${ay} C ${ax + 76} ${ay},${bx - 76} ${by},${bx} ${by}" fill="none" stroke="${weight >= 0 ? '#a1c786' : '#c3a075'}" stroke-width="${size}" opacity="${opacity}" data-weight="${weight}"><title>权重 ${format(weight)}</title></path>`;
      }
    }
    for (let j = 0; j < 6; j++) {
      const lightness = 19 + Math.abs(h[j]) * 11;
      content += `<circle cx="310" cy="${ys[j]}" r="15" fill="hsl(96 20% ${lightness}%)" stroke="#90ad72" stroke-opacity=".65"/><circle cx="310" cy="${ys[j]}" r="3" fill="#c0dda6" opacity="${0.3 + Math.abs(h[j]) * 0.7}"/><title>节点 ${j + 1}：激活 ${format(h[j])}，偏置 ${format(model.b1[j])}</title>`;
    }
    for (const [cx, cy, text, label] of [[88, 144, preview ? format(preview.x, 4) : 'x', '输入'], [532, 144, preview ? format(preview.y, 4) : 'y', '输出']]) {
      content += `<circle cx="${cx}" cy="${cy}" r="53" fill="url(#nodeGlow)"/><circle cx="${cx}" cy="${cy}" r="31" fill="#263e2b" stroke="#b0d38d" stroke-width="1.2"/><circle cx="${cx}" cy="${cy}" r="26" fill="none" stroke="#b0d38d" stroke-opacity=".13"/>`;
      content += svgText(cx, cy + 5, text, `text-anchor="middle" fill="#d6edb9" font-size="${text.length > 5 ? 10 : text.length > 3 ? 13 : 18}" font-family="Consolas,monospace"`);
      content += svgText(cx, cy + 57, label, 'text-anchor="middle" fill="#a1b48d" font-size="11"');
    }
    container.innerHTML = `<svg viewBox="0 0 620 288" role="img" aria-label="实际权重神经网络">${content}</svg>`;
  }

  function renderChart(container, model, prediction) {
    const W = 560, H = 202, left = 48, right = 19, top = 17, bottom = 30;
    const width = W - left - right, height = H - top - bottom;
    let xmin = -1, xmax = 3, ymin = -1, ymax = 5, curve = [];
    if (model.samples.length) {
      const xs = model.samples.map(s => s[0]);
      if (prediction) xs.push(prediction.x);
      const lo = Math.min(...xs), hi = Math.max(...xs), pad = Math.max((hi - lo) * 0.18, Math.max(Math.abs(lo), Math.abs(hi), 1) * 0.15);
      xmin = Math.max(-1000, lo - pad); xmax = Math.min(1000, hi + pad);
      for (let i = 0; i <= 100; i++) {
        const x = xmin + (xmax - xmin) * i / 100;
        curve.push([x, model.predict(x)]);
      }
      const ys = [...curve.map(s => s[1]), ...model.samples.map(s => s[1])];
      const vlo = Math.min(...ys), vhi = Math.max(...ys), ypad = Math.max((vhi - vlo) * 0.18, Math.abs(vhi) * 0.04, 0.1);
      ymin = vlo - ypad; ymax = vhi + ypad;
    }
    const px = x => left + (x - xmin) / (xmax - xmin) * width;
    const py = y => top + height - (y - ymin) / (ymax - ymin) * height;
    let content = '<title>预测曲线与训练样本</title><desc>横轴为输入，纵轴为输出。绿色曲线来自当前模型，深色圆点为用户提供的样本。</desc>';
    for (let i = 0; i <= 4; i++) {
      const gx = left + width * i / 4, gy = top + height * i / 4;
      content += `<path d="M${gx} ${top}V${top + height} M${left} ${gy}H${W - right}" stroke="#edf0e6" stroke-width="1" ${i > 0 && i < 4 ? 'stroke-dasharray="3 4"' : ''}/>`;
      content += svgText(gx, H - 12, format(xmin + (xmax - xmin) * i / 4, 3), 'text-anchor="middle" fill="#a7ae9b" font-size="9" font-family="Consolas,monospace"');
      content += svgText(left - 9, gy + 3, format(ymax - (ymax - ymin) * i / 4, 3), 'text-anchor="end" fill="#a7ae9b" font-size="9" font-family="Consolas,monospace"');
    }
    if (curve.length) {
      const line = curve.map(([x, y], i) => `${i ? 'L' : 'M'}${px(x).toFixed(2)} ${py(y).toFixed(2)}`).join(' ');
      content += `<defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#b7d38b" stop-opacity=".18"/><stop offset="1" stop-color="#b7d38b" stop-opacity="0"/></linearGradient></defs><path d="${line}L${px(xmax)} ${top + height}H${left}Z" fill="url(#area)"/><path data-curve="true" d="${line}" fill="none" stroke="#8caa6d" stroke-width="2.2"/>`;
      for (const [x, y] of model.samples) content += `<circle data-sample="true" cx="${px(x)}" cy="${py(y)}" r="4" stroke="white" stroke-width="2" fill="#364e35"><title>样本 ${format(x)} → ${format(y)}</title></circle>`;
      if (prediction) content += `<path d="M${px(prediction.x)} ${top}V${top + height}" stroke="#9baf83" stroke-dasharray="3 3"/><circle data-prediction="true" cx="${px(prediction.x)}" cy="${py(prediction.y)}" r="5" stroke="#365b35" stroke-width="2" fill="#d4edb3"><title>预测 ${format(prediction.x)} → ${format(prediction.y)}</title></circle>`;
    } else {
      content += `<rect x="135" y="72" width="307" height="45" rx="8" fill="#fff" fill-opacity=".94"/>${svgText(288, 99, '训练后，规律会在这里显现', 'text-anchor="middle" fill="#a0ac91" font-size="12"')}`;
    }
    content += svgText(W - 5, H - 12, 'x', 'text-anchor="end" fill="#8d9e7d" font-size="10" font-style="italic"');
    content += svgText(left - 9, 9, 'y', 'text-anchor="end" fill="#8d9e7d" font-size="10" font-style="italic"');
    container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="输入和输出的预测曲线">${content}</svg>`;
  }
  window.LearningVisuals = { format, renderNetwork, renderChart };
})();
