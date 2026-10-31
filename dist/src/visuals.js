(function () {
  'use strict';
  const format = (value, digits = 6) => {
    if (!Number.isFinite(value)) return '—';
    if (Object.is(value, -0)) return '0';
    return String(Number(value.toPrecision(digits)));
  };
  const svgText = (x, y, text, attrs = '') => `<text x="${x}" y="${y}" ${attrs}>${text}</text>`;

  function renderNetwork(container, model, preview) {
    const sizes = [1, ...model.architecture, 1];
    const trace = model.traceNormalized((preview ? preview.x : 0) / model.sx);
    const nodes = sizes.map((size, l) => Array.from({ length: size }, (_, i) => ({
      x: 53 + l * 514 / (sizes.length - 1),
      y: size === 1 ? 140 : 42 + i * 197 / (size - 1),
      r: l === 0 || l === sizes.length - 1 ? 25 : size > 6 ? 11 : 15
    })));
    let content = '<title>实际多层神经网络 ' + sizes.join(' → ') + '</title><desc>每列为真实的一层，连线对应实际权重；绿色为正、橙色为负。隐藏层使用 tanh，输出层为线性变换。</desc>';
    model.layers.forEach((layer, l) => {
      layer.weights.forEach((row, i) => row.forEach((weight, j) => {
        const a = nodes[l][j], b = nodes[l + 1][i];
        const size = Math.min(3.6, 0.5 + Math.abs(weight) * 1.9);
        const opacity = Math.min(0.75, 0.15 + Math.abs(weight) * 0.45);
        const dx = (b.x - a.x) * 0.4;
        content += '<path d="M ' + (a.x + a.r) + ' ' + a.y + ' C ' + (a.x + dx) + ' ' + a.y + ',' + (b.x - dx) + ' ' + b.y + ',' + (b.x - b.r) + ' ' + b.y + '" fill="none" stroke="' + (weight >= 0 ? '#a1c786' : '#c3a075') + '" stroke-width="' + size + '" opacity="' + opacity + '" data-weight="' + weight + '"><title>层 ' + l + ' 节点 ' + (j + 1) + ' → 层 ' + (l + 1) + ' 节点 ' + (i + 1) + '，权重 ' + format(weight) + '</title></path>';
      }));
    });
    nodes.forEach((layer, l) => {
      layer.forEach((node, i) => {
        const endpoint = l === 0 || l === sizes.length - 1;
        const activation = l === 0 ? (preview ? preview.x : 0) : trace[l - 1].outputs[i] * (l === sizes.length - 1 ? model.sy : 1);
        const value = endpoint ? (preview ? format(activation, 4) : l === 0 ? 'x' : 'y') : format(activation, 2);
        content += '<g data-layer="' + l + '" data-node="' + i + '"><circle cx="' + node.x + '" cy="' + node.y + '" r="' + node.r + '" fill="' + (endpoint ? '#2e4530' : '#263c2c') + '" stroke="#a4c583" stroke-opacity="' + (endpoint ? 1 : 0.7) + '"/><title>' + (l === 0 ? '输入' : l === sizes.length - 1 ? '输出' : '隐藏层 ' + l + ' 节点 ' + (i + 1)) + '：' + format(activation) + '</title>';
        content += svgText(node.x, node.y + (endpoint ? 5 : 3), value, 'text-anchor="middle" fill="#d3e7b6" font-size="' + (endpoint ? value.length > 4 ? 10 : 16 : sizes[l] > 6 ? 6 : 8) + '" font-family="Consolas,monospace"') + '</g>';
      });
      const label = l === 0 ? '输入' : l === sizes.length - 1 ? '输出' : '隐藏层 ' + l + ' · ' + sizes[l];
      content += svgText(layer[0].x, 278, label, 'text-anchor="middle" fill="#a1b48d" font-size="10"');
    });
    container.innerHTML = '<svg viewBox="0 0 620 294" role="img" aria-label="实际网络结构 ' + sizes.join(' → ') + '">' + content + '</svg>';
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
